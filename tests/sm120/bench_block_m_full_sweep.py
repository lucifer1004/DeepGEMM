"""Full-sweep micro-benchmark for the SM120 BLOCK_M heuristic + kNWarps refactor.

Measures wall time, real TFLOPS, kernel TFLOPS, padding efficiency, and the
autotuner's chosen (BLOCK_M, kNWarps) for a Cartesian sweep over:

    BLOCK_M ∈ {64, 96, 128}               (forced via set_mk_alignment + filter)
    num_groups ∈ {32 (TP8), 64 (TP4), 128 (TP2)}
    (n, k) ∈ {(4096,4096), (7168,2048), (2048,7168)}
    expected_m ∈ {1, 6, 32, 48, 64, 65, 72, 80, 88, 96, 104, 112, 120, 127,
                  128, 144, 160, 192, 256}
    num_sms ∈ {110 (RTX PRO 5K Pro Blackwell), 188 (RTX PRO 6K Pro Blackwell)}
    QuantConfig ∈ {FP8 ue8m0, FP8 e4m3, FP4-mixed (FP8 input + MXFP4 weight)}

Output: one CSV row per config at .omc/perf/sm120_block_m_sweep_<ts>.csv.

Run from inside the workspace venv on a SM120 host:

    cd DeepGEMM && python tests/sm120/bench_block_m_full_sweep.py
"""

import sys
import os
import csv
import time
import argparse
import contextlib
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import deep_gemm
from deep_gemm.testing import bench_kineto, calc_diff, get_arch_major
from generators import (
    generate_m_grouped_contiguous,
    MajorTypeAB,
    get_ue8m0_usage,
    KernelType,
    QuantConfig,
    reset_seed,
)


# Sweep axes
SHAPES = [
    # (n, k, label)
    (4096, 4096, "n4096_k4096"),
    (7168, 2048, "n7168_k2048"),
    (2048, 7168, "n2048_k7168"),
]
NUM_GROUPS = [32, 64, 128]
EXPECTED_M = [1, 6, 32, 48, 64, 65, 72, 80, 88, 96, 104, 112, 120, 127,
              128, 144, 160, 192, 256]
NUM_SMS = [110, 188]   # 5K Pro, 6K Pro
BLOCK_M_VALUES = [64, 96, 128]


def ceil_div(a, b):
    return (a + b - 1) // b


def kn_warps_for(bm):
    return 2 if bm % 64 == 0 else 4


def quant_label(qc):
    if qc.is_fp4_a:
        return "FP4-mixed"
    return "FP8-ue8m0" if qc.max_diff() < 0.005 else "FP8-e4m3"


def bench_one_config(num_groups, expected_m, n, k, block_m, quant_config):
    """Force BLOCK_M and measure. Returns (us_median, kernel_tflops, real_tflops,
    pad_eff, m_total, padded_m_total, diff_vs_ref) or None on error.
    Skipped (None) if (BM, alignment, num_groups) combination is invalid."""
    # MGroupedContiguous requires BM | runtime_align. Force runtime_align = BM
    # so the heuristic picks BM (and no other) as the kMTilesPerWarp choice.
    deep_gemm.set_mk_alignment_for_contiguous_layout(block_m)

    use_ue8m0 = get_ue8m0_usage(KernelType.Kernel1D1D)
    disable_ue8m0_cast = not use_ue8m0
    reset_seed()
    try:
        m, a, b, gl, d, ref_d = generate_m_grouped_contiguous(
            num_groups, expected_m, n, k,
            MajorTypeAB.KMajor, MajorTypeAB.KMajor,
            use_ue8m0=use_ue8m0, use_psum_layout=False, quant_config=quant_config,
        )
    except torch.OutOfMemoryError:
        return None
    recipe, recipe_a, recipe_b = quant_config.get_recipes()

    # Warmup + correctness sample
    try:
        deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
            a, b, d, gl,
            disable_ue8m0_cast=disable_ue8m0_cast,
            use_psum_layout=False,
            recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b,
        )
        diff_t = calc_diff(d, ref_d)
        diff = float(diff_t.item()) if hasattr(diff_t, 'item') else float(diff_t)
    except RuntimeError as e:
        # JIT failure or kernel crash — record but don't abort
        return {
            "error": str(e)[:120],
            "m_total": int(m),
        }

    # Timed run via bench_kineto (median over 30 reps)
    t_s = bench_kineto(
        lambda: deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
            a, b, d, gl,
            disable_ue8m0_cast=disable_ue8m0_cast,
            use_psum_layout=False,
            recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b,
        ),
        'gemm_', suppress_kineto_output=True,
    )
    us = t_s * 1e6

    # FLOP accounting
    padded_m_per_group = ceil_div(expected_m, block_m) * block_m
    padded_m_total = num_groups * padded_m_per_group
    real_m_total = num_groups * expected_m
    kernel_flops = 2.0 * padded_m_total * n * k
    real_flops = 2.0 * real_m_total * n * k
    kernel_tflops = kernel_flops / t_s / 1e12
    real_tflops = real_flops / t_s / 1e12
    pad_eff = real_m_total / padded_m_total

    return {
        "error": "",
        "us": us,
        "kernel_tflops": kernel_tflops,
        "real_tflops": real_tflops,
        "pad_eff": pad_eff,
        "m_total": int(m),
        "padded_m_total": padded_m_total,
        "diff": diff,
    }


def main():
    parser = argparse.ArgumentParser(description="SM120 BLOCK_M full sweep")
    parser.add_argument("--out", default=None,
                        help="CSV output path (default .omc/perf/sm120_block_m_sweep_<ts>.csv)")
    parser.add_argument("--num-groups", type=int, nargs="*", default=NUM_GROUPS)
    parser.add_argument("--num-sms", type=int, nargs="*", default=NUM_SMS)
    parser.add_argument("--limit-shapes", type=int, default=None,
                        help="Run only the first N shape entries")
    args = parser.parse_args()

    assert get_arch_major() == 12, f"SM120 only, arch_major={get_arch_major()}"

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.out or f".omc/perf/sm120_block_m_sweep_{ts}.csv"
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    print(f"# CSV output: {out_path}")

    quant_configs = list(QuantConfig.get_list_from_dtype(torch.float8_e4m3fn))
    shapes = SHAPES[: args.limit_shapes] if args.limit_shapes else SHAPES
    phys_sms = deep_gemm.get_num_sms()

    fields = [
        "ts", "num_sms_set", "phys_sms", "num_groups", "n", "k", "shape_label",
        "expected_m", "quant", "block_m", "k_n_warps", "m_total", "padded_m_total",
        "us", "real_tflops", "kernel_tflops", "pad_eff", "diff", "error",
    ]
    rows_written = 0
    t0 = time.time()
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for sms in args.num_sms:
            target_sms = min(sms, phys_sms)
            deep_gemm.set_num_sms(target_sms)
            for ng in args.num_groups:
                for n, k, shape_label in shapes:
                    for em in EXPECTED_M:
                        for qc in quant_configs:
                            qlabel = quant_label(qc)
                            for bm in BLOCK_M_VALUES:
                                result = bench_one_config(ng, em, n, k, bm, qc)
                                if result is None:
                                    continue
                                row = {
                                    "ts": ts,
                                    "num_sms_set": target_sms,
                                    "phys_sms": phys_sms,
                                    "num_groups": ng,
                                    "n": n, "k": k, "shape_label": shape_label,
                                    "expected_m": em,
                                    "quant": qlabel,
                                    "block_m": bm,
                                    "k_n_warps": kn_warps_for(bm),
                                    "m_total": result.get("m_total", -1),
                                    "padded_m_total": result.get("padded_m_total", -1),
                                    "us": result.get("us", float("nan")),
                                    "real_tflops": result.get("real_tflops", float("nan")),
                                    "kernel_tflops": result.get("kernel_tflops", float("nan")),
                                    "pad_eff": result.get("pad_eff", float("nan")),
                                    "diff": result.get("diff", float("nan")),
                                    "error": result.get("error", ""),
                                }
                                writer.writerow(row)
                                f.flush()
                                rows_written += 1
                                if rows_written % 20 == 0:
                                    elapsed = time.time() - t0
                                    print(f"  ... {rows_written} rows, {elapsed:.0f}s elapsed", flush=True)
    deep_gemm.set_num_sms(0)
    deep_gemm.set_mk_alignment_for_contiguous_layout(128)
    elapsed = time.time() - t0
    print(f"Done. {rows_written} rows in {elapsed:.1f}s. CSV at {out_path}")


if __name__ == "__main__":
    sys.exit(main() or 0)
