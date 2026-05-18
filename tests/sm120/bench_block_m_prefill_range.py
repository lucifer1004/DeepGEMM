"""BLOCK_M sweep extended into the prefill regime.

The original sweep (`bench_block_m_full_sweep.py`) covered per-expert em ∈
[1, 256]. The smart-pick heuristic derived from it picks BM=64 for em > 80.
But DSv4-Flash prefill routes O(8000·topk / num_experts_per_rank) tokens per
expert, landing at em ≈ 500-2000 — well outside the original bench range.

This script extends the sweep to em ∈ {32, 64, 96, 128, 256, 512, 1024,
2048, 4096} so we can pick a heuristic that doesn't regress prefill.

Run from inside the workspace venv on an SM120 host:

    cd DeepGEMM && \
        CUDA_VISIBLE_DEVICES=0 \
        PYTHONPATH=$PWD python tests/sm120/bench_block_m_prefill_range.py
"""

import sys
import os
import csv
import time
import argparse
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


SHAPES = [
    (4096, 4096, "n4096_k4096"),
    (7168, 2048, "n7168_k2048"),
    (2048, 7168, "n2048_k7168"),
]
NUM_GROUPS = [32, 64, 128]
EXPECTED_M = [32, 64, 96, 128, 256, 512, 1024, 2048, 4096]
NUM_SMS = [110, 188]
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
        torch.cuda.empty_cache()
        return None
    except Exception as e:
        return {"error": str(e)[:120]}
    recipe, recipe_a, recipe_b = quant_config.get_recipes()
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
        return {"error": str(e)[:120], "m_total": int(m)}
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
    padded_m_per_group = ceil_div(expected_m, block_m) * block_m
    padded_m_total = num_groups * padded_m_per_group
    real_m_total = num_groups * expected_m
    kernel_tflops = 2.0 * padded_m_total * n * k / t_s / 1e12
    real_tflops = 2.0 * real_m_total * n * k / t_s / 1e12
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
    parser = argparse.ArgumentParser(description="SM120 BLOCK_M prefill-range sweep")
    parser.add_argument("--out", default=None)
    parser.add_argument("--num-groups", type=int, nargs="*", default=NUM_GROUPS)
    parser.add_argument("--num-sms", type=int, nargs="*", default=NUM_SMS)
    parser.add_argument("--limit-shapes", type=int, default=None)
    parser.add_argument("--block-m", type=int, nargs="*", default=BLOCK_M_VALUES)
    parser.add_argument("--expected-m", type=int, nargs="*", default=EXPECTED_M)
    args = parser.parse_args()

    assert get_arch_major() == 12

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.out or f".omc/perf/sm120_block_m_prefill_{ts}.csv"
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    print(f"# CSV: {out_path}")

    quant_configs = list(QuantConfig.get_list_from_dtype(torch.float8_e4m3fn))
    shapes = SHAPES[: args.limit_shapes] if args.limit_shapes else SHAPES
    phys = deep_gemm.get_num_sms()

    fields = [
        "ts", "num_sms_set", "phys_sms", "num_groups", "n", "k", "shape_label",
        "expected_m", "quant", "block_m", "k_n_warps", "m_total", "padded_m_total",
        "us", "real_tflops", "kernel_tflops", "pad_eff", "diff", "error",
    ]
    rows = 0
    t0 = time.time()
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for sms in args.num_sms:
            target = min(sms, phys)
            deep_gemm.set_num_sms(target)
            for ng in args.num_groups:
                for n, k, lab in shapes:
                    for em in args.expected_m:
                        for qc in quant_configs:
                            ql = quant_label(qc)
                            for bm in args.block_m:
                                res = bench_one_config(ng, em, n, k, bm, qc)
                                if res is None:
                                    continue  # OOM
                                w.writerow({
                                    "ts": ts, "num_sms_set": target, "phys_sms": phys,
                                    "num_groups": ng, "n": n, "k": k, "shape_label": lab,
                                    "expected_m": em, "quant": ql, "block_m": bm,
                                    "k_n_warps": kn_warps_for(bm),
                                    "m_total": res.get("m_total", -1),
                                    "padded_m_total": res.get("padded_m_total", -1),
                                    "us": res.get("us", float("nan")),
                                    "real_tflops": res.get("real_tflops", float("nan")),
                                    "kernel_tflops": res.get("kernel_tflops", float("nan")),
                                    "pad_eff": res.get("pad_eff", float("nan")),
                                    "diff": res.get("diff", float("nan")),
                                    "error": res.get("error", ""),
                                })
                                f.flush()
                                rows += 1
                                if rows % 20 == 0:
                                    print(f"  ... {rows} rows, {time.time()-t0:.0f}s", flush=True)
    deep_gemm.set_num_sms(0)
    deep_gemm.set_mk_alignment_for_contiguous_layout(128)
    print(f"Done. {rows} rows in {time.time()-t0:.1f}s. CSV: {out_path}")


if __name__ == "__main__":
    sys.exit(main() or 0)
