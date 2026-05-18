"""Micro-benchmark: quantify the BLOCK_M=96 headroom for SM120 grouped FP8/FP4.

Today the SM120 grouped GEMM kernel only supports BLOCK_M ∈ {64, 128} (because
`kMWarps=4 × MMA_M=16 = 64` divisibility, see sm120_fp8_fp4_gemm_1d1d.cuh:90).
For workloads where `expected_m ∈ [65, 127]` per group, the heuristic must
round up to BLOCK_M=128 — wasting up to (128-65)/128 = 49% of compute on
padded rows.

This script measures both achievable choices (BLOCK_M=64 and BLOCK_M=128) at
those shapes, then estimates what BLOCK_M=96 (kernel rewrite required) would
deliver. The estimate uses the padding model:

    effective_TFLOPS(BLOCK_M, expected_m) =
        kernel_TFLOPS × (expected_m / ceil_div(expected_m, BLOCK_M) × BLOCK_M)

i.e. the fraction of the kernel's compute that's actually useful (not padded).
We take the better-performing of {BLOCK_M=64, BLOCK_M=128} as the kernel's
"raw efficiency" reference, then extrapolate BLOCK_M=96 perf assuming the
kernel hits the same raw efficiency at the snugger fit.

Output column meanings:
- "real TFLOPS" — 2·expected_m·n·k / latency. Excludes padding work from
  the FLOP count, so this is what the *user* sees.
- "kernel TFLOPS" — 2·padded_m·n·k / latency, where padded_m = ceil_div(
  expected_m, BLOCK_M) × BLOCK_M. This is the kernel's intrinsic peak; the
  gap between this and 762 TF (RTX PRO 5K Blackwell FP8 peak) is overhead.
- "BM=96 est" — projected real TFLOPS if BLOCK_M=96 were enabled, taking
  the max of the two measurements' "kernel TFLOPS" and dividing by the
  padding ratio with BLOCK_M=96.

Run from inside the dsv4-workspace venv on a SM120 host:

    PYTHONPATH=$PWD/DeepGEMM CUDA_VISIBLE_DEVICES=2 \
        .venv/bin/python DeepGEMM/tests/sm120/bench_block_m_headroom.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import deep_gemm
from deep_gemm.testing import bench_kineto, get_arch_major
from generators import (
    generate_m_grouped_contiguous,
    MajorTypeAB,
    get_ue8m0_usage,
    KernelType,
    QuantConfig,
)


# RTX PRO 5000 Blackwell — the host where the TP=4 bug surfaced. Caps the
# autotuner's view of available SMs so the heuristic's scoring matches.
_RTX_PRO_5000_BLACKWELL_SMS = 110

# DSv4-Flash MoE shapes (per-rank, TP=4: 256 experts / 4 = 64 groups).
# Other entries cover smaller per-rank slices and prefill/decode regimes.
_SHAPES = [
    # (num_groups, n, k, label)
    (64, 4096, 4096, "TP4 dsv4 4096x4096"),
    (64, 7168, 2048, "TP4 dsv4 w2 7168x2048"),
    (64, 2048, 7168, "TP4 dsv4 w1 2048x7168"),
    (32, 4096, 4096, "TP8 dsv4 4096x4096"),
]

# Sweep expected_m through the [65,127] "trapped by 128" band and adjacent
# boundary points so we can see the perf cliff at the boundary.
_EXPECTED_M = [32, 48, 64, 65, 72, 80, 96, 112, 127, 128, 144, 160, 192]

# Valid BLOCK_M choices today (kernel constraint: % 64 == 0)
_BLOCK_M_CHOICES = [64, 128]


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def bench_one(num_groups: int, expected_m: int, n: int, k: int,
              block_m: int, quant_config) -> tuple[float, float]:
    """Run a grouped GEMM at the given BLOCK_M and return (latency_us, kernel_TFLOPS).

    kernel_TFLOPS = the GEMM throughput against padded_m (what the kernel
    actually crunched). Padding-aware "real" perf is computed by the caller.
    """
    use_ue8m0 = get_ue8m0_usage(KernelType.Kernel1D1D)
    disable_ue8m0_cast = not use_ue8m0

    # Force BLOCK_M via the runtime-alignment dial.
    deep_gemm.set_mk_alignment_for_contiguous_layout(block_m)

    m, a, b, gl, d, ref_d = generate_m_grouped_contiguous(
        num_groups, expected_m, n, k,
        MajorTypeAB.KMajor, MajorTypeAB.KMajor,
        use_ue8m0=use_ue8m0,
        use_psum_layout=False,
        quant_config=quant_config,
    )

    recipe, recipe_a, recipe_b = quant_config.get_recipes()
    t = bench_kineto(
        lambda: deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
            a, b, d, gl,
            disable_ue8m0_cast=disable_ue8m0_cast,
            use_psum_layout=False,
            recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b,
        ),
        'gemm_', suppress_kineto_output=True)

    # m is per-group expected_m-padded-up-to-alignment; total padded rows
    # across groups = num_groups * ceil_div(expected_m, block_m) * block_m.
    padded_m_per_group = ceil_div(expected_m, block_m) * block_m
    padded_total = num_groups * padded_m_per_group
    kernel_tflops = 2.0 * padded_total * n * k / t / 1e12
    return t * 1e6, kernel_tflops


def estimate_block_m_96(kernel_tflops_64: float, kernel_tflops_128: float,
                        num_groups: int, expected_m: int, n: int, k: int) -> float:
    """Project real TFLOPS at hypothetical BLOCK_M=96.

    Assumption: a BLOCK_M=96 kernel would achieve similar raw kernel TFLOPS
    to the best of {64, 128} (both are within 1.5× of each other typically).
    Use the larger as a conservative upper bound for the projection.
    """
    raw = max(kernel_tflops_64, kernel_tflops_128)
    padded_m_per_group = ceil_div(expected_m, 96) * 96
    padded_total = num_groups * padded_m_per_group
    real_total = num_groups * expected_m
    # Inverse relation: real = kernel × (real/padded)
    return raw * (real_total / padded_total)


def main() -> int:
    assert get_arch_major() == 12, f"SM120 only, got arch_major={get_arch_major()}"

    # Cap visible SMs to the 5K Pro count so the heuristic's wave-count
    # scoring matches the bug's host. On hosts with fewer SMs this is a no-op.
    phys_sms = deep_gemm.get_num_sms()
    target_sms = min(_RTX_PRO_5000_BLACKWELL_SMS, phys_sms)
    deep_gemm.set_num_sms(target_sms)
    print(f"# Physical SMs: {phys_sms}; bench using set_num_sms({target_sms})")

    # Pick the FP8 quant config (matches DSv4 MoE w1/w2)
    quant_configs = [qc for qc in QuantConfig.get_list_from_dtype(torch.float8_e4m3fn)
                     if not qc.is_fp4_a]
    assert quant_configs, "No FP8 quant configs available"
    qc = quant_configs[0]
    print(f"# QuantConfig: {qc}")
    print()

    for num_groups, n, k, label in _SHAPES:
        print(f"## {label}  (num_groups={num_groups}, n={n}, k={k})")
        print(f"{'em':>5s} | {'BM=64 us':>9s} {'real TF':>8s} {'kern TF':>8s} "
              f"| {'BM=128 us':>10s} {'real TF':>8s} {'kern TF':>8s} "
              f"| {'best real':>9s} {'BM=96 est':>10s} {'gain vs best':>13s}")
        print("-" * 110)

        for em in _EXPECTED_M:
            row = f"{em:>5d} | "
            res = {}
            try:
                for bm in _BLOCK_M_CHOICES:
                    us, kern_tf = bench_one(num_groups, em, n, k, bm, qc)
                    real_total = num_groups * em
                    padded_total = num_groups * ceil_div(em, bm) * bm
                    real_tf = kern_tf * (real_total / padded_total)
                    res[bm] = (us, real_tf, kern_tf)
            except Exception as e:
                print(row + f"  ERROR: {type(e).__name__}: {str(e)[:80]}")
                continue

            us64, rtf64, ktf64 = res[64]
            us128, rtf128, ktf128 = res[128]
            best_real = max(rtf64, rtf128)
            est96 = estimate_block_m_96(ktf64, ktf128, num_groups, em, n, k)
            gain_pct = 100.0 * (est96 / best_real - 1.0)

            print(f"{em:>5d} | {us64:>9.1f} {rtf64:>8.1f} {ktf64:>8.1f} "
                  f"| {us128:>10.1f} {rtf128:>8.1f} {ktf128:>8.1f} "
                  f"| {best_real:>9.1f} {est96:>10.1f} {gain_pct:>+12.1f}%")
        print()

    # Reset state
    deep_gemm.set_num_sms(0)
    deep_gemm.set_mk_alignment_for_contiguous_layout(128)
    return 0


if __name__ == "__main__":
    sys.exit(main())
