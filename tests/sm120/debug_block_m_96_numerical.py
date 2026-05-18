"""Find a minimal repro for the kNWarps=4 (BLOCK_M=96) numerical regression.

Observed: 4g 8192x4096x4096 FP8 ue8m0 → diff 0.001644 vs threshold 0.001.
Other configs at the same shape (FP4-mixed, FP8 e4m3) pass.

This script:
1. Sweeps (num_groups, expected_m, n, k) shapes that pick BM=96 via the
   autotuner, AND smaller shapes that we can FORCE to BM=96 by setting
   runtime_align=96.
2. Reports diff vs reference for each.
3. Locates the first row/col that disagrees for the worst-failing shape.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import deep_gemm
from deep_gemm.testing import calc_diff
from generators import (
    generate_m_grouped_contiguous,
    MajorTypeAB,
    QuantConfig,
    get_ue8m0_usage,
    KernelType,
    reset_seed,
)


def run_one(num_groups, expected_m, n, k, runtime_align=None, label=""):
    use_ue8m0 = get_ue8m0_usage(KernelType.Kernel1D1D)
    disable_ue8m0_cast = not use_ue8m0
    for qc in QuantConfig.get_list_from_dtype(torch.float8_e4m3fn):
        qc_name = "FP4-mixed" if qc.is_fp4_a else f"FP8-{('ue8m0' if qc.max_diff() < 0.005 else 'e4m3')}"
        align = runtime_align if runtime_align is not None else deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout()
        deep_gemm.set_mk_alignment_for_contiguous_layout(align)
        reset_seed()
        m, a, b, gl, d, ref_d = generate_m_grouped_contiguous(
            num_groups, expected_m, n, k,
            MajorTypeAB.KMajor, MajorTypeAB.KMajor,
            use_ue8m0=use_ue8m0, use_psum_layout=False, quant_config=qc,
        )
        recipe, recipe_a, recipe_b = qc.get_recipes()
        deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
            a, b, d, gl,
            disable_ue8m0_cast=disable_ue8m0_cast,
            use_psum_layout=False,
            recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b,
        )
        diff = calc_diff(d, ref_d)
        thresh = qc.max_diff()
        status = "OK " if diff < thresh else "BAD"
        print(f"  [{status}] {label} {qc_name} m={m} diff={diff:.6f} (thresh={thresh:.6f})")
        if diff >= thresh:
            # Per-row diff to locate where it goes wrong
            row_diffs = (d.float() - ref_d.float()).abs().amax(dim=-1)
            worst_rows = row_diffs.topk(10)
            print(f"       worst 10 row indices: {worst_rows.indices.tolist()}")
            print(f"       worst 10 row max-abs: {[f'{v:.4f}' for v in worst_rows.values.tolist()]}")
            # Per-col diff to locate column structure
            col_diffs = (d.float() - ref_d.float()).abs().amax(dim=0)
            print(f"       col diff stats: max={col_diffs.max():.4f} mean={col_diffs.mean():.4f}")


def main():
    print("=== Sweep: shapes likely to pick BM=96 by autotuner ===")
    for num_groups in [1, 2, 4, 8]:
        for em in [80, 96, 128, 256, 1024, 4096, 8192]:
            run_one(num_groups, em, 4096, 4096, label=f"g={num_groups} em={em} 4096x4096")

    print()
    print("=== Force BM=96 via runtime_align=96, smaller shapes ===")
    for num_groups in [1, 4]:
        for em in [80, 96, 128, 256, 1024, 4096]:
            run_one(num_groups, em, 4096, 4096, runtime_align=96,
                    label=f"g={num_groups} em={em} 4096x4096 ALIGN96")

    print()
    print("=== Force BM=128 (baseline) on the same shapes for comparison ===")
    for num_groups in [4]:
        for em in [8192]:
            run_one(num_groups, em, 4096, 4096, runtime_align=128,
                    label=f"g={num_groups} em={em} 4096x4096 ALIGN128")


if __name__ == "__main__":
    main()
