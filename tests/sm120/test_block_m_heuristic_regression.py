"""Regression test for the SM120 BLOCK_M heuristic + kNWarps refactor.

History:

1. **Original bug (pre-aa960cd-ish)**: The grouped FP8/FP4 1D1D kernel
   hardcoded `kMWarps=4` and `MMA_M=16`, so `BLOCK_M % 64 == 0` was
   required (static_assert in `sm120_fp8_fp4_gemm_1d1d.cuh:90`). But
   `get_theoretical_mk_alignment_for_contiguous_layout` stepped by 16
   for SM120, yielding 80/96/112 — which the kernel couldn't compile.
   The autotuner's `% 16` filter let them through, crashing NVCC at JIT
   time. Surfaced at TP=4 on a remote RTX PRO 5000 Blackwell.

2. **Heuristic-only fix**: Restricted BLOCK_M to multiples of 64 in
   both the theoretical helper and the candidate filter.

3. **Kernel refactor (this revision)**: Made `kNWarps` a template
   parameter (default 2). The dispatcher now picks `kNWarps=4`
   (→`kMWarps=2`) when `BLOCK_M % 64 != 0` but `BLOCK_M % 32 == 0`.
   This unlocks `BLOCK_M=96` natively, snug-fitting the [65, 96]
   expected_m band that previously rounded up to BLOCK_M=128.

These tests run on any SM120 host. For autotuner-scoring coverage we
force `num_sms = 110` to mirror the 5K Pro.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import torch
import deep_gemm
from deep_gemm.testing import get_arch_major


pytestmark = pytest.mark.skipif(
    get_arch_major() != 12,
    reason="SM120-specific heuristic regression",
)


_RTX_PRO_5000_BLACKWELL_SMS = 110


# Kernel constraint (post-refactor):
#   - kNWarps=2 path: BLOCK_M % (kMWarps*MMA_M) = BLOCK_M % 64 == 0
#   - kNWarps=4 path: BLOCK_M % (kMWarps*MMA_M) = BLOCK_M % 32 == 0
# Union: BLOCK_M ∈ {64, 96, 128, 160, 192, 224, ...}, i.e. multiple of 32 ≥ 64.
_KERNEL_BLOCK_M_STEP = 32
_KERNEL_MIN_BLOCK_M = 64
# Theoretical helper returns 64 (decode regime) or 128 (prefill regime). The
# old (64, 96) range was based on a sweep that compared BM=96 vs BM=64 only,
# not against BM=128; the prefill bench at DSv4-Flash MoE shapes (N=4096,
# K ∈ {2048, 4096}, G=64) shows BM=128 wins consistently for per-expert em > 64.
_THEORETICAL_VALID_BLOCK_M = (64, 128)


@pytest.mark.parametrize(
    "expected_m",
    [1, 6, 32, 63, 64, 65, 72, 80, 96, 100, 127, 128, 192, 240, 1000],
)
def test_theoretical_alignment_returns_kernel_valid_block_m(expected_m):
    """The theoretical helper must never return a BLOCK_M the kernel can't compile."""
    align = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout(expected_m)
    assert align >= _KERNEL_MIN_BLOCK_M and align % _KERNEL_BLOCK_M_STEP == 0, (
        f"Theoretical alignment for expected_m={expected_m} returned {align}, "
        f"which violates BLOCK_M >= {_KERNEL_MIN_BLOCK_M} and "
        f"BLOCK_M %% {_KERNEL_BLOCK_M_STEP} == 0"
    )
    assert align in _THEORETICAL_VALID_BLOCK_M, (
        f"Theoretical alignment for expected_m={expected_m} = {align}; "
        f"expected one of {_THEORETICAL_VALID_BLOCK_M}"
    )


def test_theoretical_alignment_per_expert_em_le_64():
    """Per-expert em ≤ 64: BM=64 (snug, 100% bench win)."""
    # With num_groups omitted, expected_m is treated as per-expert em.
    assert deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout(6) == 64
    assert deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout(64) == 64
    # With num_groups: ceil(640/16) = 40 per-expert → BM=64.
    assert deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout(640, 16) == 64


def test_theoretical_alignment_per_expert_em_gt_64():
    """Per-expert em > 64: BM=128 wins.

    DSv4-Flash MoE prefill bench (N=4096, K ∈ {2048, 4096}, G=64) shows BM=128
    beats BM=96 by 3-26% across the whole prefill range, and beats BM=64 by
    24-44%. The previous heuristic returned 96 (or 64 for [97, 128]) because
    its underlying sweep didn't include BM=128 in the candidate comparison.
    """
    # Just above the decode threshold
    assert deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout(65) == 128
    assert deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout(96) == 128
    # Previously [97, 128] → 64 — now 128
    assert deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout(128) == 128
    # Production prefill (DSv4-Flash @ TP=4): M=8000, topk=8, G=64 → per-expert 1000
    assert deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout(64000, 64) == 128
    # TP=2: per-expert 500
    assert deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout(64000, 128) == 128
    # TP=8: per-expert 2000
    assert deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout(64000, 32) == 128


def test_theoretical_alignment_decode_small_batch_with_num_groups():
    """Decode at small batch: even if total em looks moderate, per-expert is tiny → BM=64.

    Regression case the previous heuristic missed: decode B=32, topk=8 → total em=256.
    Without num_groups, treated as per-expert 256 → BM=96 (wrong; per-expert ≈ 4 for
    TP=4 with 64 experts). With num_groups, picks BM=64 correctly.
    """
    # Decode B=32 at TP=4: total em=256, 64 experts, per-expert=4 → BM=64
    assert deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout(256, 64) == 64
    # Decode B=256 at TP=4: total em=2048, 64 experts, per-expert=32 → BM=64
    assert deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout(2048, 64) == 64


@pytest.mark.parametrize("runtime_align", [80, 96, 112])
def test_grouped_gemm_runs_with_caller_set_runtime_align(runtime_align):
    """A caller that hand-sets runtime_align to 80/96/112 must NOT crash the
    kernel JIT.

    Post-refactor:
      - runtime_align=80  → heuristic filter `% 32 != 0` → fall back to BLOCK_M ∈ {64, 96, 128}
      - runtime_align=96  → heuristic accepts 96 → kernel JIT'd at kNWarps=4
      - runtime_align=112 → heuristic filter `% 32 != 0` → fall back

    Mirrors the TP=4 failure mode: 256 experts / TP4 = 64 groups,
    SHAPE_N=4096, SHAPE_K=4096, expected_m_per_group small.
    """
    from generators import (
        generate_m_grouped_contiguous,
        MajorTypeAB,
        get_ue8m0_usage,
        KernelType,
        QuantConfig,
    )

    # Cap visible SMs to the RTX PRO 5000 Blackwell count so the autotuner's
    # scoring matches the host where the bug surfaced. On hosts with fewer
    # SMs this is a no-op.
    prev_sms = deep_gemm.get_num_sms()
    target_sms = min(_RTX_PRO_5000_BLACKWELL_SMS, prev_sms)
    deep_gemm.set_num_sms(target_sms)
    deep_gemm.set_mk_alignment_for_contiguous_layout(runtime_align)

    try:
        use_ue8m0 = get_ue8m0_usage(KernelType.Kernel1D1D)
        disable_ue8m0_cast = not use_ue8m0
        num_groups, expected_m, n, k = 64, runtime_align, 4096, 4096

        for qc in QuantConfig.get_list_from_dtype(torch.float8_e4m3fn):
            recipe, recipe_a, recipe_b = qc.get_recipes()
            m, a, b, grouped_layout, d, ref_d = generate_m_grouped_contiguous(
                num_groups, expected_m, n, k,
                MajorTypeAB.KMajor, MajorTypeAB.KMajor,
                use_ue8m0=use_ue8m0,
                use_psum_layout=False,
                quant_config=qc,
            )
            deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
                a, b, d, grouped_layout,
                disable_ue8m0_cast=disable_ue8m0_cast,
                use_psum_layout=False,
                recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b,
            )
            # Tight tolerance is not the goal here; we just need the JIT to
            # compile and the kernel to execute without error.
            assert d.isfinite().all(), f"NaN/Inf in output for {qc=}"
    finally:
        deep_gemm.set_num_sms(0)  # reset to physical count
        deep_gemm.set_mk_alignment_for_contiguous_layout(128)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
