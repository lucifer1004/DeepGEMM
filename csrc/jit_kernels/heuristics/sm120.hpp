#pragma once

#include <algorithm>

#include <cute/arch/mma_sm100_desc.hpp>
#include <deep_gemm/common/types.cuh>

#include "common.hpp"
#include "runtime.hpp"
#include "utils.hpp"
#include "../../utils/exception.hpp"

namespace deep_gemm {

struct SM120ArchSpec {
    static constexpr int smem_capacity = 101376;  // 99KB

    static std::vector<Layout> get_layout_candidates(const GemmDesc& desc) {
        const int elem_size = get_element_size(desc.get_mma_kind());

        // BLOCK_M candidates: smaller values reduce grouped-GEMM padding at
        // small expected_m (MoE decode), larger values amortize launch
        // overhead at prefill. Kernel constraints (sm120_fp8_fp4_gemm_1d1d.cuh
        // and sm120_fp8_gemm_1d1d.cuh):
        //   * kNumMathThreads=256 → kNumMathWarps=8
        //   * BLOCK_M must equal kMWarps × kMTilesPerWarp × MMA_M with
        //     MMA_M=16 and kNumMathWarps = kMWarps × kNWarps
        //   * The dispatcher picks kNWarps={2,4} based on BLOCK_M:
        //       - BLOCK_M % 64 == 0 → kNWarps=2 (kMWarps=4): step=64
        //       - else BLOCK_M % 32 == 0 → kNWarps=4 (kMWarps=2): step=32
        //     Valid BLOCK_M ∈ {64, 96, 128, 160, 192, 224, ...}.
        //   * Plus the caller's runtime alignment cap and expected_m fit
        const int expected_m = desc.get_expected_m();
        const int runtime_align = heuristics_runtime->get_mk_alignment_for_contiguous_layout();
        constexpr int kMinBlockM = 64;
        // Valid step in M: any value divisible by 32 (covers both kNWarps=2
        // step-64 and kNWarps=4 step-32 paths).
        constexpr int kBlockMStep = 32;
        std::vector<int> block_m_candidates;
        // For MGroupedContiguous, BLOCK_M must DIVIDE runtime_align — the
        // caller pads each expert's M-slice to a multiple of runtime_align,
        // and the scheduler tiles M in BLOCK_M strides starting from expert
        // boundaries. If BLOCK_M doesn't divide runtime_align, a tile can
        // straddle two experts, mixing their data → catastrophic numerical
        // error (observed: BM=64 with align=96 → diff 0.08 on FP8 ue8m0).
        const bool needs_align_divisibility =
            (desc.gemm_type == GemmType::MGroupedContiguous or
             desc.gemm_type == GemmType::MGroupedContiguousWithPsumLayout) and
            runtime_align > 0;
        // Three sizes worth considering across MoE workloads:
        //   64  — snug for expected_m ≤ 64 (decode and small prefill)
        //   96  — snug for expected_m ∈ [65, 96] (mid-prefill MoE band)
        //   128 — snug for expected_m ∈ [97, 128] (typical prefill)
        for (int bm : {64, 96, 128}) {
            if (runtime_align > 0 and bm > runtime_align)
                continue;
            if (needs_align_divisibility and runtime_align % bm != 0)
                continue;
            if (expected_m > 0 and bm > expected_m and bm > kMinBlockM)
                continue;
            block_m_candidates.push_back(bm);
        }
        // Allow runtime_align itself as a candidate if it satisfies the
        // kernel's cooperative-warp divisibility (% kBlockMStep) and is in
        // [64, 128].
        if (runtime_align >= kMinBlockM and runtime_align <= 128 and runtime_align % kBlockMStep == 0
            and std::find(block_m_candidates.begin(), block_m_candidates.end(), runtime_align) == block_m_candidates.end()) {
            block_m_candidates.push_back(runtime_align);
        }
        if (block_m_candidates.empty())
            block_m_candidates.push_back(128);

        // Block K candidates: BK=64 enables 4 pipeline stages (better TMA hiding),
        // but only beneficial for large M (>= 2048) and non-mixed dtypes.
        const bool is_mixed = (desc.a_dtype != desc.b_dtype);
        std::vector<int> block_k_candidates;
        if (!is_mixed and expected_m >= 2048)
            block_k_candidates.push_back(64 / elem_size);
        block_k_candidates.push_back(128 / elem_size);

        // Block N candidates. Kernel structural minimum BLOCK_N=16 (BLOCK_N/MMA_N
        // must be divisible by kNWarps=2). The default cp.async/TMA alignment
        // floor `(BN * elem_size) % 64 == 0` forces BN ∈ {64, 128} on the normal
        // path. The AB-swap small-N path (`desc.n ≤ 32` after the caller swap,
        // see apis/einsum.hpp::fp8_bmm) relaxes that floor and only emits
        // {16, 32} so we don't waste lanes covering M_orig ≤ 32.
        const int n_for_tile = desc.get_expected_n() > 0 ? desc.get_expected_n() : desc.n;
        const bool is_small_n_swap = (n_for_tile > 0 and n_for_tile <= 32);
        std::vector<int> block_n_candidates;
        int step = std::lcm(16, heuristics_runtime->get_block_n_multiple_of());
        for (int i = step; i <= 256; i += step) {
            const int aligned = i * get_element_size(desc.get_mma_kind());
            if (is_small_n_swap) {
                if (i > 32)
                    continue;
            } else if (aligned % 64 != 0) {
                continue;
            }
            block_n_candidates.push_back(i);
        }

        // MN-major B: ldmatrix.trans.x2 handles multi-atom SMEM correctly
        const int mn_major_b_max_n = 128;

        std::vector<Layout> candidates;
        for (int block_m : block_m_candidates) {
        for (int block_k : block_k_candidates) {
        for (int block_n : block_n_candidates) {
            if (block_n > 128 or block_n > mn_major_b_max_n)
                continue;
            // BLOCK_M=96 uses kNWarps=4 (kMWarps=2), which requires
            // kNTiles = BLOCK_N / MMA_N (=8) to be divisible by 4, i.e.
            // BLOCK_N >= 32. The small-N swap-AB path emits BLOCK_N ∈
            // {16, 32}, which would otherwise let BM=96+BN=16 reach NVCC
            // and trip the "N tiles must divide evenly" static_assert.
            if (block_m == 96 and block_n < 32)
                continue;

            const auto layout = Layout{0, block_m, block_n, block_k, 1, 1};
            const auto storage_config = get_storage_config(desc, layout);

            if (storage_config.swizzle_a_mode < 64 or storage_config.swizzle_b_mode < 64)
                continue;

            int num_stages = get_pipeline_config(desc, layout, storage_config).num_stages;
            if (num_stages < 2)
                continue;

            candidates.push_back(layout);
        }
        }
        }

        DG_HOST_ASSERT(not candidates.empty());
        return candidates;
    }

    static int get_smem_bytes_per_k(const at::ScalarType& dtype, int block_k) {
        return (dtype == kPackedFP4) ? (block_k / 2) : (block_k * static_cast<int>(c10::elementSize(dtype)));
    }

    static int get_smem_d_size_for_swizzle(const GemmDesc& desc, const Layout& layout, int swizzle_cd, int store_m) {
        const int cd_size = c10::elementSize(desc.cd_dtype);
        if (swizzle_cd > 0 and cd_size <= 2
            and layout.block_n * cd_size >= swizzle_cd
            and (layout.block_n * cd_size) % swizzle_cd == 0)
            return (layout.block_n * cd_size / swizzle_cd) * swizzle_cd * store_m;
        return 0;
    }

    static int get_smem_per_stage(const GemmDesc& desc, const Layout& layout) {
        const bool b_padded_fp4 = (desc.a_dtype != kPackedFP4 && desc.b_dtype == kPackedFP4);
        const int smem_a = layout.block_m * get_smem_bytes_per_k(desc.a_dtype, layout.block_k);
        const int smem_b = layout.block_n *
            (b_padded_fp4 ? layout.block_k : get_smem_bytes_per_k(desc.b_dtype, layout.block_k));
        const int smem_sfa = (desc.kernel_type == KernelType::Kernel1D1D)
            ? align(layout.block_m * static_cast<int>(sizeof(int32_t)), 128) : 0;
        const int smem_sfb = (desc.kernel_type == KernelType::Kernel1D1D)
            ? align(layout.block_n * static_cast<int>(sizeof(int32_t)), 128) : 0;
        return smem_a + smem_b + smem_sfa + smem_sfb;
    }

    static StorageConfig get_storage_config(const GemmDesc& desc, const Layout& layout) {
        const auto load_block_m = layout.block_m;
        const auto load_block_n = layout.block_n;

        const auto smem_k_bytes_a = get_smem_bytes_per_k(desc.a_dtype, layout.block_k);
        const auto swizzle_mode_a = get_swizzle_mode(smem_k_bytes_a, 1);
        // Mixed FP8xFP4: B uses .b4x16_p64 padded SMEM (row stride = block_k, same as FP8)
        const bool b_padded_fp4 = (desc.a_dtype != kPackedFP4 && desc.b_dtype == kPackedFP4);
        const auto smem_row_bytes_b = (desc.major_b == cute::UMMA::Major::K)
            ? (b_padded_fp4 ? layout.block_k : get_smem_bytes_per_k(desc.b_dtype, layout.block_k))
            : layout.block_n * static_cast<int>(c10::elementSize(desc.b_dtype));
        const auto swizzle_mode_b = get_swizzle_mode(smem_row_bytes_b, 1);

        // swizzle_mode_cd: enable TMA-store (128 B) only when the BN row fits a
        // full swizzle row. Below that the kernel falls back to direct-store,
        // which is also the path the AB-swap small-N caller needs (TMA-store
        // can't write into the caller's row-major (M_orig, N_orig) layout).
        const int cd_size = static_cast<int>(c10::elementSize(desc.cd_dtype));
        const int swizzle_mode_cd = (cd_size <= 2 and layout.block_n * cd_size >= 128) ? 128 : 0;

        // Sub-tile epilogue: reduce SMEM_D by storing smaller M sub-tiles.
        // Try store_block_m = 64 (sub-tile) and see if it gains pipeline stages.
        constexpr int kNumMaxStages = 16;
        const int smem_barriers = kNumMaxStages * 8 * 2;
        const int per_stage = get_smem_per_stage(desc, layout);
        const int smem_d_full = get_smem_d_size_for_swizzle(desc, layout, swizzle_mode_cd, layout.block_m);
        const int stages_full = std::min((smem_capacity - smem_barriers - smem_d_full) / per_stage, kNumMaxStages);

        // Sub-tile is only valid when it divides BLOCK_M evenly. The TMA-store
        // epilogue computes kNumEpiMSubs = BLOCK_M / kEpiSubM via integer
        // division — a non-divisor leaves the tail rows unwritten (BLOCK_M=96
        // with store_m=64 silently drops rows 64-95).
        int store_m = layout.block_m;
        constexpr int kSubTileM = 64;
        if (swizzle_mode_cd > 0 and layout.block_m > kSubTileM
            and layout.block_m % kSubTileM == 0) {
            const int smem_d_sub = get_smem_d_size_for_swizzle(desc, layout, swizzle_mode_cd, kSubTileM);
            const int stages_sub = std::min((smem_capacity - smem_barriers - smem_d_sub) / per_stage, kNumMaxStages);
            if (stages_sub > stages_full)
                store_m = kSubTileM;
        }

        return {
            load_block_m, load_block_n,
            store_m, 0,
            swizzle_mode_a, swizzle_mode_b, swizzle_mode_cd
        };
    }

    static int get_smem_d_size(const GemmDesc& desc, const Layout& layout) {
        const auto storage = get_storage_config(desc, layout);
        return get_smem_d_size_for_swizzle(desc, layout, storage.swizzle_cd_mode, storage.store_block_m);
    }

    static PipelineConfig get_pipeline_config(const GemmDesc& desc, const Layout& layout, const StorageConfig& storage_config) {
        constexpr int kNumMaxStages = 16;

        const int smem_barriers = kNumMaxStages * 8 * 2;
        const int smem_a_per_stage = storage_config.load_block_m * get_smem_bytes_per_k(desc.a_dtype, layout.block_k);
        const bool b_padded_fp4 = (desc.a_dtype != kPackedFP4 && desc.b_dtype == kPackedFP4);
        const int smem_b_per_stage = storage_config.load_block_n *
            (b_padded_fp4 ? layout.block_k : get_smem_bytes_per_k(desc.b_dtype, layout.block_k));

        int smem_sfa_per_stage = 0;
        int smem_sfb_per_stage = 0;
        if (desc.kernel_type == KernelType::Kernel1D1D) {
            smem_sfa_per_stage = align(layout.block_m * static_cast<int>(sizeof(int32_t)), 128);
            smem_sfb_per_stage = align(layout.block_n * static_cast<int>(sizeof(int32_t)), 128);
        }

        const int smem_tensormap =
            desc.gemm_type == GemmType::KGroupedContiguous ? 2 * static_cast<int>(sizeof(CUtensorMap)) : 0;

        const int smem_d = get_smem_d_size_for_swizzle(desc, layout, storage_config.swizzle_cd_mode,
                                                       storage_config.store_block_m);

        const int smem_extra = smem_barriers + smem_tensormap + smem_d;
        const int smem_per_stage = smem_a_per_stage + smem_b_per_stage + smem_sfa_per_stage + smem_sfb_per_stage;
        const int num_stages = std::min(
            (smem_capacity - smem_extra) / smem_per_stage,
            kNumMaxStages);
        return {
            smem_extra + num_stages * smem_per_stage,
            num_stages
        };
    }

    static constexpr int mma_m = 16;

    static LaunchConfig get_launch_config(const GemmDesc& desc, const Layout& layout) {
        // Warp-specialized: 1 load warp group (128 threads) + 2 MMA warp groups (256 threads)
        return {
            desc.num_sms,
            1,
            384,              // total threads
            128, 256,         // kNumTMAThreads = 128, kNumMathThreads = 256
            0, 0
        };
    }

    static LayoutInfo get_layout_info(const GemmDesc& desc, const Layout& layout) {
        const auto num_blocks =
            ceil_div(desc.get_expected_m(), layout.block_m) *
            ceil_div(desc.get_expected_n(), layout.block_n) *
            desc.get_expected_num_groups();
        const auto num_waves = ceil_div(num_blocks, desc.num_sms);
        const auto num_last_blocks = num_blocks % desc.num_sms;
        const auto last_wave_util = num_last_blocks == 0 ? desc.num_sms : num_last_blocks;

        const int64_t expected_k = desc.get_expected_k();
        int64_t flops_per_block = 2LL * layout.block_m * layout.block_n * expected_k;

        // Empirical warp-spec MMA efficiency model.
        // A reuse = BN/8 (each A fragment reused across N-tiles).
        // B reuse = BM/8 (each B fragment reused across M-tiles, symmetric).
        // Cooperative layout: larger BN reduces epilogue overhead (fewer tiles)
        // and larger BM amortises B-operand fragment loads — both contribute
        // independently, but the joint boost only fires when BOTH operands
        // get enough reuse, so we compose multiplicatively. Old model used
        // only a_reuse, which underweighted BM at large BN and let the
        // dispatcher pick BM=64 over BM=128 at prefill (M ≥ ~8K), giving
        // a 60% regression vs the BM=128 actual measurement.
        //
        // Boundary cases (preserved from the old model):
        //   - BM=128, BN=128: 0.69 + 0.12·1·1 = 0.81
        //   - BM=128, BN=32:  0.69 + 0.12·0·1 = 0.69
        //   - BM=32,  BN=128: 0.69 + 0.12·1·0 = 0.69
        // New cases:
        //   - BM=64,  BN=128: 0.69 + 0.12·1·0.33 = 0.73 (was 0.81)
        const double a_reuse = static_cast<double>(layout.block_n) / 8.0;
        const double b_reuse = static_cast<double>(layout.block_m) / 8.0;
        const double a_factor = std::min(1.0, (a_reuse - 4.0) / 12.0);
        const double b_factor = std::min(1.0, (b_reuse - 4.0) / 12.0);
        double mma_efficiency = 0.69 + 0.12 * a_factor * b_factor;

        // Per-BM kernel-path efficiency factor. BLOCK_M values not divisible
        // by 64 use the kNWarps=4 (kMWarps=2) cooperative warp layout, which
        // is intrinsically faster per padded-FLOP than the kNWarps=2 path
        // (BM divisible by 64). Median kernel-TFLOPS ratio (BM=96 / BM=64)
        // measured across 1026 configs is 1.26 (see
        // DeepGEMM/tests/sm120/bench_block_m_full_sweep.py). Without this
        // factor the cycle model under-weights BM=96 and over-prefers BM=128.
        const double kernel_eff_factor = (layout.block_m % 64 == 0) ? 1.0 : 1.26;

        const double peak_flops_per_ns = (desc.a_dtype == at::kBFloat16) ? 380000.0 : 762000.0;
        double block_ns = flops_per_block / (peak_flops_per_ns * mma_efficiency * kernel_eff_factor);

        float wave_efficiency = static_cast<float>(num_blocks) / (num_waves * desc.num_sms);
        int64_t num_cycles = static_cast<int64_t>(block_ns * num_blocks / wave_efficiency);

        return {num_waves, last_wave_util, num_cycles, layout};
    }

    static bool compare(const LayoutInfo& a, const LayoutInfo& b) {
        if (a.num_waves != b.num_waves and (a.num_waves == 1 or b.num_waves == 1))
            return a.num_waves < b.num_waves;

        if (a.num_cycles != b.num_cycles)
            return a.num_cycles < b.num_cycles;

        // Tie-break: prefer larger N tile for better reuse
        if (a.layout.block_n != b.layout.block_n)
            return a.layout.block_n > b.layout.block_n;
        // Tie-break: prefer smaller K tile (more pipeline stages for TMA hiding)
        if (a.layout.block_k != b.layout.block_k)
            return a.layout.block_k < b.layout.block_k;
        return false;
    }
};

} // namespace deep_gemm
