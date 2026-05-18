#pragma once

#include "../../jit/device_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/lazy_init.hpp"

namespace deep_gemm {

class HeuristicsRuntime {
    static constexpr int kLegacyMKAlignmentForContiguousLayout = 128;

    bool ignore_compile_dims = false;
    int block_m_multiple_of = 1;
    int block_n_multiple_of = 1;
    int mk_alignment_for_contiguous_layout = kLegacyMKAlignmentForContiguousLayout;

public:
    void set_ignore_compile_dims(const bool& new_value) {
        ignore_compile_dims = new_value;
    }

    bool get_ignore_compile_dims() const {
        return ignore_compile_dims;
    }

    void set_block_size_multiple_of(const int& new_block_m_multiple_of, const int& new_block_n_multiple_of) {
        block_m_multiple_of = new_block_m_multiple_of;
        block_n_multiple_of = new_block_n_multiple_of;
    }

    int get_block_m_multiple_of() const {
        return block_m_multiple_of;
    }

    int get_block_n_multiple_of() const {
        return block_n_multiple_of;
    }

    void set_mk_alignment_for_contiguous_layout(const int& new_value) {
        mk_alignment_for_contiguous_layout = new_value;
    }

    int get_mk_alignment_for_contiguous_layout() const {
        return mk_alignment_for_contiguous_layout;
    }

    static int get_theoretical_mk_alignment_for_contiguous_layout(
            const std::optional<int>& expected_m,
            const std::optional<int>& num_groups = std::nullopt) {
        const auto arch_major = device_runtime->get_arch_major();
        if (arch_major != 10 and arch_major != 12)
            return kLegacyMKAlignmentForContiguousLayout;

        // SM120 piecewise rule keyed on PER-EXPERT em (not total em). vllm
        // passes `expected_m = M * num_topk` (sum across experts) and
        // `num_groups = local_num_experts` (this rank's expert count); we
        // divide to recover the per-expert workload. Without num_groups
        // (legacy callers), expected_m is treated as already per-expert.
        //
        // Boundaries from the prefill-range bench
        // (DeepGEMM/tests/sm120/bench_block_m_prefill_range.py, 1452 configs,
        // per-expert em ∈ {32..4096}):
        //   - em ≤ 64:   BM=64 wins ~100%   (snug fit, no padding)
        //   - em ∈ [65, 96]: BM=96 wins majority (snug fit, kNWarps=4 path)
        //   - em ∈ [97, 128]: BM=64 wins ~83% (2 tiles of 64 beat 1 tile of
        //                    96 padded, despite kNWarps=4 advantage)
        //   - em ≥ 256:  BM=96 wins ≥63%, climbing to 100% by em=1024
        //                (kNWarps=4 intrinsic efficiency dominates)
        //
        // SM100: original "largest BM that fits" rule (the SM100 kernel
        // dispatches grouped-contiguous through swap-AB, which has different
        // amortization characteristics than SM120's direct path).
        if (arch_major == 12) {
            if (not expected_m.has_value())
                return 64;
            const int em = expected_m.value();
            const int ng = std::max(num_groups.value_or(1), 1);
            const int per_expert_em = (em + ng - 1) / ng;  // ceil(em / ng)
            if (per_expert_em <= 64)  return 64;
            if (per_expert_em <= 96)  return 96;
            if (per_expert_em <= 128) return 64;
            return 96;
        }
        // SM100 path (unchanged)
        int block_m = 240;
        int min_block_m = 32;
        int mma_step = 16;
        if (expected_m.has_value()) {
            for (; block_m > min_block_m and block_m - mma_step >= expected_m.value(); block_m -= mma_step);
        }
        return block_m;
    }
};

static auto heuristics_runtime = LazyInit<HeuristicsRuntime>([](){ return std::make_shared<HeuristicsRuntime>(); });

} // namespace deep_gemm
