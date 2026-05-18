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

    static int get_theoretical_mk_alignment_for_contiguous_layout(const std::optional<int>& expected_m) {
        const auto arch_major = device_runtime->get_arch_major();
        if (arch_major != 10 and arch_major != 12)
            return kLegacyMKAlignmentForContiguousLayout;

        // SM120: data-driven smart pick (see DeepGEMM/tests/sm120/
        // bench_block_m_full_sweep.py). The cycle-model autotuner consistently
        // over-prefers BLOCK_M=128, but measurements show that:
        //   - BLOCK_M=64 wins for em ≤ 64 (snug fit, no padding)
        //   - BLOCK_M=96 wins for em ∈ [65, 80] (kNWarps=4 path is intrinsically
        //     ~25% faster per padded-FLOP than kNWarps=2, and the snugger fit
        //     vs BLOCK_M=128 compounds the gain)
        //   - BLOCK_M=64 wins for em > 80 (smaller blocks → more pipeline
        //     stages → better TMA overlap; the cycle-model's amortization
        //     argument for BLOCK_M=128 doesn't survive measurement)
        // Aggregate across the bench: +6.5% real-TFLOPS vs the
        // "largest-BM-that-fits" rule, within 0.6% of the per-config oracle.
        //
        // SM100: original "largest BM that fits" rule (the SM100 kernel
        // dispatches grouped-contiguous through swap-AB, which has different
        // amortization characteristics than SM120's direct path).
        if (arch_major == 12) {
            if (not expected_m.has_value())
                return 64;
            const int em = expected_m.value();
            if (em <= 64)
                return 64;
            if (em <= 80)
                return 96;
            return 64;
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
