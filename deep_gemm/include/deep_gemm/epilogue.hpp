#pragma once

#include <deep_gemm/epilogue/operators.hpp>

namespace deep_gemm {

constexpr uint8_t kEpilogueAIVId = 1;
constexpr uint32_t kNumEpiloguePipeEvents = 8;

template <
    typename cd_dtype_t,
    typename epilogue_operator_t,
    bool kWithAccumulation,
    uint32_t BLOCK_M, uint32_t BLOCK_N,
    uint32_t MAD_M, uint32_t MAD_N,
    uint32_t kNumEpilogueStages,
    asc_store_l2_cache_mode kL2Store = asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM>
struct Epilogue {
    epilogue_operator_t epilogue_operator;

    static_assert(sizeof(epilogue_operator_t) == sizeof(EpilogueOperatorArgs),
                  "epilogue operators must not add state to EpilogueOperatorArgs");

    static constexpr bool is_bf16_out = std::is_same_v<cd_dtype_t, bfloat16_t>;
    static constexpr bool is_fp8_out = std::is_same_v<cd_dtype_t, float8_e4m3_t>;
    static constexpr bool with_alpha = std::is_same_v<epilogue_operator_t, epilogue::operators::ScaleByAlpha>;
    static constexpr bool with_output_sf = std::is_same_v<epilogue_operator_t, epilogue::operators::QuantizeToFP8>;
    static constexpr bool with_transform = with_alpha || with_output_sf;
    using fp8_operator_t = epilogue::operators::QuantizeToFP8;
    using ub_dtype_t = std::conditional_t<with_output_sf, bfloat16_t, std::conditional_t<with_transform, float, cd_dtype_t>>;

    static constexpr uint32_t kNumMadCDStages = (BLOCK_M / MAD_M) * (BLOCK_N / MAD_N);
    static constexpr uint32_t kNumSFGroups = with_output_sf ? MAD_N / fp8_operator_t::kSFPackN : 1;

    static constexpr ub_ptr<ub_dtype_t> ub{MAD_M, MAD_N};
    static constexpr ub_ptr<uint16_t> sf_ub{kNumSFGroups, MAD_M, ub.offset(kNumEpilogueStages)};

    static_assert(MAD_N % get_frac_k<ub_dtype_t>() == 0, "L0C-to-UB row stride must be 32-byte aligned");
    static_assert(kNumEpilogueStages >= kNumMadCDStages, "epilogue stages must be >= CD stages");
    static_assert(!with_output_sf || MAD_N % fp8_operator_t::kSFPackN == 0,
                  "FP8 output requires MAD tiles with a multiple of 64 columns");
    static_assert(is_fp8_out == with_output_sf, "FP8 output requires the FP8 quantization epilogue");
    static_assert(!with_output_sf || !kWithAccumulation, "FP8 output does not support accumulation");
    static_assert(ub.offset(kNumEpilogueStages) <= UBSizeBytes, "UB overflow");
    static_assert(!with_output_sf || sf_ub.offset(kNumEpilogueStages) <= UBSizeBytes, "UB overflow");

    __aicore__ void aic_wait_free(uint32_t s_epi) const {
        wait_intra_block(PIPE_FIX, kEpilogueAIVId, s_epi);
    }

    template <typename L0C>
    __aicore__ void aic_store(L0C l0c_tile, uint32_t s_epi, uint32_t m_elems, uint32_t n_elems) const {
        copy_l0c_to_ub(ub[s_epi], l0c_tile, m_elems, n_elems, asc_dual_dst_mode::DUAL_DST_DISABLE, kEpilogueAIVId);
        set_intra_block(PIPE_FIX, kEpilogueAIVId, s_epi);
    }

    __aicore__ void aiv_drain(gm_ptr<cd_dtype_t, Major::K> d, uint32_t m_idx, uint32_t n_idx,
                              uint32_t actual_m, uint32_t actual_n, uint32_t& s_epi, uint32_t batch_idx = 0) const {
        const uint64_t stride_d = d.stride_outer;

        for (uint32_t m_mad = 0; m_mad < actual_m; m_mad += MAD_M) {
            for (uint32_t n_mad = 0; n_mad < actual_n; n_mad += MAD_N) {
                const uint32_t eff_rows = min(actual_m - m_mad, MAD_M);
                const uint32_t eff_cols = min(actual_n - n_mad, MAD_N);
                auto gm_dst = d.index(m_idx + m_mad, n_idx + n_mad).ptr();
                const auto ub_stage = ub[s_epi];
                const ub_ptr<cd_dtype_t> output_ub(MAD_M, MAD_N, ub_stage.addr);

                wait_intra_block(with_transform ? PIPE_V : PIPE_MTE3, kEpilogueAIVId, s_epi);
                if constexpr (with_alpha) {
                    epilogue_operator_t::template apply_values<cd_dtype_t, MAD_M * MAD_N>(ub_stage.ptr(), output_ub.ptr(),
                                                                                          epilogue_operator.alpha);
                } else if constexpr (with_output_sf) {
                    epilogue_operator_t::template apply_values<MAD_M, MAD_N>(ub_stage.ptr(), output_ub.ptr(), sf_ub[s_epi].ptr());
                }
                if constexpr (with_transform) {
                    asc_sync_notify(PIPE_V, PIPE_MTE3, static_cast<event_t>(s_epi % kNumEpiloguePipeEvents));
                    asc_sync_wait(PIPE_V, PIPE_MTE3, static_cast<event_t>(s_epi % kNumEpiloguePipeEvents));
                }
                copy_ub_to_gm(gm_dst, output_ub, eff_rows, eff_cols, stride_d * sizeof(cd_dtype_t), kL2Store);
                if constexpr (with_output_sf) {
                    auto sfd = reinterpret_cast<__gm__ uint16_t*>(epilogue_operator.sfd);
                    const uint32_t sf_pair_idx =
                        (batch_idx * epilogue_operator.shape_n + n_idx + n_mad) / fp8_operator_t::kSFPackN;
                    copy_ub_to_gm(sfd + sf_pair_idx * epilogue_operator.sfd_stride + m_idx + m_mad,
                                  sf_ub[s_epi], eff_cols / fp8_operator_t::kSFPackN, eff_rows,
                                  epilogue_operator.sfd_stride * sizeof(uint16_t), kL2Store);
                }
                set_intra_block(PIPE_MTE3, kEpilogueAIVId, s_epi);

                s_epi = s_epi == kNumEpilogueStages - 1 ? 0 : s_epi + 1;
            }
        }
    }

    __aicore__ void aiv_initialize() const {
        if constexpr (with_transform)
            set_intra_blocks<PIPE_V, kNumEpilogueStages>(kEpilogueAIVId);
        else
            set_intra_blocks<PIPE_MTE3, kNumEpilogueStages>(kEpilogueAIVId);
        if constexpr (kWithAccumulation) {
            if constexpr (is_bf16_out)
                asc_set_atomic_add_bfloat();
            else
                asc_set_atomic_add_float();
        }
    }
};

} // namespace deep_gemm
