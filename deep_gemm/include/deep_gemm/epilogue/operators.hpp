#pragma once

#include <deep_gemm/ascend.hpp>

namespace deep_gemm::math {

template <typename quant_dtype_t>
__simd_callee__ inline void get_ue8m0_sf_exp(
    vector_uint16_t& sf_exp, vector_bfloat16_t values, vector_bool group_mask
) {
    constexpr bool kIsFP8 = std::is_same_v<quant_dtype_t, float8_e4m3_t>;
    static_assert(kIsFP8 || is_fp4<quant_dtype_t>(), "quantized type must be E4M3 or E2M1");
    constexpr uint16_t kMantissaBits = 7;
    constexpr uint16_t kMantissaMask = (1u << kMantissaBits) - 1;
    constexpr uint16_t kQuantMaxMantissa = kIsFP8 ? 0x60 : 0x40;
    constexpr uint16_t kQuantMaxExponent = kIsFP8 ? 8 : 2;
    constexpr uint16_t kMinSFExponent = kIsFP8 ? 105 : 1;

    const auto value_mask = asc_create_mask_b16(PAT_VL64);
    const auto full_mask = asc_create_mask_b16(PAT_ALL);
    vector_uint16_t abs_bits, amax_bits, sign_mask, quant_max_exp;
    asc_duplicate_scalar(sign_mask, static_cast<uint16_t>((1u << 15) - 1), value_mask);
    asc_and(abs_bits, reinterpret_cast<vector_uint16_t&>(values), sign_mask, value_mask);
    asc_reduce_max(amax_bits, abs_bits, group_mask);
    asc_duplicate(sf_exp, amax_bits, full_mask);
    asc_add_scalar(sf_exp, sf_exp, kMantissaMask - kQuantMaxMantissa, full_mask);
    asc_shiftright_scalar(sf_exp, sf_exp, kMantissaBits, full_mask);
    asc_max_scalar(sf_exp, sf_exp, kMinSFExponent + kQuantMaxExponent, full_mask);
    asc_duplicate_scalar(quant_max_exp, kQuantMaxExponent, full_mask);
    asc_sub(sf_exp, sf_exp, quant_max_exp, full_mask);
}

__simd_callee__ inline void get_ue8m0_sf_inv(vector_bfloat16_t& sf_inv, vector_uint16_t sf_exp) {
    constexpr uint16_t kMantissaBits = 7;
    constexpr uint16_t kReciprocalExponentBias = 254;
    const auto mask = asc_create_mask_b16(PAT_ALL);
    vector_uint16_t inverse_bits;
    asc_duplicate_scalar(inverse_bits, kReciprocalExponentBias, mask);
    asc_sub(inverse_bits, inverse_bits, sf_exp, mask);
    asc_shiftleft_scalar(inverse_bits, inverse_bits, kMantissaBits, mask);
    sf_inv = reinterpret_cast<vector_bfloat16_t&>(inverse_bits);
}

__simd_callee__ inline void cast_bf16(vector_uint8_t& output, vector_bfloat16_t values) {
    const auto value_mask = asc_create_mask_b16(PAT_VL64);

    vector_bfloat16_t zero;
    vector_float values_f32, unused;
    asc_duplicate_scalar(zero, static_cast<bfloat16_t>(0), value_mask);
    asc_intlv(reinterpret_cast<vector_bfloat16_t&>(values_f32),
              reinterpret_cast<vector_bfloat16_t&>(unused), zero, values);

    vector_fp8_e4m3fn_t converted;
    vector_uint16_t packed;
    asc_float2e4m3_rn_sat(converted, values_f32, asc_create_mask_b32(PAT_ALL), ASC_DISPERSE_FIRST_QUARTER);
    asc_pack_to_low(packed, reinterpret_cast<vector_uint32_t&>(converted));
    asc_pack_to_low(output, packed);
}

} // namespace deep_gemm::math

namespace deep_gemm::epilogue::operators {

// Operators add behavior without adding state to EpilogueOperatorArgs.
struct Identity : EpilogueOperatorArgs {};

struct ScaleByAlpha : Identity {
    template <typename cd_dtype_t, uint32_t kNumValues>
    static __simd_vf__ inline void apply_values(__ubuf__ float* values, __ubuf__ cd_dtype_t* output, float alpha) {
        constexpr uint32_t kNumFP32PerVector = asc_get_vf_len() / sizeof(float);
        static_assert(kNumValues % kNumFP32PerVector == 0);

        const auto mask = asc_create_mask_b32(PAT_ALL);
        for (uint32_t value_idx = 0; value_idx < kNumValues; value_idx += kNumFP32PerVector) {
            vector_float value;
            asc_loadalign(value, values + value_idx);
            asc_mul_scalar(value, value, alpha, mask);
            if constexpr (std::is_same_v<cd_dtype_t, float>) {
                asc_storealign(output + value_idx, value, mask);
            } else {
                vector_bfloat16_t converted;
                vector_uint16_t packed;
                asc_float2bfloat16_rn(converted, value, mask, ASC_POSITION_EVEN);
                asc_pack_to_low(packed, reinterpret_cast<vector_uint32_t&>(converted));
                asc_storealign(output + value_idx, reinterpret_cast<vector_bfloat16_t&>(packed),
                               asc_create_mask_b16(PAT_VL64));
            }
        }
    }
};

struct QuantizeToFP8 : Identity {
    static constexpr uint32_t kSFPackN = MX_SF_DIVISOR;

    static __simd_callee__ inline void cast_vector(
        __ubuf__ bfloat16_t* values, __ubuf__ float8_e4m3_t* output,
        vector_uint16_t& sf_exp_0, vector_uint16_t& sf_exp_1
    ) {
        const auto mask_b16 = asc_create_mask_b16(PAT_VL64);
        const auto mask_32 = asc_create_mask_b16(PAT_VL32);
        vector_bool mask_32_hi;
        asc_not(mask_32_hi, mask_32, mask_b16);

        vector_bfloat16_t values_bf16;
        asc_loadalign(values_bf16, values);

        vector_bfloat16_t sf_inv_0, sf_inv_1, sf_inv;
        math::get_ue8m0_sf_exp<float8_e4m3_t>(sf_exp_0, values_bf16, mask_32);
        math::get_ue8m0_sf_exp<float8_e4m3_t>(sf_exp_1, values_bf16, mask_32_hi);
        math::get_ue8m0_sf_inv(sf_inv_0, sf_exp_0);
        math::get_ue8m0_sf_inv(sf_inv_1, sf_exp_1);
        asc_select(sf_inv, sf_inv_0, sf_inv_1, mask_32);
        asc_mul(values_bf16, values_bf16, sf_inv, mask_b16);

        vector_uint8_t converted;
        math::cast_bf16(converted, values_bf16);
        asc_storealign(reinterpret_cast<__ubuf__ uint8_t*>(output), converted,
                       asc_create_mask_b8(PAT_VL64));
    }

    template <uint32_t kNumRows, uint32_t kNumCols>
    static __simd_vf__ inline void apply_values(
        __ubuf__ bfloat16_t* values, __ubuf__ float8_e4m3_t* output, __ubuf__ uint16_t* output_sf
    ) {
        static_assert(kNumCols % kSFPackN == 0);
        constexpr uint32_t kNumRowsPerVector = asc_get_vf_len() / sizeof(uint16_t);
        constexpr uint32_t kNumSFGroups = kNumCols / kSFPackN;
        const auto mask_b16 = asc_create_mask_b16(PAT_ALL);

        for (uint32_t m_base = 0; m_base < kNumRows; m_base += kNumRowsPerVector) {
            vector_uint16_t row_indices;
            asc_arange(reinterpret_cast<vector_int16_t&>(row_indices), 0);
            const uint32_t num_rows = min(kNumRows - m_base, kNumRowsPerVector);
            for (uint32_t row_idx = 0; row_idx < num_rows; ++row_idx) {
                vector_bool row_mask;
                asc_eq_scalar(row_mask, row_indices, row_idx, mask_b16);

                for (uint32_t sf_group_idx = 0; sf_group_idx < kNumSFGroups; ++sf_group_idx) {
                    vector_uint16_t sf_exp_0, sf_exp_1;
                    const uint32_t value_idx = (m_base + row_idx) * kNumCols + sf_group_idx * kSFPackN;
                    cast_vector(values + value_idx, output + value_idx, sf_exp_0, sf_exp_1);
                    asc_shiftleft_scalar(sf_exp_1, sf_exp_1, 8, mask_b16);
                    asc_or(sf_exp_0, sf_exp_0, sf_exp_1, mask_b16);
                    asc_storealign(output_sf + sf_group_idx * kNumRows + m_base, sf_exp_0, row_mask);
                }
            }
        }
    }
};

} // namespace deep_gemm::epilogue::operators
