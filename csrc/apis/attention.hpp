#pragma once

#include <optional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/torch.h>

#include <deep_jit/utils/exception.hpp>

#include "../jit_kernels/mqa_logits.hpp"
#include "../utils/dtype.hpp"

namespace deep_gemm::attention_api {

namespace py = pybind11;

constexpr uint32_t kLogitsStrideAlignBytes = 512;

static uint32_t get_logits_stride(uint32_t num_logits) {
    const uint32_t kLogitsAlignElems = kLogitsStrideAlignBytes / c10::elementSize(torch::kBFloat16);
    return ceil_div(num_logits, kLogitsAlignElems) * kLogitsAlignElems;
}

static torch::Tensor fp8_fp4_mqa_logits(
    const std::pair<torch::Tensor, std::optional<torch::Tensor>>& q,
    const std::pair<torch::Tensor, torch::Tensor>& kv,
    const torch::Tensor& weights,
    const torch::Tensor& cu_seq_len_k_start,
    const torch::Tensor& cu_seq_len_k_end,
    uint32_t max_seqlen_k
) {
    const auto& q_data = q.first;
    const auto& sf_q = q.second;
    const auto& kv_data = kv.first;
    const auto& sf_kv = kv.second;

    DJ_HOST_ASSERT(q_data.dim() == 3);
    DJ_HOST_ASSERT(kv_data.dim() == 2);
    DJ_HOST_ASSERT(weights.dim() == 2);
    DJ_HOST_ASSERT(cu_seq_len_k_start.is_contiguous() and cu_seq_len_k_end.is_contiguous());
    DJ_HOST_ASSERT(cu_seq_len_k_start.scalar_type() == torch::kInt and cu_seq_len_k_end.scalar_type() == torch::kInt);
    DJ_HOST_ASSERT(weights.scalar_type() == torch::kBFloat16, "Ascend MQA logits expects bf16 weights");

    const auto num_q_tokens = static_cast<uint32_t>(q_data.size(0));
    const auto num_heads = static_cast<uint32_t>(q_data.size(1));
    const uint32_t head_dim = q_data.scalar_type() == kPackedFP4 ? static_cast<uint32_t>(q_data.size(2)) * 2u : static_cast<uint32_t>(q_data.size(2));
    const auto num_kv_tokens = static_cast<uint32_t>(kv_data.size(0));
    const bool is_fp4 = q_data.scalar_type() == kPackedFP4;
    const uint32_t head_dim_storage = is_fp4 ? head_dim / 2u : head_dim;
    DJ_HOST_ASSERT(num_heads >= 4 and num_heads <= 64 and num_heads % 4 == 0);
    DJ_HOST_ASSERT(head_dim == 64 or head_dim == 128,
                   "Initial Ascend MQA logits kernel supports head_dim=64/128 only");
    DJ_HOST_ASSERT(max_seqlen_k > 0);
    const MQALogitsDesc desc = {
        .is_fp4 = is_fp4,
        .num_q_tokens = num_q_tokens,
        .num_kv_tokens = num_kv_tokens,
        .num_heads = num_heads,
        .head_dim = head_dim,
    };
    const MQALogitsConfig config = {
        .split_kv = 512u,
        .mad_m = 64u / num_heads * num_heads,
        .num_cd_stages = 2u,
        .num_q_stages = 2u,
        .num_kv_stages = 4u,
        .num_accum_stages = 2u,
        .num_logits_stages = 3u,
        .launch_options = {.num_blocks = runtime->get_num_sms()},
    };
    DJ_HOST_ASSERT(weights.size(0) == num_q_tokens and weights.size(1) == num_heads and weights.stride(1) == 1);
    DJ_HOST_ASSERT(cu_seq_len_k_start.numel() == num_q_tokens and cu_seq_len_k_end.numel() == num_q_tokens);
    DJ_HOST_ASSERT(q_data.is_contiguous() and kv_data.is_contiguous());
    DJ_HOST_ASSERT(kv_data.size(1) == head_dim_storage);

    DJ_HOST_ASSERT((q_data.scalar_type() == torch::kFloat8_e4m3fn and kv_data.scalar_type() == torch::kFloat8_e4m3fn) or
                   (q_data.scalar_type() == kPackedFP4 and kv_data.scalar_type() == kPackedFP4));
    DJ_HOST_ASSERT(sf_q.has_value(), "MQA logits requires Q scale factors");

    // TODO (important): refactor SF
    auto sf_q_view = sf_q.value();
    if (sf_q_view.dim() == 3) {
        DJ_HOST_ASSERT(sf_q_view.size(0) == num_q_tokens and sf_q_view.size(1) == num_heads);
        sf_q_view = sf_q_view.reshape({static_cast<int64_t>(num_q_tokens) * num_heads, sf_q_view.size(2)});
    }
    DJ_HOST_ASSERT(sf_q_view.scalar_type() == torch::kShort and sf_kv.scalar_type() == torch::kShort);
    DJ_HOST_ASSERT(sf_q_view.dim() == 2 and sf_q_view.size(0) == num_q_tokens * num_heads and
                   sf_q_view.size(1) == head_dim / 64);
    DJ_HOST_ASSERT(sf_kv.dim() == 2 and sf_kv.size(0) == num_kv_tokens and sf_kv.size(1) == head_dim / 64);
    auto sf_q_packed = sf_q_view;
    auto sf_kv_packed = sf_kv;
    if (sf_q_packed.stride(0) != 1 or sf_q_packed.stride(1) != static_cast<int64_t>(num_q_tokens) * num_heads)
        sf_q_packed = sf_q_packed.t().contiguous().t();
    if (sf_kv_packed.stride(0) != 1 or sf_kv_packed.stride(1) != static_cast<int64_t>(num_kv_tokens))
        sf_kv_packed = sf_kv_packed.t().contiguous().t();

    const uint32_t logits_stride = get_logits_stride(max_seqlen_k);
    auto logits = torch::empty_strided({num_q_tokens, max_seqlen_k}, {logits_stride, 1}, q_data.options().dtype(torch::kBFloat16));

    deep_gemm::launch_mqa_logits(desc, config, static_cast<uint32_t>(weights.stride(0)), logits_stride,
                                 q_data.data_ptr(), kv_data.data_ptr(), sf_q_packed.data_ptr(), sf_kv_packed.data_ptr(),
                                 weights.data_ptr(), logits.data_ptr(), cu_seq_len_k_start.data_ptr(),
                                 cu_seq_len_k_end.data_ptr());
    return logits;
}

static torch::Tensor get_paged_mqa_logits_metadata(
    const torch::Tensor& context_lens,
    uint32_t num_heads,
    const torch::Tensor& indices
) {
    DJ_HOST_ASSERT(context_lens.dim() == 2 and context_lens.size(1) == 1 and context_lens.is_contiguous() and
                   context_lens.scalar_type() == torch::kInt);
    const uint32_t num_q_tokens = static_cast<uint32_t>(context_lens.size(0));
    DJ_HOST_ASSERT(indices.dim() == 1 and indices.size(0) == num_q_tokens and
                   indices.scalar_type() == torch::kInt and indices.is_contiguous());

    const PagedMQALogitsConfig config = select_paged_mqa_logits_config(num_heads);
    const uint32_t num_cores = runtime->get_num_sms();
    auto metadata = torch::empty({num_cores + 1, 2}, context_lens.options());
    launch_paged_mqa_logits_metadata(num_q_tokens, 8 * config.mad_m / num_heads, config.split_kv,
                                     config.mad_m / num_heads, num_cores, context_lens.data_ptr(),
                                     indices.data_ptr(), metadata.data_ptr());
    return metadata;
}

static torch::Tensor fp8_fp4_paged_mqa_logits(
    const std::pair<torch::Tensor, std::optional<torch::Tensor>>& q,
    const torch::Tensor& fused_kv_cache,
    const torch::Tensor& weights,
    const torch::Tensor& context_lens,
    const torch::Tensor& block_table,
    const torch::Tensor& schedule_meta,
    uint32_t max_context_len,
    const torch::Tensor& indices,
    bool use_ascend_kv_layout
) {
    const auto& q_data = q.first;
    const auto& sf_q = q.second;

    DJ_HOST_ASSERT(q_data.dim() == 4 and fused_kv_cache.dim() == 4 and weights.dim() == 2);
    DJ_HOST_ASSERT(context_lens.dim() == 2 and context_lens.is_contiguous());
    DJ_HOST_ASSERT(block_table.dim() == 2 and block_table.stride(1) == 1);
    DJ_HOST_ASSERT(context_lens.scalar_type() == torch::kInt and block_table.scalar_type() == torch::kInt);
    DJ_HOST_ASSERT(schedule_meta.dim() == 2 and schedule_meta.is_contiguous() and
                   schedule_meta.scalar_type() == torch::kInt);
    DJ_HOST_ASSERT(weights.scalar_type() == torch::kBFloat16, "Ascend MQA logits expects bf16 weights");

    const uint32_t num_q_tokens = q_data.size(0);
    const uint32_t num_heads = q_data.size(2);
    const bool is_fp4 = q_data.scalar_type() == kPackedFP4;
    const uint32_t head_dim_storage = static_cast<uint32_t>(q_data.size(3));
    const uint32_t head_dim = is_fp4 ? head_dim_storage * 2u : head_dim_storage;
    const uint32_t page_size = fused_kv_cache.size(1);
    DJ_HOST_ASSERT(head_dim == 64 or head_dim == 128, "Ascend MQA logits kernel supports head_dim=64/128 only");
    DJ_HOST_ASSERT(page_size == 64 or page_size == 128, "Ascend paged MQA logits supports page_size=64/128");
    const uint32_t sf_pairs = head_dim / 64u;
    const uint32_t sf_bytes = sf_pairs * sizeof(int16_t);
    // NOTES: Shape is [num_pages, page_size, 1, head_dim_storage + sf_bytes].
    // Each physical page is [KV data | SF]; only stride(0) spans pages.
    DJ_HOST_ASSERT(fused_kv_cache.size(2) == 1 and fused_kv_cache.size(3) == head_dim_storage + sf_bytes and
                   fused_kv_cache.scalar_type() == torch::kByte);
    DJ_HOST_ASSERT(fused_kv_cache.stride(1) == head_dim_storage + sf_bytes and fused_kv_cache.stride(3) == 1 and
                   fused_kv_cache.stride(0) % sf_bytes == 0);
    const PagedMQALogitsDesc desc = {
        .is_fp4 = is_fp4,
        .num_q_tokens = num_q_tokens,
        .num_heads = num_heads,
        .head_dim = head_dim,
        .page_size = page_size,
    };
    const PagedMQALogitsConfig config = select_paged_mqa_logits_config(num_heads);

    DJ_HOST_ASSERT(q_data.size(1) == 1);
    DJ_HOST_ASSERT(q_data.scalar_type() == torch::kFloat8_e4m3fn or q_data.scalar_type() == kPackedFP4);
    DJ_HOST_ASSERT(q_data.is_contiguous());
    DJ_HOST_ASSERT(sf_q.has_value() and sf_q.value().scalar_type() == torch::kShort);
    DJ_HOST_ASSERT(indices.dim() == 1 and indices.size(0) == num_q_tokens and
                   indices.scalar_type() == torch::kInt and indices.is_contiguous());
    DJ_HOST_ASSERT(weights.size(0) == num_q_tokens and weights.size(1) == num_heads and weights.stride(1) == 1);
    DJ_HOST_ASSERT(context_lens.size(0) == num_q_tokens and context_lens.size(1) == 1);
    DJ_HOST_ASSERT(block_table.size(0) == num_q_tokens);

    auto sf_q_view = sf_q.value();
    if (sf_q_view.dim() == 4)
        sf_q_view = sf_q_view.reshape({static_cast<int64_t>(num_q_tokens) * num_heads,
                                       sf_q_view.size(3)});
    DJ_HOST_ASSERT(sf_q_view.dim() == 2 and sf_q_view.size(0) == num_q_tokens * num_heads and
                   sf_q_view.size(1) == sf_pairs);
    if (sf_q_view.stride(0) != 1 or
        sf_q_view.stride(1) != static_cast<int64_t>(num_q_tokens) * num_heads)
        sf_q_view = sf_q_view.t().contiguous().t();

    const uint32_t num_cores = runtime->get_num_sms();
    DJ_HOST_ASSERT(schedule_meta.size(0) == static_cast<int64_t>(num_cores) + 1 and schedule_meta.size(1) == 2);
    const uint32_t logits_stride = get_logits_stride(max_context_len);
    auto logits = torch::empty_strided({num_q_tokens, max_context_len}, {logits_stride, 1}, q_data.options().dtype(torch::kBFloat16));
    deep_gemm::launch_paged_mqa_logits(desc, config, use_ascend_kv_layout,
                                       static_cast<uint32_t>(weights.stride(0)), logits_stride,
                                       static_cast<uint32_t>(block_table.stride(0)),
                                       static_cast<uint64_t>(fused_kv_cache.stride(0)), q_data.data_ptr(),
                                       sf_q_view.data_ptr(), fused_kv_cache.data_ptr(), weights.data_ptr(),
                                       logits.data_ptr(), context_lens.data_ptr(), block_table.data_ptr(),
                                       indices.data_ptr(), schedule_meta.data_ptr());
    return logits;
}


static void register_apis(pybind11::module_& m) {

    m.def("fp8_fp4_mqa_logits", fp8_fp4_mqa_logits,
          py::arg("q"), py::arg("kv"), py::arg("weights"),
          py::arg("cu_seq_len_k_start"), py::arg("cu_seq_len_k_end"),
          py::arg("max_seqlen_k"));
    m.def("get_paged_mqa_logits_metadata", get_paged_mqa_logits_metadata,
          py::arg("context_lens"), py::arg("num_heads"), py::arg("indices"));
    m.def("fp8_fp4_paged_mqa_logits", fp8_fp4_paged_mqa_logits,
          py::arg("q"), py::arg("kv_cache"), py::arg("weights"),
          py::arg("context_lens"), py::arg("block_table"), py::arg("schedule_meta"),
          py::arg("max_context_len"), py::arg("indices"),
          py::arg("use_ascend_kv_layout") = false);
}

} // namespace deep_gemm::attention_api
