import enum
import gc
import itertools
import os
import random
import time
import torch

import deep_gemm
from deep_gemm.testing.bench import bench_msprof, close_persistent_profiler
from deep_gemm.testing.par_compile import par_compile
from utils import calc_diff, count_bytes


class Mode(enum.Enum):
    """How each row's valid KV range [ks, ke) is generated."""
    FullRange = enum.auto()  # ks=0, ke=seq_len_kv (whole sequence)
    Causal    = enum.auto()  # single-machine causal (ke = arange + offset)
    Cp        = enum.auto()  # context-parallel chunked causal
    Random    = enum.auto()  # per-row random interval [min(a, b), max(a, b))

    def __str__(self) -> str: return self.name


DEFAULT_MODES = (Mode.FullRange, Mode.Cp, Mode.Causal, Mode.Random)

MQA_BOUNDARY_NUM_HEADS = tuple(range(4, 65, 4))
MQA_PERF_NUM_HEADS = (4, 8, 12, 20, 16, 32, 64)


def ceil_to_ue8m0_with_byte(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    bits = x.abs().float().view(torch.int)
    exp = (((bits >> 23) & 0xFF) + (bits & 0x7FFFFF).bool().int()).clamp(1, 254)
    return (exp << 23).view(torch.float), exp.to(torch.uint8)


def quantize_to_fp4_raw(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    ax = x.abs()
    code = torch.zeros_like(x, dtype=torch.uint8)
    for boundary in (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0):
        code += (ax > boundary).to(torch.uint8)
    sign = (x < 0) & (code != 0)
    values = torch.tensor((0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0), device=x.device, dtype=torch.float32)
    repr_f32 = values[code.long()]
    repr_f32 = torch.where(sign, -repr_f32, repr_f32)
    code = code | (sign.to(torch.uint8) * 8)
    packed = (code[..., ::2] & 0x0f) | ((code[..., 1::2] & 0x0f) << 4)
    return packed.contiguous().view(torch.int8), repr_f32


def quantize_mqa_tensor(x: torch.Tensor, use_fp4: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sf_div = 6.0 if use_fp4 else 448.0
    sf_shape = (*x.shape[:-1], x.shape[-1] // 32, 32)
    sf, sf_byte = ceil_to_ue8m0_with_byte(x.float().reshape(sf_shape).abs().amax(dim=-1).clamp(1e-4) / sf_div)
    x_scaled = (x.float().reshape(sf_shape) / sf[..., None]).reshape_as(x)
    if use_fp4:
        x_quant, x_repr = quantize_to_fp4_raw(x_scaled)
    else:
        x_quant = x_scaled.to(torch.float8_e4m3fn)
        x_repr = x_quant.float()
    x_ref = (x_repr.reshape(sf_shape) * sf[..., None]).reshape_as(x)
    sf_packed = sf_byte[..., ::2].to(torch.int16) | (sf_byte[..., 1::2].to(torch.int16) << 8)
    return x_quant, sf_packed.contiguous(), x_ref


def ref_mqa_logits(q: torch.Tensor, kv: torch.Tensor, weights: torch.Tensor,
                   cu_seq_len_k_start: torch.Tensor, cu_seq_len_k_end: torch.Tensor) -> torch.Tensor:
    seq_len, num_heads, _ = q.shape
    seq_len_kv = kv.shape[0]
    q_f32 = q.float()
    kv_f32 = kv.float()
    weights_f32 = weights.float()

    logits = torch.empty((seq_len, seq_len_kv), dtype=torch.float32, device=q.device)
    kv_positions = torch.arange(seq_len_kv, device=q.device)
    split_kv = int(os.getenv('DG_MQA_SPLIT_KV', '256'))
    for kv_base in range(0, seq_len_kv, split_kv):
        kv_end = min(kv_base + split_kv, seq_len_kv)
        score = torch.einsum('qhd,kd->qhk', q_f32, kv_f32[kv_base:kv_end])
        out = (score.relu() * weights_f32[:, :, None]).sum(dim=1)
        mask = ((kv_positions[kv_base:kv_end][None, :] >= cu_seq_len_k_start[:, None]) &
                (kv_positions[kv_base:kv_end][None, :] < cu_seq_len_k_end[:, None]))
        logits[:, kv_base:kv_end] = out.masked_fill(~mask, float('-inf'))
    return logits


def generate_ks_ke(seq_len: int, seq_len_kv: int, mode: Mode) -> tuple[torch.Tensor, torch.Tensor, int]:
    ks = torch.zeros((seq_len,), device='npu', dtype=torch.int32)
    if mode == Mode.Random:
        a = torch.randint(0, seq_len_kv + 1, (seq_len,), device='npu', dtype=torch.int32)
        b = torch.randint(0, seq_len_kv + 1, (seq_len,), device='npu', dtype=torch.int32)
        ks = torch.minimum(a, b)
        ke = torch.maximum(a, b)
    elif mode == Mode.FullRange:
        ke = torch.full((seq_len,), seq_len_kv, device='npu', dtype=torch.int32)
    elif mode == Mode.Causal:
        ke = torch.arange(seq_len, device='npu', dtype=torch.int32) + (seq_len_kv - seq_len)
    else:  # Mode.Cp
        assert seq_len_kv % seq_len == 0 and seq_len % 2 == 0
        cp_size = seq_len_kv // seq_len
        chunk_size = seq_len // 2
        cp_id = cp_size // 3
        ke = torch.empty((seq_len,), device='npu', dtype=torch.int32)
        offsets = torch.arange(chunk_size, device='npu', dtype=torch.int32)
        ke[:chunk_size] = cp_id * chunk_size + offsets
        ke[chunk_size:] = (cp_size * 2 - 1 - cp_id) * chunk_size + offsets
    return ks, ke, int((ke - ks).max().item())


def run_mqa_logits_case(seq_len: int, seq_len_kv: int, num_heads: int, head_dim: int,
                        skip_prof: bool, use_fp4: bool = False,
                        mode: Mode = Mode.FullRange, collect: list = None) -> None:
    dtype_name = 'FP4' if use_fp4 else 'FP8'
    case_tag = f'SQ={seq_len:4}, SK={seq_len_kv:5}, H={num_heads:2}, D={head_dim:3}, {str(mode):9}'

    fill = torch.ones if os.getenv('DG_MQA_TEST_PATTERN') == 'ones' else torch.randn
    q_bf16 = fill((seq_len, num_heads, head_dim), device='npu', dtype=torch.bfloat16)
    kv_bf16 = fill((seq_len_kv, head_dim), device='npu', dtype=torch.bfloat16)
    weights = fill((seq_len, num_heads), device='npu', dtype=torch.bfloat16)
    cu_seq_len_k_start, cu_seq_len_k_end, max_seqlen_k = generate_ks_ke(seq_len, seq_len_kv, mode)
    valid_len = cu_seq_len_k_end - cu_seq_len_k_start

    q, q_sf, q_ref = quantize_mqa_tensor(q_bf16, use_fp4)
    kv, kv_sf, kv_ref = quantize_mqa_tensor(kv_bf16, use_fp4)

    def run_kernel():
        return deep_gemm.fp8_fp4_mqa_logits(
            (q, q_sf), (kv, kv_sf), weights, cu_seq_len_k_start, cu_seq_len_k_end, max_seqlen_k)

    if collect is not None:
        collect.append(run_kernel)
        return

    ref_bf16 = ref_mqa_logits(q_bf16, kv_bf16, weights, cu_seq_len_k_start, cu_seq_len_k_end).to(torch.bfloat16)
    ref = ref_mqa_logits(q_ref, kv_ref, weights, cu_seq_len_k_start, cu_seq_len_k_end).to(torch.bfloat16)
    kernel_out = run_kernel()
    assert kernel_out.dtype == torch.bfloat16
    assert kernel_out.stride(0) * kernel_out.element_size() % 512 == 0 and kernel_out.stride(1) == 1
    assert kernel_out.shape == (seq_len, max_seqlen_k)
    out = torch.full((seq_len, seq_len_kv), float('-inf'), device='npu', dtype=torch.bfloat16)
    starts = cu_seq_len_k_start.cpu().tolist()
    ends = cu_seq_len_k_end.cpu().tolist()
    for i, (start, end) in enumerate(zip(starts, ends)):
        out[i, start:end] = kernel_out[i, :end - start]

    # Compressed rows define only their valid prefixes; compare those via a dense masked view.
    ref_f, out_f = ref.float(), out.float()
    ref_neginf = torch.isinf(ref_f) & (ref_f < 0)
    assert torch.equal(torch.isinf(out_f) & (out_f < 0), ref_neginf), 'logits -inf mask mismatch'
    out = out.masked_fill(ref_neginf, 0)
    ref = ref.masked_fill(ref_neginf, 0)
    ref_bf16 = ref_bf16.masked_fill(ref_neginf, 0)
    if os.getenv('DG_MQA_TEST_PATTERN') == 'ones':
        out_f = out.float()
        ref_ones = ref.float()
        assert torch.isfinite(out_f).all(), 'ones MQA logits produced non-finite output'
        max_err = (out_f - ref_ones).abs().max().item()
        if max_err >= 1e-5:
            print(' > Ones full-reduce failure:')
            print(f'   shape={tuple(out.shape)}, dtype={out.dtype}, heads={num_heads}, D={head_dim}, max_err={max_err:.3e}')
            print(f'   out min/max={out_f.min().item():.6e}/{out_f.max().item():.6e}')
            print(f'   out rows[0:8,0:16]=\n{out_f[:8, :16]}')
        assert max_err < 1e-5, f'ones full reduce max_err {max_err:.3e}'
        print(f' > {dtype_name} ones full reduce ({case_tag}): OK')
        return

    orig_diff = calc_diff(out.float(), ref_bf16.float())
    diff = calc_diff(out.float(), ref.float())
    orig_tol = 2e-2 if use_fp4 else 1e-3
    tol = 5e-4
    if (torch.isnan(out).any() or torch.isinf(out).any() or torch.isnan(torch.tensor(diff)) or
            torch.isnan(torch.tensor(orig_diff)) or diff >= tol or orig_diff >= orig_tol):
        out_f = out.float()
        ref_f = ref.float()
        finite = torch.isfinite(out_f)
        print(' > Debug MQA logits failure:')
        print(f'   {case_tag}, shape={tuple(out.shape)}, dtype={out.dtype}, '
              f'sim_diff={diff:.3e}/{tol:.1e}, orig_diff={orig_diff:.3e}/{orig_tol:.1e}')
        print(f'   out nan={torch.isnan(out_f).sum().item()}, inf={torch.isinf(out_f).sum().item()}, finite={finite.sum().item()}/{out.numel()}')
        print(f'   ref nan={torch.isnan(ref_f).sum().item()}, inf={torch.isinf(ref_f).sum().item()}')
        if finite.any():
            print(f'   out finite min/max={out_f[finite].min().item():.6e}/{out_f[finite].max().item():.6e}')
            print(f'   ref finite min/max={ref_f[torch.isfinite(ref_f)].min().item():.6e}/{ref_f[torch.isfinite(ref_f)].max().item():.6e}')
            err = (out_f - ref_f).abs()
            print(f'   abs err max/mean={err.max().item():.6e}/{err.mean().item():.6e}')
        bad = ~finite
        if bad.any():
            bad_idx = bad.nonzero()[0].tolist()
            q_idx, kv_idx = bad_idx
            q0, q1 = max(0, q_idx - 1), min(seq_len, q_idx + 2)
            k0, k1 = max(0, kv_idx - 8), min(seq_len_kv, kv_idx + 8)
            print(f'   first bad index={bad_idx}')
            print(f'   out window=\n{out_f[q0:q1, k0:k1]}')
            print(f'   ref window=\n{ref_f[q0:q1, k0:k1]}')
        else:
            print(f'   out first row first 16={out_f[0, :16]}')
            print(f'   ref first row first 16={ref_f[0, :16]}')
            print(f'   out first row first 64={out_f[0, :64]}')
            print(f'   ref first row first 64={ref_f[0, :64]}')
            print(f'   out first 4 rows first 16=\n{out_f[:4, :16]}')
            print(f'   ref first 4 rows first 16=\n{ref_f[:4, :16]}')
    assert orig_diff < orig_tol, f'MQA logits original-bf16 diff {orig_diff:.3e} exceeds {orig_tol:.1e}'
    assert diff < tol, f'MQA logits simulated diff {diff:.3e} exceeds {tol:.1e}'

    if not skip_prof:
        prof = bench_msprof(run_kernel, kernel_names='mqa_logits')
        valid_sum = valid_len.sum().item()
        flops = 2.0 * valid_sum * num_heads * head_dim
        bytes_moved = count_bytes(q, kv, q_sf, kv_sf, weights, kernel_out,
                                  cu_seq_len_k_start, cu_seq_len_k_end)
        print(f' > {dtype_name} Perf ({case_tag}) {prof.dur_us:7.1f} us | {prof.tflops(flops):4.0f} TFLOPS | '
              f'{prof.gbps(bytes_moved):5.0f} GB/s | '
              f'AIC[mad={prof.aic_mad*100:4.1f}%, mte2={prof.aic_mte2*100:4.1f}%, '
              f'fix={prof.aic_fixpipe*100:4.1f}%] | AIV[vec={prof.aiv_vec*100:4.1f}%]')
    else:
        print(f' > {dtype_name} ({case_tag}): OK')


def compile_mqa_kernels(run_all) -> None:
    kernels = []
    run_all(kernels)
    par_compile(kernels)
    del kernels
    gc.collect()
    torch.npu.empty_cache()


def test_mqa_logits(case_specs, skip_prof: bool, *, num_heads, use_fp4, head_dims) -> None:
    seq_len = int(os.getenv('DG_MQA_SQ', '4096'))
    seq_len_kv = int(os.getenv('DG_MQA_SK', '8192'))
    cases = tuple(
        (seq_len, seq_len_kv, heads, head_dim, skip_prof, fp4, mode)
        for mode, fp4, head_dim, heads in itertools.product(case_specs, use_fp4, head_dims, num_heads)
    )

    def run_all(collect=None):
        for case in cases:
            run_mqa_logits_case(*case, collect=collect)

    print(f'Testing MQA logits ({len(cases)} cases):')
    compile_mqa_kernels(run_all)

    run_all()
    print()


def test_mqa_logits_boundaries() -> None:
    full_seq_len, head_dim = 32, 128
    cases = []
    split_kv = 512
    for delta in (-1, 1):
        # FullRange forces the split+1 case to schedule and store its final one-token split.
        modes = (Mode.FullRange, Mode.Causal) if delta < 0 else (Mode.FullRange,)
        for mode in modes:
            cases.append((full_seq_len, split_kv + delta, 32, head_dim, True, delta > 0, mode))
    cases.append((512 // 32 + 1, split_kv, 32, head_dim, True, False, Mode.FullRange))
    for heads in MQA_BOUNDARY_NUM_HEADS:
        cases.append((full_seq_len, split_kv, heads, head_dim, True, False, Mode.FullRange))
    cases.append((full_seq_len, split_kv, 8, head_dim, True, True, Mode.Random))

    def run_all(collect=None):
        for case in cases:
            run_mqa_logits_case(*case, collect=collect)

    print('Testing MQA logits boundaries:')
    compile_mqa_kernels(run_all)
    run_all()
    print()


def ref_paged_mqa_logits(q, kv, weights, context_lens, block_table, max_context_len):
    num_q_tokens = q.size(0)
    page_size = kv.size(1)
    out = torch.zeros((num_q_tokens, max_context_len), device=q.device)
    for q_idx, (context_len,) in enumerate(context_lens.cpu().tolist()):
        pages = (context_len + page_size - 1) // page_size
        keys = kv[block_table[q_idx, :pages].long()].flatten(0, 1).float()
        scores = torch.einsum('hd,kd->hk', q[q_idx, 0].float(), keys)
        logits = (scores.relu() * weights[q_idx, :, None]).sum(0)
        out[q_idx, :context_len] = logits[:context_len]
    return out


def reorder_paged_mqa_pages(pages, order):
    assert order in ('random', 'forward', 'reverse', 'pair_swap')
    if order == 'reverse':
        return pages.flip(0)
    if order == 'pair_swap':
        paired = pages.numel() // 2 * 2
        return torch.cat((pages[:paired].view(-1, 2).flip(1).flatten(),
                          pages[paired:]))
    return pages


def run_paged_mqa_logits_case(batch_size, num_heads, head_dim, avg_kv, use_fp4, request_tokens,
                              page_size=64, use_ascend_kv_layout=False, *, skip_prof=False,
                              random_lens=False, request_context_lens=None,
                              page_order='random', collect: list = None):
    # Drop the resident profiler so this case's reference traffic stays out of the PMU channel
    close_persistent_profiler()
    assert len(request_tokens) == batch_size
    request_tokens = torch.tensor(request_tokens, device='npu', dtype=torch.int32)
    min_tokens, max_tokens = int(request_tokens.min()), int(request_tokens.max())
    assert min_tokens >= 1
    request_tag = (f'TPR={min_tokens}' if min_tokens == max_tokens else
                   f'TPR={min_tokens}-{max_tokens}')
    indices = torch.arange(batch_size, device='npu', dtype=torch.int32).repeat_interleave(request_tokens)
    num_q_tokens = indices.numel()

    if request_context_lens is None:
        request_context_lens = (torch.randint(
            int(.7 * avg_kv), int(1.3 * avg_kv),
            (batch_size,), device='npu', dtype=torch.int32)
            if random_lens else
            torch.full((batch_size,), avg_kv, device='npu', dtype=torch.int32))
    else:
        assert len(request_context_lens) == batch_size
        request_context_lens = torch.tensor(request_context_lens, device='npu', dtype=torch.int32)
    offsets = torch.cat([torch.arange(int(n), device='npu', dtype=torch.int32)
                         for n in request_tokens.cpu()])
    context_lens = (request_context_lens.repeat_interleave(request_tokens) + offsets).view(-1, 1)
    request_context_lens += request_tokens - 1

    max_context_len = int(context_lens.max())
    pages_per_request = (request_context_lens + page_size - 1) // page_size
    max_pages, num_pages = int(pages_per_request.max()), int(pages_per_request.sum())
    q_bf16 = torch.randn((num_q_tokens, 1, num_heads, head_dim), device='npu', dtype=torch.bfloat16)
    kv_bf16 = torch.randn((num_pages, page_size, head_dim), device='npu', dtype=torch.bfloat16)
    weights = torch.randn((num_q_tokens, num_heads), device='npu', dtype=torch.bfloat16)
    q, q_sf, q_ref = quantize_mqa_tensor(q_bf16, use_fp4)
    kv, kv_sf, kv_ref = quantize_mqa_tensor(kv_bf16, use_fp4)
    if use_ascend_kv_layout:
        kv = kv.reshape(num_pages, page_size, -1, 32).permute(0, 2, 1, 3).contiguous().view_as(kv)
        kv_sf = kv_sf.reshape(num_pages, page_size // 16, 16, -1).permute(0, 1, 3, 2).contiguous().view(
            num_pages, -1, page_size)
    # Fused page layout: [page_size * data_bytes | page_size * sf_bytes] per page.
    data_bytes = kv.contiguous().view(torch.uint8).numel() // (num_pages * page_size)
    sf_bytes = kv_sf.contiguous().view(torch.uint8).numel() // (num_pages * page_size)
    fused = torch.cat([
        kv.contiguous().view(torch.uint8).view(num_pages, -1),
        kv_sf.contiguous().view(torch.uint8).view(num_pages, -1),
    ], dim=1).view(num_pages, page_size, 1, data_bytes + sf_bytes)

    pages = (torch.randperm(num_pages, device='npu', dtype=torch.int32)
             if page_order == 'random' else
             torch.arange(num_pages, device='npu', dtype=torch.int32))
    block_table = torch.zeros((batch_size, max_pages), device='npu', dtype=torch.int32)
    offset = 0
    for request, count in enumerate(pages_per_request.cpu().tolist()):
        block_table[request, :count] = reorder_paged_mqa_pages(
            pages[offset:offset + count], page_order)
        offset += count
    block_table = block_table.repeat_interleave(request_tokens, 0)

    def make_metadata():
        return deep_gemm.get_paged_mqa_logits_metadata(context_lens, num_heads, indices)

    def kernel(metadata):
        return deep_gemm.fp8_fp4_paged_mqa_logits(
            (q, q_sf), fused, weights, context_lens, block_table, metadata,
            max_context_len, indices, use_ascend_kv_layout)

    if collect is not None:
        collect.append(lambda: kernel(make_metadata()))
        return

    reference = ref_paged_mqa_logits(q_ref, kv_ref, weights, context_lens, block_table, max_context_len)
    metadata = make_metadata()
    out = kernel(metadata)
    assert out.stride(0) * out.element_size() % 512 == 0 and out.stride(1) == 1
    valid = torch.arange(max_context_len, device='npu')[None] < context_lens.view(-1, 1)
    diff = calc_diff(out.float().masked_fill(~valid, 0), reference.masked_fill(~valid, 0))
    tolerance = 5e-4
    assert out.shape == reference.shape and out.dtype == torch.bfloat16
    assert diff < tolerance, f'paged MQA simulated diff {diff:.3e} exceeds {tolerance:.1e}'
    assert torch.equal(kernel(metadata).masked_fill(~valid, 0), out.masked_fill(~valid, 0))

    profile = 'random' if random_lens else 'exact'
    tag = (f'Profile={profile:6}, Fmt={"fp4" if use_fp4 else "fp8":3}, Page={page_size:3}, '
           f'AscendKV={str(use_ascend_kv_layout):5}, Order={page_order:9}, BSZ={batch_size:3}, '
           f'{request_tag:8}, H={num_heads:2}, D={head_dim:3}')
    if skip_prof:
        print(f' > {tag}: OK')
        return
    prof = bench_msprof(lambda: kernel(metadata), kernel_names='paged_mqa_logits_impl')
    meta_prof = bench_msprof(make_metadata, kernel_names='paged_mqa_logits_metadata_simt_impl', flush_l2=False)
    accessed = int(context_lens.sum())
    unique_kv = int(request_context_lens.sum())
    flops = 2.0 * accessed * num_heads * head_dim
    # KV traffic is the unique rows per request, output writes one row per valid (query, context) pair
    kv_row_bytes = data_bytes + sf_bytes
    bytes_moved = (count_bytes(q, q_sf, weights, context_lens, block_table, indices) +
                   unique_kv * kv_row_bytes + accessed * out.element_size())
    print(f' > Dev={torch.npu.current_device():1}, {tag}, L={avg_kv:4}: '
          f'{prof.tflops(flops):6.1f} TFLOPS, {prof.dur_us:6.1f} us, '
          f'{prof.gbps(bytes_moved):5.0f} GB/s, '
          f'Metadata={meta_prof.dur_us:4.1f} us | '
          f'AIC[mad={prof.aic_mad*100:4.1f}%, mte2={prof.aic_mte2*100:4.1f}%, '
          f'fix={prof.aic_fixpipe*100:4.1f}%] | AIV[vec={prof.aiv_vec*100:4.1f}%]',
          flush=True)


def test_paged_mqa_logits(skip_prof, *, num_heads, head_dims, use_fp4,
                          min_tpr, max_tpr, page_sizes, use_ascend_kv_layout):
    assert 1 <= min_tpr <= max_tpr
    use_ascend_kv_layout = use_ascend_kv_layout if isinstance(use_ascend_kv_layout, tuple) else (use_ascend_kv_layout,)
    cases = []
    for batch_size, heads, head_dim, avg_kv, fp4, page_size, ascend_kv_layout in itertools.product(
            (256,), num_heads, head_dims, (8192,), use_fp4, page_sizes, use_ascend_kv_layout):
        request_tokens = tuple(torch.randint(min_tpr, max_tpr + 1, (batch_size,)).tolist())
        cases.append((batch_size, heads, head_dim, avg_kv, fp4, request_tokens, page_size, ascend_kv_layout))
    limit = int(os.getenv('DG_MQA_NUM_CASES', '0'))
    if limit:
        cases = random.Random(100000).sample(cases, min(limit, len(cases)))

    def run_all(collect=None):
        for case in cases:
            run_paged_mqa_logits_case(*case, skip_prof=skip_prof, random_lens=True, collect=collect)

    print(f'Testing MX Paged MQA Logits ({len(cases)} cases):')
    compile_mqa_kernels(run_all)
    run_all()
    print()


def test_paged_mqa_logits_boundaries():
    boundary_context_lens = (0, 1, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 513)
    batch_size = len(boundary_context_lens)
    request_tokens = (1, 8, 2, 7, 3, 6, 4, 5, 1, 8, 2, 7, 3, 6)
    # num_heads, head_dim, avg_kv, use_fp4, request_tokens, page_order
    base_cases = [(heads, 128, 512, False, (1,) * batch_size, 'random')
                  for heads in MQA_BOUNDARY_NUM_HEADS]
    base_cases += [(heads, 128, 512, True, request_tokens, 'random')
                   for heads in MQA_BOUNDARY_NUM_HEADS]
    base_cases.append((64, 64, 512, True, (6,) * batch_size, 'forward'))
    cases = tuple(
        (*case, page_size, use_ascend_kv_layout, page_order)
        for page_size in (64, 128)
        for use_ascend_kv_layout in (False, True)
        for *case, page_order in base_cases
    )

    def run_all(collect=None):
        for *args, page_order in cases:
            run_paged_mqa_logits_case(
                batch_size, *args, skip_prof=True,
                request_context_lens=boundary_context_lens,
                page_order=page_order, collect=collect)

    print('Testing MX Paged MQA Logits boundaries:')
    compile_mqa_kernels(run_all)
    run_all()
    print()



if __name__ == '__main__':
    from argparse import ArgumentParser

    def make_lookup(mapping):
        def lookup(s):
            try:
                return mapping[s.lower()]
            except KeyError:
                raise ValueError(s)  # argparse turns this into a friendly 'invalid choice'
        return lookup

    to_mode = make_lookup({mode.name.lower(): mode for mode in Mode})
    _formats = {'fp8': False, 'fp4': True}
    to_format = make_lookup(_formats)

    parser = ArgumentParser()
    parser.add_argument('select', nargs='*',
                        choices=('all', 'prefill', 'paged', 'boundary', 'paged_boundary',
                                 ),
                        help='Tests to run (default: all'
                             ')')
    parser.add_argument('--mode', nargs='+', type=to_mode, choices=list(Mode), default=None,
                        help='ks/ke modes to test, e.g. --mode Causal Cp (default: built-in point set)')
    parser.add_argument('--formats', nargs='+', type=to_format,
                        choices=list(_formats.values()), default=list(_formats.values()),
                        metavar='{fp8,fp4}',
                        help='Quantized input formats to test (default: both)')
    parser.add_argument('--num_heads', nargs='+', type=int, default=MQA_PERF_NUM_HEADS,
                        help='Head counts to test, e.g. --num_heads 4 8 (default: 4 8 12 20 16 32 64)')
    parser.add_argument('--head_dims', nargs='+', type=int, default=(128,),
                        help='Head dims to test, e.g. --head_dims 64 128 (default: 128)')
    parser.add_argument('--page_sizes', '--page-sizes', nargs='+', type=int, choices=(64, 128), default=(64, 128),
                        help='Page sizes to test for paged MQA logits (default: 64 128)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed (default: 0)')
    parser.add_argument('--min-tpr', type=int, default=3,
                        help='Minimum random tokens per request for paged MQA logits (default: 3)')
    parser.add_argument('--max-tpr', type=int, default=7,
                        help='Maximum random tokens per request for paged MQA logits (default: 7)')
    parser.add_argument('--use-ascend-kv-layout', action='store_true',
                        help='Use the Ascend-optimized KV cache layout (default: disabled)')
    parser.add_argument('--skip-prof', '--no-prof', dest='skip_prof', action='store_true')

    args = parser.parse_args()
    if args.min_tpr < 1 or args.max_tpr < args.min_tpr:
        parser.error('--min-tpr and --max-tpr must define a positive nonempty range')
    torch.manual_seed(args.seed)

    selected = set(args.select or ('all',))
    if 'all' in selected:
        selected = {'prefill', 'paged', 'boundary', 'paged_boundary'}
    if 'prefill' in selected:
        case_specs = args.mode if args.mode is not None else DEFAULT_MODES
        test_mqa_logits(case_specs, args.skip_prof, use_fp4=args.formats,
                        num_heads=args.num_heads, head_dims=args.head_dims)

    if 'paged' in selected:
        test_paged_mqa_logits(args.skip_prof, num_heads=args.num_heads,
                              head_dims=args.head_dims, use_fp4=args.formats,
                              min_tpr=args.min_tpr, max_tpr=args.max_tpr,
                              page_sizes=args.page_sizes,
                              use_ascend_kv_layout=args.use_ascend_kv_layout)

    if 'boundary' in selected:
        test_mqa_logits_boundaries()
    if 'paged_boundary' in selected:
        test_paged_mqa_logits_boundaries()
