"""Common utilities for GEMM tests: correctness metrics and perf reporting."""

import torch




def calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    """Cosine-similarity based diff, consistent with DeepGEMM."""
    x, y = x.float(), y.float()
    denominator = (x * x + y * y).sum()
    if denominator == 0:
        return 0.0
    sim = 2 * (x * y).sum() / denominator
    return (1 - sim).item()


def assert_equal(out: torch.Tensor, ref: torch.Tensor, msg: str,
                 slices: list[slice] = None, threshold: float = 0.0) -> None:
    """Check the kernel output against the aclnn reference, per group range if given (else
    the whole tensor). threshold == 0 demands bit-for-bit equality (FP8 MX: identical cube +
    cast path). threshold > 0 uses calc_diff (BF16: the cube kernel and aclnn differ by up to
    ~1 ULP from different K-accumulation, so an exact match is not expected)."""
    spans = slices if slices is not None else [slice(0, out.size(0))]
    for s in spans:
        if threshold == 0.0:
            assert torch.equal(out[s], ref[s]), f'{msg} (rows {s})'
        else:
            diff = calc_diff(out[s].float(), ref[s].float())
            assert diff < threshold, f'{msg} (rows {s}, diff={diff:.2e})'


def count_bytes(*tensors) -> int:
    """Total bytes across all tensors."""
    flat_tensors: list[torch.Tensor] = []
    for t in tensors:
        if isinstance(t, tuple):
            flat_tensors.extend(t)
        else:
            flat_tensors.append(t)
    return sum(t.numel() * t.element_size() for t in flat_tensors if t is not None)


def print_perf(m, n, k, tags, flops, bytes_moved, prof):
    """Print a compact DeepGEMM-style perf line. `tags` (e.g. 'layout=NT, fp32, acc=0')
    is folded into the dimension parens."""
    line = (f' > Perf (m={m:6}, n={n:6}, k={k:6}, {tags}): ')
    if prof is not None:
        line += (f'{prof.dur_us:9.3f} us | {prof.tflops(flops):4.0f} TFLOPS | {prof.gbps(bytes_moved):5.0f} GB/s | '
            f'mad={prof.aic_mad*100:4.1f}% | mte2={prof.aic_mte2*100:4.1f}%')
    else:
        line += 'OK'
    print(line, flush=True)
