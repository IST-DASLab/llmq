import pytest
import torch
from pyllmq import kernels as K



@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fill_constant_basic(dtype: torch.dtype):
    val = 3.25
    x = torch.empty((8, 16), device="cuda", dtype=dtype)
    K.fill_constant(x, val)
    assert torch.allclose(x, torch.full_like(x, val))


@pytest.mark.parametrize("rows,cols", [(7, 11),  (1024, 2048)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_transpose_matches_torch(rows: int, cols: int, dtype: torch.dtype):
    src = torch.randn((rows, cols), device="cuda", dtype=dtype)
    dst = torch.empty((cols, rows), device="cuda", dtype=dtype)
    K.transpose(dst, src, rows, cols)
    ref = src.t()
    assert torch.allclose(dst, ref)


def test_abs_max_writes_scalar():
    x = torch.tensor([[-1.0, 2.5, -0.3], [0.7, -4.2, 3.3]], device="cuda", dtype=torch.float32)
    scale = torch.empty((), device="cuda", dtype=torch.float32)  # 0-dim scalar
    K.abs_max(scale, x)
    assert torch.isfinite(scale)
    assert torch.allclose(scale, torch.tensor(4.2, device="cuda", dtype=torch.float32))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(5, 13), (128, 24524)])
def test_global_norm_squared(dtype: torch.dtype, shape: tuple[int, int]):
    x = torch.randn(*shape, device="cuda", dtype=dtype)
    # TODO figure our correct block count
    out = torch.zeros((256,), device="cuda", dtype=torch.float32)
    K.global_norm_squared(out, x)
    ref = (x.float() ** 2).sum()
    assert pytest.approx(ref.item(), 1e-5, abs=1e-6) == out.sum().item()


def _rmsnorm_reference(inp, weight, eps):
    # inp: (B, T, C), weight: (C)
    var = (inp.float() ** 2).mean(dim=-1, keepdim=True)
    rms = torch.sqrt(var + eps)
    r_rms = 1.0 / rms
    out = inp.float() * r_rms * weight.float()
    return out.to(inp.dtype), r_rms.squeeze(-1)


@pytest.mark.parametrize("B,T,C", [(2, 3, 16), (1, 2, 64),  (8, 256, 1024)])
def test_rmsnorm_forward_matches_reference(B, T, C):
    torch.manual_seed(0)
    dtype = torch.float32  # kernel supports fp32 and bf16; use fp32 for numeric stability in tests
    inp = torch.randn((B, T, C), device="cuda", dtype=dtype)
    weight = torch.randn((C,), device="cuda", dtype=dtype)

    out = torch.empty_like(inp)
    rms = torch.empty((B, T), device="cuda", dtype=torch.float32)

    eps = 1e-6
    # absmax is optional; pass None
    K.rmsnorm_forward(out, rms, inp, weight, None, eps)

    ref_out, ref_rms = _rmsnorm_reference(inp, weight, eps)

    assert pytest.approx(ref_rms.cpu(), 5e-4, abs=5e-5) == rms.cpu()
    assert pytest.approx(ref_out.cpu(), 5e-4, abs=5e-5) == out.cpu()


def _swiglu_reference(x: torch.Tensor) -> torch.Tensor:
    # x: (B, T, 2*C) -> out: (B, T, C)
    a, b = torch.tensor_split(x.float(), 2, dim=-1)
    return (torch.nn.functional.silu(b) * a).to(x.dtype)


@pytest.mark.parametrize("B,T,C", [(1, 8, 128), (1, 1, 16), (4, 5, 32)])
def test_swiglu_forward_matches_reference(B, T, C):
    torch.manual_seed(123)
    dtype = torch.float32
    inp = torch.randn((B, T, 2 * C), device="cuda", dtype=dtype)
    out = torch.empty((B, T, C), device="cuda", dtype=dtype)

    # absmax is optional; pass None
    K.swiglu_forward(out, inp, None)

    ref = _swiglu_reference(inp)
    assert pytest.approx(ref.cpu(), 5e-4, 5e-5) == out.cpu()


@pytest.mark.parametrize("B,T", [(1, 512), (2, 1024), (7, 2048)])
def test_grouped_loss_sum_basic(B, T):
    # per-token losses, sum over sequence dimension per batch element
    losses = torch.rand((B, T), device="cuda", dtype=torch.float32)
    out = torch.empty((T//512,), device="cuda", dtype=torch.float32)
    K.grouped_loss_sum(out, losses)
    assert pytest.approx(losses.reshape((B, -1, 512)).sum(dim=2).sum(dim=0).cpu(), 1e-5, abs=1e-6) == out.cpu()


@pytest.mark.parametrize("mean,std", [(0.0, 1.0), (1.5, 0.25)])
def test_fill_normal_stats(mean, std):
    # Use large N for stable statistics
    N = 256 * 1024
    x = torch.empty((N,), device="cuda", dtype=torch.float32)
    seed = 0xABCDEF01
    K.fill_normal(x, float(mean), float(std), seed, 0)

    # Statistics
    m = x.mean().item()
    s = x.std(unbiased=True).item()

    # Tolerances scale with 1/sqrt(N); use generous but meaningful bounds
    assert abs(m - mean) < max(3e-3, 0.02 * abs(std))
    assert abs(s - std) / max(std, 1e-12) < 0.03

    # Determinism for same seed/subsequence
    y = torch.empty_like(x)
    K.fill_normal(y, float(mean), float(std), seed, 0)
    assert torch.allclose(x, y)
