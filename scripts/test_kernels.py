import pytest
import torch
from pyllmq import kernels as K


DTYPES = [torch.float32, torch.bfloat16]

# Tolerances by dtype: (rtol, atol)
TOL = {
    torch.float32: (5e-4, 5e-5),
    torch.bfloat16: (5e-3, 5e-3),
}


# ============================================================
# fill_constant
# ============================================================

@pytest.mark.parametrize("shape", [(8, 16), (1,), (128, 256), (4, 8, 32)])
@pytest.mark.parametrize("value", [0.0, 1.0, 3.25, -2.5])
@pytest.mark.parametrize("dtype", DTYPES)
def test_fill_constant_basic(shape, value, dtype):
    x = torch.empty(shape, device="cuda", dtype=dtype)
    K.fill_constant(x, value)
    assert x.cpu() == pytest.approx(torch.full(shape, value).cpu(), rel=0, abs=0)


# ============================================================
# transpose
# ============================================================

@pytest.mark.parametrize("rows,cols", [(7, 11), (1024, 2048), (64, 128), (3, 512)])
@pytest.mark.parametrize("dtype", DTYPES)
def test_transpose_matches_torch(rows, cols, dtype):
    src = torch.randn((rows, cols), device="cuda", dtype=dtype)
    dst = torch.empty((cols, rows), device="cuda", dtype=dtype)
    K.transpose(dst, src, rows, cols)
    assert dst.cpu() == pytest.approx(src.t().cpu(), rel=0, abs=0)


# ============================================================
# abs_max
# ============================================================

@pytest.mark.parametrize("shape", [(2, 3), (16, 64), (4, 128, 32)])
@pytest.mark.parametrize("dtype", DTYPES)
def test_abs_max_writes_scalar(shape, dtype):
    x = torch.randn(shape, device="cuda", dtype=dtype)
    # Plant a known maximum so we can assert the exact answer
    ref = x.abs().max()
    result = torch.empty((), device="cuda", dtype=torch.float32)
    K.abs_max(result, x)
    assert torch.isfinite(result)
    assert result.cpu() == pytest.approx(ref.cpu(), rel=0, abs=0)


# ============================================================
# global_norm_squared
# ============================================================

@pytest.mark.parametrize("shape", [(5, 13), (128, 24524), (256, 1024), (1, 4096)])
@pytest.mark.parametrize("dtype", DTYPES)
def test_global_norm_squared(shape, dtype):
    x = torch.randn(shape, device="cuda", dtype=dtype)
    out = torch.zeros((256,), device="cuda", dtype=torch.float32)
    K.global_norm_squared(out, x)
    ref = (x.float() ** 2).sum()
    rtol = 1e-5 if dtype == torch.float32 else 1e-2
    assert out.sum().cpu() == pytest.approx(ref.cpu(), rel=rtol)


# ============================================================
# rmsnorm_forward
# ============================================================

def _rmsnorm_reference(inp, weight, eps):
    var = (inp.float() ** 2).mean(dim=-1, keepdim=True)
    rms = torch.sqrt(var + eps)
    r_rms = 1.0 / rms
    out = inp.float() * r_rms * weight.float()
    return out.to(inp.dtype), r_rms.squeeze(-1)


@pytest.mark.parametrize("B,T,C", [(2, 3, 16), (1, 2, 64), (8, 256, 1024)])
@pytest.mark.parametrize("dtype", DTYPES)
def test_rmsnorm_forward_matches_reference(B, T, C, dtype):
    torch.manual_seed(0)
    inp = torch.randn((B, T, C), device="cuda", dtype=dtype)
    weight = torch.randn((C,), device="cuda", dtype=dtype)
    out = torch.empty_like(inp)
    rms = torch.empty((B, T), device="cuda", dtype=torch.float32)
    eps = 1e-6

    K.rmsnorm_forward(out, rms, inp, weight, None, eps)
    ref_out, ref_rms = _rmsnorm_reference(inp, weight, eps)

    rtol, atol = TOL[dtype]
    assert rms.cpu() == pytest.approx(ref_rms.cpu(), rel=rtol, abs=atol)
    assert out.float().cpu() == pytest.approx(ref_out.float().cpu(), rel=rtol, abs=atol)


# ============================================================
# swiglu_forward
# ============================================================

def _swiglu_reference(x: torch.Tensor) -> torch.Tensor:
    a, b = torch.tensor_split(x.float(), 2, dim=-1)
    return (torch.nn.functional.silu(b) * a).to(x.dtype)


@pytest.mark.parametrize("B,T,C", [(1, 8, 128), (1, 1, 16), (4, 5, 32), (2, 16, 256)])
@pytest.mark.parametrize("dtype", DTYPES)
def test_swiglu_forward_matches_reference(B, T, C, dtype):
    torch.manual_seed(123)
    inp = torch.randn((B, T, 2 * C), device="cuda", dtype=dtype)
    out = torch.empty((B, T, C), device="cuda", dtype=dtype)

    K.swiglu_forward(out, inp, None)

    ref = _swiglu_reference(inp)
    rtol, atol = TOL[dtype]
    assert out.float().cpu() == pytest.approx(ref.float().cpu(), rel=rtol, abs=atol)


# ============================================================
# grouped_loss_sum
# ============================================================

@pytest.mark.parametrize("B,T", [(1, 512), (2, 1024), (7, 2048), (4, 512)])
def test_grouped_loss_sum_basic(B, T):
    losses = torch.rand((B, T), device="cuda", dtype=torch.float32)
    out = torch.empty((T // 512,), device="cuda", dtype=torch.float32)
    K.grouped_loss_sum(out, losses)
    ref = losses.reshape(B, -1, 512).sum(dim=2).sum(dim=0)
    assert out.cpu() == pytest.approx(ref.cpu(), rel=1e-5, abs=1e-6)


# ============================================================
# fill_normal
# ============================================================

@pytest.mark.parametrize("mean,std", [(0.0, 1.0), (1.5, 0.25), (-1.0, 2.0)])
def test_fill_normal_stats(mean, std):
    N = 256 * 1024
    seed = 0xABCDEF01
    x = torch.empty((N,), device="cuda", dtype=torch.float32)
    K.fill_normal(x, float(mean), float(std), seed, 0)

    m = x.mean().item()
    s = x.std(unbiased=True).item()

    assert abs(m - mean) < max(3e-3, 0.02 * abs(std))
    assert abs(s - std) / max(std, 1e-12) < 0.03

    # Determinism: same seed + subsequence must give identical output
    y = torch.empty_like(x)
    K.fill_normal(y, float(mean), float(std), seed, 0)
    assert x.cpu() == pytest.approx(y.cpu())


# ============================================================
# rope forward + backward
# ============================================================

def _make_rope_freqs(T, head_dim, theta, dtype, device):
    assert head_dim % 2 == 0
    half = head_dim // 2
    idx = torch.arange(half, device=device, dtype=torch.float32)
    inv_freq = theta ** (-2 * idx / head_dim)
    t = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(1)
    angles = t * inv_freq.unsqueeze(0)
    cos = torch.cos(angles).to(dtype)
    sin = torch.sin(angles).to(dtype)
    return torch.stack([cos, sin], dim=-1).flatten(start_dim=1)  # (T, head_dim)


def _rope_python(x, freqs_cis, Nq, Nkv, backward=False):
    B, T, N, HD = x.shape
    half = HD // 2
    cos = freqs_cis[:, 0::2].float()
    sin = freqs_cis[:, 1::2].float()
    if backward:
        sin = -sin
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]

    q = x[:, :, :Nq, :]
    k = x[:, :, Nq:Nq + Nkv, :]
    v = x[:, :, Nq + Nkv:, :]

    def rotate(h):
        h = h.float()
        return torch.cat([
            h[..., :half] * cos - h[..., half:] * sin,
            h[..., :half] * sin + h[..., half:] * cos,
            ], dim=-1).to(x.dtype)

    return torch.cat([rotate(q), rotate(k), v], dim=2)


@pytest.mark.parametrize("B,T,Nq,Nkv,HD", [
    (1, 8, 2, 1, 8),
    (2, 4, 1, 2, 16),
    (2, 16, 4, 2, 32),
    (4, 32, 8, 4, 64),
])
@pytest.mark.parametrize("dtype", DTYPES)
def test_rope_forward_backward_matches_python(B, T, Nq, Nkv, HD, dtype):
    device = "cuda"
    C = (Nq + 2 * Nkv) * HD
    x = torch.randn((B, T, C), device=device, dtype=dtype)
    out = torch.empty_like(x)

    freq_dtype = torch.float32 if dtype == torch.float32 else torch.float16
    freqs = _make_rope_freqs(T, HD, 1_000_000.0, freq_dtype, device)

    K.rope_forward(out, x, freqs, None, Nq, Nkv)
    ref = _rope_python(x.view(B, T, Nq + 2 * Nkv, HD), freqs, Nq, Nkv).view(B, T, C)

    rtol, atol = TOL[dtype]
    assert out.float().cpu() == pytest.approx(ref.float().cpu(), rel=rtol, abs=atol)

    dout = torch.randn_like(x)
    dinp = torch.empty_like(x)
    K.rope_backward(dinp, dout, freqs, None, Nq, Nkv)
    ref_bw = _rope_python(dout.view(B, T, Nq + 2 * Nkv, HD), freqs, Nq, Nkv, backward=True).view(B, T, C)
    assert dinp.float().cpu() == pytest.approx(ref_bw.float().cpu(), rel=rtol, abs=atol)


# ============================================================
# fused_residual_rmsnorm_forward
# ============================================================

@pytest.mark.parametrize("B,T,C", [(2, 5, 64), (1, 3, 256), (4, 8, 128), (8, 16, 512)])
@pytest.mark.parametrize("dtype", DTYPES)
def test_fused_residual_rmsnorm_forward_reference(B, T, C, dtype):
    device = "cuda"
    torch.manual_seed(0)
    inp1 = torch.randn((B, T, C), device=device, dtype=dtype)
    inp2 = torch.randn_like(inp1)
    weight = torch.randn((C,), device=device, dtype=dtype)
    residual = torch.empty_like(inp1)
    normed = torch.empty_like(inp1)
    rrms = torch.empty((B, T), device=device, dtype=torch.float32)
    eps = 1e-6

    K.fused_residual_rmsnorm_forward(residual, normed, rrms, inp1, inp2, weight, None, eps)

    res_ref = (inp1.float() + inp2.float()).to(dtype)
    var = (res_ref.float() ** 2).mean(dim=-1, keepdim=True)
    r_rms = 1.0 / torch.sqrt(var + eps)
    norm_ref = (res_ref.float() * r_rms * weight.float()).to(dtype)

    rtol, atol = TOL[dtype]
    assert residual.cpu() == pytest.approx(res_ref.cpu(), rel=rtol, abs=atol)
    assert normed.float().cpu() == pytest.approx(norm_ref.float().cpu(), rel=rtol, abs=atol)
    assert rrms.cpu() == pytest.approx(r_rms.squeeze(-1).float().cpu(), rel=rtol, abs=atol)


# ============================================================
# vector_add_sr
# ============================================================

@pytest.mark.parametrize("nelem", [4096, 16384, 65536])
@pytest.mark.parametrize("dtype", DTYPES)
def test_vector_add_sr_determinism_and_accuracy(nelem, dtype):
    device = "cuda"
    a = torch.randn((nelem,), device=device, dtype=dtype)
    b = torch.randn_like(a)
    out1 = torch.empty_like(a)
    out2 = torch.empty_like(a)
    seed = 12345
    scale = torch.tensor(0.75, dtype=torch.float32)
    K.vector_add_sr(out1, a, b, scale, seed)
    K.vector_add_sr(out2, a, b, scale, seed)

    # Determinism
    assert out1.cpu() == pytest.approx(out2.cpu())

    # Accuracy vs fp32 reference; stochastic rounding may introduce ~1 ulp at target dtype
    ref = scale.item() * (a.float() + b.float())
    assert out1.float().cpu() == pytest.approx(ref.cpu(), rel=5e-3, abs=5e-3)


# ============================================================
# quantize_with_abs_max
# ============================================================

@pytest.mark.parametrize("N", [1024, 8192, 65536])
@pytest.mark.parametrize("dtype", DTYPES)
def test_quantize_with_abs_max_bf16(N, dtype):
    device = "cuda"
    x = torch.randn((N,), device=device, dtype=dtype)
    abs_max_val = torch.tensor([x.float().abs().max().item()], device=device, dtype=torch.float32)
    out = torch.empty((N,), device=device, dtype=torch.bfloat16)
    scale = torch.empty((), device=device, dtype=torch.float32)
    K.quantize_with_abs_max(out, scale, x, abs_max_val)
    assert scale.item() == pytest.approx(1.0)
    assert out.float().cpu() == pytest.approx(x.bfloat16().float().cpu(), rel=0.01)


@pytest.mark.parametrize("N", [1024, 8192])
@pytest.mark.parametrize("dtype", DTYPES)
def test_quantize_with_abs_max_fp8(N, dtype):
    device = "cuda"
    x = torch.randn((N,), device=device, dtype=dtype)
    abs_max_val = torch.tensor([x.float().abs().max().item()], device=device, dtype=torch.float32)
    out = torch.empty((N,), device=device, dtype=torch.float8_e4m3fn)
    scale = torch.empty((), device=device, dtype=torch.float32)
    K.quantize_with_abs_max(out, scale, x, abs_max_val)
    assert scale.item() == pytest.approx(abs_max_val.item() / 448.0)
    dequant = out.float() * scale
    assert dequant.cpu() == pytest.approx(x.float().cpu(), rel=0.1)
