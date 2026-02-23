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


# ---------------- Rope ----------------

def _make_rope_freqs(T: int, head_dim: int, theta: float, dtype: torch.dtype, device: str):
    assert head_dim % 2 == 0
    half = head_dim // 2
    idx = torch.arange(half, device=device, dtype=torch.float32)
    inv_freq = theta ** (-2 * idx / head_dim)
    t = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(1)
    angles = t * inv_freq.unsqueeze(0)
    cos = torch.cos(angles).to(dtype)
    sin = torch.sin(angles).to(dtype)
    freqs = torch.stack([cos, sin], dim=-1).flatten(start_dim=1)  # (T, head_dim)
    return freqs


def _rope_python(x, freqs_cis, Nq: int, Nkv: int, backward: bool = False):
    """
    x:         (B, T, Nq+2*Nkv, HD)
    freqs_cis: (T, HD) interleaved [cos0, sin0, cos1, sin1, ...]
    """
    B, T, N, HD = x.shape
    half = HD // 2

    cos = freqs_cis[:, 0::2].float()  # (T, half)
    sin = freqs_cis[:, 1::2].float()  # (T, half)
    if backward:
        sin = -sin

    cos = cos[None, :, None, :]  # (1, T, 1, half)
    sin = sin[None, :, None, :]

    # split into query, key, value
    q = x[:, :, :Nq, :]
    k = x[:, :, Nq:Nq+Nkv, :]
    v = x[:, :, Nq+Nkv:, :]

    def rotate(h):
        h = h.float()
        return torch.cat([
            h[..., :half] * cos - h[..., half:] * sin,
            h[..., :half] * sin + h[..., half:] * cos,
            ], dim=-1).to(x.dtype)

    return torch.cat([rotate(q), rotate(k), v], dim=2)

@pytest.mark.parametrize("B,T,Nq,Nkv,HD", [(1, 8, 2, 1, 8), (2, 4, 1, 2, 16)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rope_forward_backward_matches_python(B, T, Nq, Nkv, HD, dtype):
    device = "cuda"
    theta = 1_000_000.0
    C = (Nq + 2 * Nkv) * HD
    x = torch.randn((B, T, C), device=device, dtype=dtype)
    out = torch.empty_like(x)

    # freqs dtype must match kernel expectations: fp32 for fp32, fp16 for bf16
    freq_dtype = torch.float32 if dtype == torch.float32 else torch.float16
    freqs = _make_rope_freqs(T, HD, theta, freq_dtype, device)

    # optional absmax path
    K.rope_forward(out, x, freqs, None, Nq, Nkv)
    ref = _rope_python(x.view(B, T, Nq + 2 * Nkv, HD), freqs, Nq, Nkv, backward=False).view(B, T, C)
    assert out.float().cpu() == pytest.approx(ref.float().cpu(), 5e-4, abs=5e-5)

    dout = torch.randn_like(x)
    dinp = torch.empty_like(x)
    K.rope_backward(dinp, dout, freqs, None, Nq, Nkv)
    ref_bw = _rope_python(dout.view(B, T, Nq + 2 * Nkv, HD), freqs, Nq, Nkv, backward=True).view(B, T, C)
    assert dinp.float().cpu() == pytest.approx(ref_bw.float().cpu(), 5e-4, abs=5e-5)


@pytest.mark.parametrize("B,T,C", [(2, 5, 64), (1, 3, 256)])
@pytest.mark.parametrize("dtype", [torch.float32])
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

    res_ref = inp1.float() + inp2.float()
    var = (res_ref ** 2).mean(dim=-1, keepdim=True)
    rms = torch.sqrt(var + eps)
    r_rms = 1.0 / rms
    norm_ref = (res_ref * r_rms * weight.float()).to(dtype)

    assert torch.allclose(residual, res_ref.to(dtype))
    assert pytest.approx(norm_ref.cpu(), 5e-4, abs=5e-5) == normed.cpu()
    assert pytest.approx(r_rms.squeeze(-1).cpu(), 5e-4, abs=5e-5) == rrms.cpu()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_vector_add_sr_determinism_and_accuracy(dtype):
    device = "cuda"
    nelem = 4096
    a = torch.randn((nelem,), device=device, dtype=dtype)
    b = torch.randn_like(a)
    out1 = torch.empty_like(a)
    out2 = torch.empty_like(a)
    seed = 12345
    scale = torch.tensor(0.75, dtype=torch.float32)  # CPU 0-dim for binding float conversion
    K.vector_add_sr(out1, a, b, scale, seed)
    K.vector_add_sr(out2, a, b, scale, seed)
    # determinism
    assert torch.equal(out1, out2)
    # accuracy vs fp32 add scaled
    ref = scale.item() * (a.float() + b.float())
    # stochastic rounding may introduce 1 ulp error at target dtype
    assert torch.allclose(out1.float(), ref, rtol=5e-3, atol=5e-3)


def test_quantize_with_abs_max_bf16():
    device = "cuda"
    N = 8192
    x = torch.randn((N,), device=device, dtype=torch.float32)
    abs_max = torch.tensor([x.abs().max().item()], device=device, dtype=torch.float32)
    # Use bf16 path for broader dtype support
    out = torch.empty((N,), device=device, dtype=torch.bfloat16)
    scale = torch.empty((), device=device, dtype=torch.float32)
    K.quantize_with_abs_max(out, scale, x, abs_max)
    assert scale.item() == 1.0
    assert out.cpu().float() == pytest.approx(x.bfloat16().float().cpu(),  rel=0.01)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_quantize_with_abs_max_fp8(dtype: torch.dtype):
    device = "cuda"
    N = 8192
    x = torch.randn((N,), device=device, dtype=dtype)
    abs_max = torch.tensor([x.abs().max().item()], device=device, dtype=torch.float32)
    # Use bf16 path for broader dtype support
    out = torch.empty((N,), device=device, dtype=torch.float8_e4m3fn)
    scale = torch.empty((), device=device, dtype=torch.float32)
    K.quantize_with_abs_max(out, scale, x, abs_max)
    assert scale.item() == pytest.approx(abs_max.item() / 448.0)
    dequant = out.float() * scale
    assert dequant.cpu() == pytest.approx(x.float().cpu(), rel=0.1)
