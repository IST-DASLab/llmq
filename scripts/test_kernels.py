import math

import pytest
import torch
import torch.nn.functional as F
from pyllmq import kernels as K


DTYPES = [torch.float32, torch.bfloat16]

# Tolerances by dtype for approximate kernels: (rtol, atol)
TOL = {
    torch.float32: (5e-4, 5e-5),
    torch.bfloat16: (1e-2, 5e-3),
}


# ============================================================
# fill_constant
# Fills every element with a compile-time constant: result must
# be bit-exact, so rel=0, abs=0.
# ============================================================

@pytest.mark.parametrize("shape", [(8, 16), (1,), (128, 256), (4, 8, 32)])
@pytest.mark.parametrize("value", [0.0, 1.0, 3.25, -2.5])
@pytest.mark.parametrize("dtype", DTYPES)
def test_fill_constant_basic(shape, value, dtype):
    x = torch.empty(shape, device="cuda", dtype=dtype)
    K.fill_constant(x, value)
    ref = torch.full(shape, value, dtype=dtype)
    # Exact: every element is independently written to the same constant.
    assert x.float().cpu() == pytest.approx(ref.float().cpu(), rel=0, abs=0)


# ============================================================
# transpose
# Byte-level shuffle — no arithmetic, so result must be exact.
# ============================================================

@pytest.mark.parametrize("rows,cols", [(7, 11), (1024, 2048), (64, 128), (3, 512)])
@pytest.mark.parametrize("dtype", DTYPES)
def test_transpose_matches_torch(rows, cols, dtype):
    src = torch.randn((rows, cols), device="cuda", dtype=dtype)
    dst = torch.empty((cols, rows), device="cuda", dtype=dtype)
    K.transpose(dst, src)
    # Transposing rearranges values without touching bits — must be exact.
    assert dst.float().cpu() == pytest.approx(src.t().float().cpu(), rel=0, abs=0)


# ============================================================
# abs_max
# Reduction over exact fp values — output is one of the input
# values, so the scalar result must be exact.
# ============================================================

@pytest.mark.parametrize("shape", [(16, 64), (4, 128, 32)])
@pytest.mark.parametrize("dtype", DTYPES)
def test_abs_max_writes_scalar(shape, dtype):
    x = torch.randn(shape, device="cuda", dtype=dtype)
    ref = x.abs().max()
    result = torch.empty((), device="cuda", dtype=torch.float32)
    K.abs_max(result, x)
    assert torch.isfinite(result)
    # abs_max just selects an existing value — no arithmetic error possible.
    assert result.cpu() == pytest.approx(ref.float().cpu(), rel=0, abs=0)


# ============================================================
# global_norm_squared
# Involves floating-point accumulation so tolerances are needed.
# ============================================================

@pytest.mark.parametrize("shape", [(128, 24524), (256, 1024), (1, 4096)])
@pytest.mark.parametrize("dtype", DTYPES)
def test_global_norm_squared(shape, dtype):
    x = torch.randn(*shape, device="cuda", dtype=dtype)
    out = torch.zeros((256,), device="cuda", dtype=torch.float32)
    K.global_norm_squared(out, x)
    ref = (x.float() ** 2).sum()
    rtol = 1e-5 if dtype == torch.float32 else 1e-2
    assert out.sum().cpu() == pytest.approx(ref.cpu(), rel=rtol)


# ============================================================
# rmsnorm
# ============================================================

def _rmsnorm_reference(inp, weight, eps):
    var = (inp.float() ** 2).mean(dim=-1, keepdim=True)
    rms = torch.sqrt(var + eps)
    r_rms = 1.0 / rms
    out = inp.float() * r_rms * weight.float()
    return out.to(inp.dtype), r_rms.squeeze(-1)


def _rmsnorm_backward_reference(dout, inp, weight, rstd):
    # dout, inp: (B, T, C), weight: (C,), rstd: (B, T)
    B, T, C = inp.shape
    dout_f = dout.float()
    inp_f = inp.float()
    w_f = weight.float()
    r = rstd.unsqueeze(-1)  # (B, T, 1)

    # dweight: sum over B, T
    dweight = (dout_f * inp_f * r).sum(dim=(0, 1))

    # dinp
    normed = inp_f * r                        # (B, T, C)
    dy_w = dout_f * w_f                       # (B, T, C)
    dot = (dy_w * normed).sum(dim=-1, keepdim=True)  # (B, T, 1)
    dinp = r * (dy_w - normed * dot / C)

    return dinp.to(inp.dtype), dweight.to(weight.dtype)


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


@pytest.mark.parametrize("B,T,C", [(2, 3, 16), (1, 4, 64), (4, 8, 256)])
@pytest.mark.parametrize("dtype", DTYPES)
def test_rmsnorm_backward(B, T, C, dtype):
    torch.manual_seed(0)
    inp = torch.randn((B, T, C), device="cuda", dtype=dtype)
    weight = torch.randn((C,), device="cuda", dtype=dtype)
    dout = torch.randn((B, T, C), device="cuda", dtype=dtype)
    eps = 1e-6

    # Forward to get rstd
    _, rstd = _rmsnorm_reference(inp, weight, eps)
    rstd = rstd.to(torch.float32).cuda()

    dinp = torch.empty_like(inp)
    dweight = torch.zeros((C,), device="cuda", dtype=dtype)
    scratch = torch.zeros(K.get_rmsnorm_backward_scratch_size(C), device="cuda", dtype=torch.float32)
    dresidual = torch.zeros_like(inp)

    K.rmsnorm_backward(dinp, dweight, scratch, dresidual, dout, inp, weight, rstd, None)

    ref_dinp, ref_dweight = _rmsnorm_backward_reference(dout, inp, weight, rstd)
    rtol, atol = TOL[dtype]
    assert dinp.float().cpu() == pytest.approx(ref_dinp.float().cpu(), rel=rtol, abs=atol)
    assert dweight.float().cpu() == pytest.approx(ref_dweight.float().cpu(), rel=rtol, abs=atol)


# ============================================================
# swiglu_forward
# ============================================================

def _swiglu_reference(x: torch.Tensor) -> torch.Tensor:
    a, b = torch.tensor_split(x.float(), 2, dim=-1)
    return (torch.nn.functional.silu(b) * a).to(x.dtype)


def _swiglu_backward_reference(dout, inp):
    inp_f = inp.detach().float().requires_grad_(True)
    out = _swiglu_reference(inp_f)
    out.backward(dout.float())
    return inp_f.grad.to(inp.dtype)


@pytest.mark.parametrize("B,T,C", [(1, 16, 128), (8, 8, 16), (5, 32, 32), (32, 32, 1024)])
@pytest.mark.parametrize("dtype", DTYPES)
def test_swiglu_forward_matches_reference(B, T, C, dtype):
    torch.manual_seed(123)
    inp = torch.randn((B, T, 2 * C), device="cuda", dtype=dtype)
    out = torch.empty((B, T, C), device="cuda", dtype=dtype)

    K.swiglu_forward(out, inp, None)

    ref = _swiglu_reference(inp)
    rtol, atol = TOL[dtype]
    assert out.float().cpu() == pytest.approx(ref.float().cpu(), rel=rtol, abs=atol)


@pytest.mark.parametrize("B,T,C", [(2, 8, 64), (4, 16, 128)])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_swiglu_forward_quant_fp8(B, T, C, dtype):
    """swiglu_forward_quant writes fp8 output, using the supplied abs-max to scale values"""
    torch.manual_seed(42)
    inp = torch.randn((B, T, 2 * C), device="cuda", dtype=dtype)
    out = torch.empty((B, T, C), device="cuda", dtype=torch.float8_e4m3fn)
    scale = torch.empty((), device="cuda", dtype=torch.float32)

    ref = _swiglu_reference(inp.float())
    abs_max = ref.abs().max().float()
    K.swiglu_forward_quant(out, scale, inp, abs_max)

    dequant = out.float() * scale
    # fp8 has limited precision — use loose tolerance
    assert dequant.cpu() == pytest.approx(ref.cpu(), rel=0.125, abs=1e-2)

    expected_scale = abs_max / 448.0
    assert scale.cpu() == pytest.approx(expected_scale.cpu(), rel=1e-3)

@pytest.mark.parametrize("B,T,C", [(1, 8, 256), (4, 16, 64)])
@pytest.mark.parametrize("dtype", DTYPES)
def test_swiglu_backward(B, T, C, dtype):
    torch.manual_seed(0)
    inp = torch.randn((B, T, 2 * C), device="cuda", dtype=dtype)
    dout = torch.randn((B, T, C), device="cuda", dtype=dtype)
    dinp = torch.empty_like(inp)

    K.swiglu_backward(dinp, dout, inp, None)

    ref = _swiglu_backward_reference(dout, inp)
    rtol, atol = TOL[dtype]

    assert dinp.float().cpu() == pytest.approx(ref.float().cpu(), rel=rtol, abs=atol)


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

    # Determinism: same seed + subsequence must give bit-identical output.
    y = torch.empty_like(x)
    K.fill_normal(y, float(mean), float(std), seed, 0)
    assert x.cpu() == pytest.approx(y.cpu(), rel=0, abs=0)


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
    assert residual.float().cpu() == pytest.approx(res_ref.float().cpu(), rel=rtol, abs=atol)
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

    # Determinism: same seed must give bit-identical output.
    assert out1.float().cpu() == pytest.approx(out2.float().cpu(), rel=0, abs=0)

    # Accuracy vs fp32 reference; stochastic rounding may introduce ~1 ulp at target dtype.
    ref = scale.item() * (a.float() + b.float())
    assert out1.float().cpu() == pytest.approx(ref.cpu(), rel=1e-2, abs=5e-3)


# ============================================================
# quantize_with_abs_max
# ============================================================

@pytest.mark.parametrize("N", [1024, 8192, 65536])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_quantize_with_abs_max_bf16(N, dtype):
    device = "cuda"
    x = torch.randn((N,), device=device, dtype=dtype)
    abs_max_val = torch.tensor([x.float().abs().max().item()], device=device, dtype=torch.float32)
    out = torch.empty((N,), device=device, dtype=torch.bfloat16)
    scale = torch.empty((), device=device, dtype=torch.float32)
    K.quantize_with_abs_max(out, scale, x, abs_max_val)
    assert scale.item() == pytest.approx(1.0, rel=0, abs=0)
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


# ============================================================
# fused_classifier
# ============================================================


@pytest.mark.parametrize("write_dlogits", [False, True])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("B,T,V", [(2, 4, 16), (1, 8, 32), (4, 2, 64)])
def test_fused_classifier_losses(B, T, V, write_dlogits, dtype):
    """Per-token losses, lse, and (optionally) dlogits must match reference with no z-regularisation."""
    torch.manual_seed(0)
    logits = torch.randn((B, T, V), device="cuda", dtype=dtype)
    targets = torch.randint(0, V, (B, T), device="cuda", dtype=torch.int32)

    losses = torch.zeros((B, T), device="cuda", dtype=torch.float32)
    lse = torch.empty((B, T), device="cuda", dtype=torch.float32)
    dloss = 1.0
    logits_copy = logits.clone()  # kernel mutates logits in place

    K.fused_classifier(logits_copy, losses, lse, dloss, targets, 0.0, write_dlogits)

    ref_lse = torch.logsumexp(logits.float(), dim=-1)
    assert lse.cpu() == pytest.approx(ref_lse.cpu(), rel=1e-4, abs=1e-5)

    ref_losses = F.cross_entropy(logits.reshape(B * T, V).float(), targets.reshape(B * T).long(), reduction="none").reshape(B, T)
    assert losses.cpu() == pytest.approx(ref_losses.cpu(), rel=1e-4, abs=1e-5)

    if write_dlogits:
        probs = torch.softmax(logits.float(), dim=-1)
        onehot = torch.zeros_like(probs)
        onehot.scatter_(-1, targets.long().unsqueeze(-1), 1.0)
        ref_dlogits = probs - onehot  # dloss=1 everywhere
        assert logits_copy.float().cpu() == pytest.approx(ref_dlogits.cpu(), rel=1e-2, abs=1e-3)



# ============================================================
# adamw_update
# ============================================================

def _adamw_reference(params, grads, m, v, lr, beta1, beta2, t, eps, wd):
    m_new = beta1 * m.float() + (1 - beta1) * grads.float()
    v_new = beta2 * v.float() + (1 - beta2) * grads.float() ** 2
    m_hat = m_new / (1 - beta1 ** t)
    v_hat = v_new / (1 - beta2 ** t)
    params_new = params.float() * (1 - lr * wd) - lr * m_hat / (v_hat.sqrt() + eps)
    return params_new, m_new, v_new


@pytest.mark.parametrize("N", [1024, 8192, 65536])
@pytest.mark.parametrize("p_dtype, g_dtype, m_dtype, v_dtype", [
    (torch.float32, torch.float32, torch.float32, torch.float32),
    (torch.bfloat16, torch.bfloat16, torch.float32, torch.float32),
    (torch.bfloat16, torch.bfloat16, torch.bfloat16, torch.float32),
    (torch.bfloat16, torch.bfloat16, torch.bfloat16, torch.bfloat16),
])
def test_adamw_update_matches_reference(N, p_dtype, g_dtype, m_dtype, v_dtype):
    torch.manual_seed(0)
    params = torch.randn((N,), device="cuda", dtype=p_dtype)
    grads = torch.randn((N,), device="cuda", dtype=g_dtype)
    m = torch.randn((N,), device="cuda", dtype=m_dtype) * 0.1
    v = torch.rand((N,), device="cuda", dtype=v_dtype) * 0.01 + 1e-8
    g_scale = torch.rand((), device="cuda", dtype=torch.float32) * 0.5 + 0.5

    lr, beta1, beta2, t, eps, wd = 1e-3, 0.9, 0.999, 1, 1e-8, 0.1
    ref_params, ref_m, ref_v = _adamw_reference(
        params.clone().float(), grads.float() * g_scale, m.clone().float(), v.clone().float(), lr, beta1, beta2, t, eps, wd
    )

    K.adamw_update(params, grads, m, v, lr, beta1, beta2, t, eps, wd, g_scale)

    rtol, atol = TOL[p_dtype]
    assert params.float().cpu() == pytest.approx(ref_params.cpu(), rel=rtol, abs=atol)
    rtol, atol = TOL[m_dtype]
    assert m.float().cpu() == pytest.approx(ref_m.cpu(), rel=rtol, abs=atol)
    rtol, atol = TOL[v_dtype]
    assert v.float().cpu() == pytest.approx(ref_v.cpu(), rel=rtol, abs=atol)


# ============================================================
# vector_reduce_sr
# ============================================================
@pytest.mark.parametrize("n_shards,nelem", [(2, 4096), (4, 8192), (8, 16384)])
@pytest.mark.parametrize("dtype", DTYPES)
def test_vector_reduce_sr_sum_matches_reference(n_shards, nelem, dtype):
    """With scale=1, the reduction must equal the sum across shards."""
    torch.manual_seed(0)
    src = torch.randn((n_shards * nelem,), device="cuda", dtype=dtype)
    dest = torch.empty((nelem,), device="cuda", dtype=dtype)
    scale = torch.tensor(1.0, dtype=torch.float32)

    K.vector_reduce_sr(dest, src, scale, n_shards, seed=0)

    ref = src.view(n_shards, nelem).float().sum(dim=0)
    rtol, atol = TOL[dtype]
    assert dest.float().cpu() == pytest.approx(ref.cpu(), rel=rtol, abs=atol)



@pytest.mark.parametrize("n_shards,nelem", [(2, 4096), (4, 8192)])
@pytest.mark.parametrize("dtype", DTYPES)
def test_vector_reduce_sr_determinism(n_shards, nelem, dtype):
    src = torch.randn((n_shards * nelem,), device="cuda", dtype=dtype)
    dest1 = torch.empty((nelem,), device="cuda", dtype=dtype)
    dest2 = torch.empty((nelem,), device="cuda", dtype=dtype)
    scale = torch.tensor(0.5, dtype=torch.float32)

    K.vector_reduce_sr(dest1, src, scale, n_shards, seed=99)
    K.vector_reduce_sr(dest2, src, scale, n_shards, seed=99)

    assert dest1.float().cpu() == pytest.approx(dest2.float().cpu(), rel=0, abs=0)


@pytest.mark.parametrize("n_shards,nelem", [(2, 4096), (4, 8192)])
@pytest.mark.parametrize("dtype", DTYPES)
def test_vector_reduce_sr_scale_zero(n_shards, nelem, dtype):
    src = torch.randn((n_shards * nelem,), device="cuda", dtype=dtype)
    dest = torch.empty((nelem,), device="cuda", dtype=dtype)
    scale = torch.tensor(0.0, dtype=torch.float32)

    K.vector_reduce_sr(dest, src, scale, n_shards, seed=0)

    assert dest.float().cpu() == pytest.approx(torch.zeros(nelem).cpu(), rel=0, abs=0)


@pytest.mark.parametrize("n_shards,nelem", [(2, 4096), (4, 8192)])
@pytest.mark.parametrize("dtype", DTYPES)
def test_vector_reduce_sr_skip(n_shards, nelem, dtype):
    """Skipped shard should not contribute to the result."""
    torch.manual_seed(42)
    src = torch.randn((n_shards * nelem,), device="cuda", dtype=dtype)
    dest = torch.empty((nelem,), device="cuda", dtype=dtype)
    scale = torch.tensor(1.0, dtype=torch.float32)

    skip = 1
    K.vector_reduce_sr(dest, src, scale, n_shards, skip=skip, seed=0)

    shards = src.view(n_shards, nelem).float()
    ref = sum(shards[k] for k in range(n_shards) if k != skip)
    rtol, atol = TOL[dtype]
    assert dest.float().cpu() == pytest.approx(ref.cpu(), rel=rtol, abs=atol)


@pytest.mark.parametrize("n_shards,nelem", [(2, 4096), (4, 8192)])
@pytest.mark.parametrize("dtype", DTYPES)
def test_vector_reduce_sr_accumulate(n_shards, nelem, dtype):
    """With accumulate=True, dest's initial values should be included in the sum."""
    torch.manual_seed(7)
    src = torch.randn((n_shards * nelem,), device="cuda", dtype=dtype)
    dest = torch.randn((nelem,), device="cuda", dtype=dtype)
    dest_initial = dest.clone()
    scale = torch.tensor(1.0, dtype=torch.float32)

    K.vector_reduce_sr(dest, src, scale, n_shards, accumulate=True, seed=0)

    ref = src.view(n_shards, nelem).float().sum(dim=0) + dest_initial.float()
    rtol, atol = TOL[dtype]
    assert dest.float().cpu() == pytest.approx(ref.cpu(), rel=rtol, abs=atol)


@pytest.mark.parametrize("dtype", DTYPES)
def test_vector_reduce_sr_stochastic_rounding_unbiased(dtype: torch.dtype, nelem=4096):
    """SR rounding should be unbiased: mean of many rounded values ≈ true mean."""
    n_shards = 2
    # Use values that are exactly halfway between representable values to maximise rounding effect
    src = torch.ones((n_shards * nelem,), device="cuda", dtype=dtype) * 0.5
    results = []
    for seed in range(20):
        dest = torch.empty((nelem,), device="cuda", dtype=dtype)
        scale = torch.tensor(1.0, dtype=torch.float32)
        K.vector_reduce_sr(dest, src, scale, n_shards, seed=seed)
        results.append(dest.float().cpu())

    mean_result = torch.stack(results).mean(dim=0)
    # True answer is 1.0 (sum of two 0.5 shards); mean over seeds should be close
    assert mean_result == pytest.approx(torch.ones(nelem).cpu(), rel=0.01, abs=0.01)


# ============================================================
# backward_bias
# ============================================================
@pytest.mark.parametrize("B,T,OC", [(2, 5, 64), (1, 3, 256), (4, 8, 128), (8, 16, 512)])
@pytest.mark.parametrize("dtype", DTYPES)
def test_backward_bias(B, T, OC, dtype):
    device = "cuda"
    torch.manual_seed(0)

    dout = torch.randn((B, T, OC), device=device, dtype=dtype)
    dbias = torch.zeros((OC,), device=device, dtype=dtype)
    dbias_buffer = torch.empty(K.get_bias_backward_scratch_size(dtype, OC) // 4, device=device, dtype=torch.float32)

    K.backward_bias(dbias, dout, None, None, dbias_buffer)

    ref = dout.float().sum(dim=(0, 1)).to(dtype)

    rtol, atol = TOL[dtype]
    assert dbias.float().cpu() == pytest.approx(ref.float().cpu(), rel=rtol, abs=atol)


@pytest.mark.parametrize("B,T,OC", [(2, 5, 64), (1, 3, 256), (4, 8, 128), (8, 16, 512)])
@pytest.mark.parametrize("dtype", DTYPES)
def test_backward_bias_with_scale(B, T, OC, dtype):
    device = "cuda"
    torch.manual_seed(0)

    dout = torch.randn((B, T, OC), device=device, dtype=dtype)
    dbias = torch.zeros((OC,), device=device, dtype=dtype)
    dbias_buffer = torch.zeros(K.get_bias_backward_scratch_size(dtype, OC) // 4, device=device, dtype=torch.float32)
    scale_a = torch.tensor(0.25, device=device, dtype=torch.float32)
    scale_b = torch.tensor(2.0, device=device, dtype=torch.float32)

    scale = scale_a.item() * scale_b.item()
    ref = (dout.float().sum(dim=(0, 1)) * scale).to(dtype)
    K.backward_bias(dbias, dout, scale_a, scale_b, dbias_buffer)
    rtol, atol = TOL[dtype]
    assert dbias.float().cpu() == pytest.approx(ref.float().cpu(), rel=rtol, abs=atol)


@pytest.mark.parametrize("B,T,OC", [(2, 5, 64), (1, 3, 256), (4, 8, 128), (8, 16, 512)])
@pytest.mark.parametrize("dtype", DTYPES)
def test_backward_bias_accumulates(B, T, OC, dtype):
    """Verify that backward_bias adds into dbias rather than overwriting it."""
    device = "cuda"
    torch.manual_seed(0)

    dout = torch.randn((B, T, OC), device=device, dtype=dtype)
    initial = torch.randn((OC,), device=device, dtype=dtype)
    dbias = initial.clone()
    dbias_buffer = torch.zeros(K.get_bias_backward_scratch_size(dtype, OC) // 4, device=device, dtype=torch.float32)

    K.backward_bias(dbias, dout, None, None, dbias_buffer)

    ref = (initial.float() + dout.float().sum(dim=(0, 1))).to(dtype)

    rtol, atol = TOL[dtype]
    assert dbias.float().cpu() == pytest.approx(ref.float().cpu(), rel=rtol, abs=atol)

# ============================================================
# matmul
# ============================================================

@pytest.fixture(scope="module")
def cublas_handle():
    handle = K.create_cublas_handle()
    try:
        yield handle
    finally:
        K.destroy_cublas_handle(handle)

@pytest.fixture(scope="module")
def workspace():
    # 32MB workspace, typical for cublasLt
    return torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)


# EMMTranspose: TT=0, TN=1, NT=2, NN=3
MODES = {
    "NN": 3,  # C = A @ B
    "NT": 2,  # C = A @ B.T
    "TN": 1,  # C = A.T @ B
    "TT": 0,  # C = A.T @ B.T
}


def ref_matmul(a: torch.Tensor, b: torch.Tensor, bias, mode_str: str, accumulate: bool, c_ref: torch.Tensor) -> torch.Tensor:
    a_f = a.float()
    b_f = b.float()
    if mode_str == "NN":
        out = a_f @ b_f
    elif mode_str == "NT":
        out = a_f @ b_f.transpose(-1, -2)
    elif mode_str == "TN":
        out = a_f.transpose(-1, -2) @ b_f
    elif mode_str == "TT":
        out = a_f.transpose(-1, -2) @ b_f.transpose(-1, -2)
    if bias is not None:
        out = out + bias.float()
    if accumulate:
        out = out + c_ref.float()
    return out


@pytest.mark.parametrize("M,K_dim,N", [
    (1, 64, 64),
    (10, 64, 128),
    (3, 256, 256),
    (32, 128, 64),
])
@pytest.mark.parametrize("mode_str", list(MODES.keys()))
@pytest.mark.parametrize("dtype", DTYPES)
def test_matmul_basic(M, K_dim, N, mode_str, dtype, cublas_handle, workspace):
    torch.manual_seed(0)
    device = "cuda"
    mode = MODES[mode_str]
    rtol, atol = TOL[dtype]

    if mode_str in ("NN", "NT"):
        a = torch.randn(M, K_dim, device=device, dtype=dtype)
    else:  # TN, TT: A is [K, M]
        a = torch.randn(K_dim, M, device=device, dtype=dtype)

    if mode_str in ("NN", "TN"):
        b = torch.randn(K_dim, N, device=device, dtype=dtype)
    else:  # NT, TT: B is [N, K]
        b = torch.randn(N, K_dim, device=device, dtype=dtype)

    c = torch.empty(M, N, device=device, dtype=dtype)
    K.matmul(c, a, b, None, None, None, cublas_handle, workspace, mode, False)

    ref = ref_matmul(a, b, None, mode_str, False, c)
    assert c.float().cpu() == pytest.approx(ref.cpu(), rel=rtol, abs=atol)


@pytest.mark.parametrize("M,K_dim,N", [(32, 64, 128), (64, 128, 64)])
@pytest.mark.parametrize("dtype", DTYPES)
def test_matmul_bias(M, K_dim, N, dtype, cublas_handle, workspace):
    torch.manual_seed(0)
    device = "cuda"
    rtol, atol = TOL[dtype]

    a = torch.randn(M, K_dim, device=device, dtype=dtype)
    b = torch.randn(K_dim, N, device=device, dtype=dtype)
    bias = torch.randn(N, device=device, dtype=dtype)
    c = torch.zeros(M, N, device=device, dtype=dtype)

    K.matmul(c, a, b, bias, None, None, cublas_handle, workspace, MODES["NN"], False)

    ref = ref_matmul(a, b, bias, "NN", False, c)
    assert c.float().cpu() == pytest.approx(ref.cpu(), rel=rtol, abs=atol)


@pytest.mark.parametrize("M,K_dim,N", [(32, 64, 128), (64, 128, 64)])
@pytest.mark.parametrize("mode_str", list(MODES.keys()))
@pytest.mark.parametrize("dtype", DTYPES)
def test_matmul_accumulate(M, K_dim, N, mode_str, dtype, cublas_handle, workspace):
    torch.manual_seed(0)
    device = "cuda"
    mode = MODES[mode_str]
    rtol, atol = TOL[dtype]

    if mode_str in ("NN", "NT"):
        a = torch.randn(M, K_dim, device=device, dtype=dtype)
    else:
        a = torch.randn(K_dim, M, device=device, dtype=dtype)

    if mode_str in ("NN", "TN"):
        b = torch.randn(K_dim, N, device=device, dtype=dtype)
    else:
        b = torch.randn(N, K_dim, device=device, dtype=dtype)

    c = torch.randn(M, N, device=device, dtype=dtype)
    c_ref = c.clone()

    K.matmul(c, a, b, None, None, None, cublas_handle, workspace, mode, True)

    ref = ref_matmul(a, b, None, mode_str, True, c_ref)
    assert c.float().cpu() == pytest.approx(ref.cpu(), rel=rtol, abs=atol)


@pytest.mark.parametrize("M,K_dim,N", [(32, 64, 128)])
@pytest.mark.parametrize("dtype", DTYPES)
def test_matmul_bias_and_accumulate(M, K_dim, N, dtype, cublas_handle, workspace):
    torch.manual_seed(0)
    device = "cuda"
    rtol, atol = TOL[dtype]
    if dtype == torch.bfloat16:
        rtol = 1.5e-2
        atol = 6e-2

    a = torch.randn(M, K_dim, device=device, dtype=dtype)
    b = torch.randn(K_dim, N, device=device, dtype=dtype)
    bias = torch.randn(N, device=device, dtype=dtype)
    c = torch.randn(M, N, device=device, dtype=dtype)
    c_ref = c.clone()

    K.matmul(c, a, b, bias, None, None, cublas_handle, workspace, MODES["NN"], True)

    ref = ref_matmul(a, b, bias, "NN", True, c_ref)
    assert c.float().cpu() == pytest.approx(ref.cpu(), rel=rtol, abs=atol)

# ============================================================
# encoder_forward / encoder_backward
# ============================================================

def _encoder_forward_reference(inp: torch.Tensor, wte: torch.Tensor, wpe: torch.Tensor | None) -> torch.Tensor:
    """Token embedding + optional positional embedding lookup."""
    out = wte[inp]  # (B, T, C)
    if wpe is not None:
        T = inp.shape[1]
        out = out + wpe[:T]
    return out


@pytest.mark.parametrize("B,T,V,C", [(2, 8, 64, 32), (1, 16, 128, 64), (4, 4, 32, 16)])
@pytest.mark.parametrize("dtype", DTYPES)
def test_encoder_forward_with_wpe(B, T, V, C, dtype):
    """Token + position embedding lookup must match reference, with and without wpe."""
    torch.manual_seed(0)
    device = "cuda"

    inp = torch.randint(0, V, (B, T), device=device, dtype=torch.int32)
    wte = torch.randn((V, C), device=device, dtype=dtype)
    wpe = torch.randn((T, C), device=device, dtype=dtype)
    out = torch.empty((B, T, C), device=device, dtype=dtype)

    K.encoder_forward(out, inp, wte, wpe)

    ref = _encoder_forward_reference(inp, wte, wpe)
    # Embedding lookup is a pure gather — no arithmetic error possible.
    assert out.float().cpu() == pytest.approx(ref.float().cpu(), rel=0, abs=0)


@pytest.mark.parametrize("B,T,V,C", [(2, 8, 64, 32), (3, 12, 50, 48)])
@pytest.mark.parametrize("dtype", DTYPES)
def test_encoder_forward_no_wpe(B, T, V, C, dtype):
    """With wpe=None the output must equal wte[inp] exactly — no position bias added."""
    torch.manual_seed(1)
    device = "cuda"

    inp = torch.randint(0, V, (B, T), device=device, dtype=torch.int32)
    wte = torch.randn((V, C), device=device, dtype=dtype)
    out = torch.empty((B, T, C), device=device, dtype=dtype)

    K.encoder_forward(out, inp, wte, None)

    ref = wte[inp]  # shape (B, T, C)
    assert out.float().cpu() == pytest.approx(ref.float().cpu(), rel=0, abs=0)


def _make_encoder_backward_buffers(V: int, C: int, B: int, T: int, device: str, dtype: torch.dtype):
    """Allocate the auxiliary buffers required by encoder_backward."""
    dwte = torch.zeros((V, C), device=device, dtype=dtype)
    cg_max = int(math.ceil(C / 32))
    scratch = torch.zeros((B, T, 5*cg_max), device=device, dtype=torch.int32)
    workload_indices = torch.zeros((B, T, cg_max), device="cpu", dtype=torch.int32)
    bucket_info = torch.zeros((B, T, 4*cg_max), device="cpu", dtype=torch.int32)
    return dwte, scratch, workload_indices, bucket_info


@pytest.mark.parametrize("B,T,V,C", [(2, 8, 16, 32), (1, 4, 8, 16), (3, 6, 32, 64)])
@pytest.mark.parametrize("dtype", DTYPES)
def test_encoder_backward_gradient_accumulation(B, T, V, C, dtype):
    """dwte[token] must accumulate dout contributions from every position that used that token."""
    torch.manual_seed(0)
    device = "cuda"
    event = torch.cuda.Event()
    event.record()  # pytorch events are lazy.

    # Repeat token index 0 at every position to stress accumulation.
    inp = torch.zeros((B, T), device=device, dtype=torch.int32)
    inp_cpu = inp.cpu()
    dout = torch.randn((B, T, C), device=device, dtype=dtype)
    dwte, scratch, workload_indices, bucket_info = _make_encoder_backward_buffers(V, C, B, T, device, dtype)

    K.encoder_backward(dwte, scratch, workload_indices, bucket_info, dout, inp, inp_cpu, sync_event=event.cuda_event, seed=0)

    # Reference: scatter-add over all (b, t) pairs.
    ref_dwte = torch.zeros((V, C), dtype=torch.float32)
    for b in range(B):
        for t in range(T):
            ref_dwte[inp_cpu[b, t]] += dout[b, t].float().cpu()

    rtol, atol = TOL[dtype]
    assert dwte[0].float().cpu() == pytest.approx(ref_dwte[0].cpu(), rel=rtol, abs=atol)
    # Tokens that were never used must stay zero.
    assert dwte[1:].abs().max().item() == 0.0


@pytest.mark.parametrize("B,T,V,C", [(2, 8, 16, 32), (4, 4, 32, 64)])
@pytest.mark.parametrize("dtype", DTYPES)
def test_encoder_backward_seed_determinism(B, T, V, C, dtype):
    """Same seed must produce bit-identical dwte results."""
    torch.manual_seed(42)
    device = "cuda"
    event = torch.cuda.Event()
    event.record()  # pytorch events are lazy.

    inp = torch.randint(0, V, (B, T), device=device, dtype=torch.int32)
    inp_cpu = inp.cpu()
    dout = torch.randn((B, T, C), device=device, dtype=dtype)

    seed = 0xDEADBEEF

    dwte1, scratch1, wi1, bi1 = _make_encoder_backward_buffers(V, C, B, T, device, dtype)
    K.encoder_backward(dwte1, scratch1, wi1, bi1, dout, inp, inp_cpu, seed=seed, sync_event=event.cuda_event)

    dwte2, scratch2, wi2, bi2 = _make_encoder_backward_buffers(V, C, B, T, device, dtype)
    K.encoder_backward(dwte2, scratch2, wi2, bi2, dout, inp, inp_cpu, seed=seed, sync_event=event.cuda_event)

    # Bit-exact: same seed must yield identical output.
    assert dwte1.float().cpu() == pytest.approx(dwte2.float().cpu(), rel=0, abs=0)


# ============================================================
# quantize_and_transpose_with_abs_max
# ============================================================

@pytest.mark.parametrize("rows,cols", [(32, 32), (128, 256), (512, 1024)])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_quantize_and_transpose_with_abs_max_bf16(rows, cols, dtype):
    device = "cuda"
    x = torch.randn((rows, cols), device=device, dtype=dtype)
    abs_max_val = torch.tensor([x.float().abs().max().item()], device=device, dtype=torch.float32)
    out = torch.empty((cols, rows), device=device, dtype=torch.bfloat16)
    scale = torch.empty((), device=device, dtype=torch.float32)
    K.quantize_and_transpose_with_abs_max(out, scale, x, abs_max_val)

    # no assert for scale, as scale is unused for bf16

    expected = x.bfloat16().T.contiguous()
    assert out.float().cpu() == pytest.approx(expected.float().cpu(), rel=0.01)


@pytest.mark.parametrize("rows,cols", [(32, 32), (128, 256), (512, 1024)])
@pytest.mark.parametrize("dtype", DTYPES)
def test_quantize_and_transpose_with_abs_max_fp8(rows, cols, dtype):
    device = "cuda"
    x = torch.randn((rows, cols), device=device, dtype=dtype)
    abs_max_val = torch.tensor([x.float().abs().max().item()], device=device, dtype=torch.float32)
    out = torch.empty((cols, rows), device=device, dtype=torch.float8_e4m3fn)
    scale = torch.empty((), device=device, dtype=torch.float32)
    K.quantize_and_transpose_with_abs_max(out, scale, x, abs_max_val)

    assert scale.item() == pytest.approx(abs_max_val.item() / 448.0)

    # Dequantize and verify values match input (with fp8 tolerance)
    dequant = out.float() * scale

    # Verify transpose: reshape input as [rows, cols], transpose to [cols, rows]
    expected = x.float().T.contiguous()
    assert dequant.cpu() == pytest.approx(expected.cpu(), rel=0.1, abs=1e-4)
