import pytest
import torch
from pyllmq import kernels as K
from conftest import DTYPES, TOL
from test_kernels_rope import _make_rope_freqs, _rope_python


# ============================================================
# qk_norm_forward
# ============================================================

def _qk_norm_reference(inp, q_wgt, k_wgt, Nq, Nkv, eps):
    B, T, _ = inp.shape
    N = Nq + 2 * Nkv
    HeadDim = q_wgt.shape[0]
    x = inp.float().view(B, T, N, HeadDim)

    # Only norm Q and K heads
    Nqk = Nq + Nkv
    x_qk = x[:, :, :Nqk, :]                                    # (B, T, Nq+Nkv, HeadDim)

    var = (x_qk ** 2).mean(dim=-1, keepdim=True)
    s = torch.rsqrt(var + eps)                                  # (B, T, Nq+Nkv, 1)

    wgt = torch.cat([
        q_wgt.float().unsqueeze(0).expand(Nq, -1),
        k_wgt.float().unsqueeze(0).expand(Nkv, -1),
    ], dim=0).view(1, 1, Nqk, HeadDim)

    x_qk_normed = ((x_qk * s).to(inp.dtype) * wgt).to(inp.dtype)  # (B, T, Nq+Nkv, HeadDim)

    # V heads are passed through unchanged
    x_v = x[:, :, Nqk:, :].to(inp.dtype)                       # (B, T, Nkv, HeadDim)

    # r_rms: pad V slots with 1.0 (or zeros) since they aren't normed
    r_rms_qk = s.squeeze(-1).float()                            # (B, T, Nq+Nkv)
    r_rms_v  = torch.ones(B, T, Nkv, dtype=torch.float32, device=inp.device)
    r_rms    = torch.cat([r_rms_qk, r_rms_v], dim=-1)          # (B, T, N)

    out = torch.cat([x_qk_normed, x_v], dim=2).view(B, T, N * HeadDim)
    return out, r_rms


@pytest.mark.parametrize("B,T,Nq,Nkv,HeadDim", [
    (1, 4, 2, 1, 16),
    (2, 8, 4, 2, 64),
    (4, 16, 8, 4, 128),
])
@pytest.mark.parametrize("dtype", DTYPES)
def test_qk_norm_forward(B, T, Nq, Nkv, HeadDim, dtype):
    torch.manual_seed(0)
    device = "cuda"
    N = Nq + 2 * Nkv
    eps = 1e-6

    inp    = torch.randn((B, T, N * HeadDim), device=device, dtype=dtype)
    q_wgt  = torch.randn((HeadDim,), device=device, dtype=dtype)
    k_wgt  = torch.randn((HeadDim,), device=device, dtype=dtype)
    out    = torch.empty_like(inp)
    # NOTE: the kernel does not write rrms for value heads at all; init to one to match reference
    r_rms  = torch.ones((B, T, N), device=device, dtype=torch.float32)

    K.qk_norm_forward(out, r_rms, inp, q_wgt, k_wgt, eps, Nq, Nkv)
    ref_out, ref_r_std = _qk_norm_reference(inp, q_wgt, k_wgt, Nq, Nkv, eps)

    rtol, atol = TOL[dtype]
    assert r_rms.cpu().numpy() == pytest.approx(ref_r_std.cpu().numpy(), rel=rtol, abs=atol)
    assert out.float().cpu().numpy() == pytest.approx(ref_out.float().cpu().numpy(), rel=rtol, abs=atol)


# ---- In-place forward: out is inp ----
# Exercises the early-return path for V-heads (inp == out) and the normalization path for QK-heads.
@pytest.mark.parametrize("B,T,Nq,Nkv,HeadDim", [
    (1, 1, 4, 2, 64),
    (2, 4, 2, 1, 128),
])
@pytest.mark.parametrize("dtype", DTYPES)
def test_qk_norm_forward_inplace(B, T, Nq, Nkv, HeadDim, dtype):
    torch.manual_seed(0)
    device = "cuda"
    N = Nq + 2 * Nkv
    eps = 1e-6

    inp = torch.randn((B, T, N * HeadDim), device=device, dtype=dtype)
    q_wgt = torch.randn((HeadDim,), device=device, dtype=dtype)
    k_wgt = torch.randn((HeadDim,), device=device, dtype=dtype)
    v_start = Nq + Nkv
    v_end = N

    inp_copy = inp.clone()

    out = inp
    r_rms = torch.ones((B, T, N), device=device, dtype=torch.float32)

    K.qk_norm_forward(out, r_rms, inp, q_wgt, k_wgt, eps, Nq, Nkv)

    actual_v = out[:, :, v_start * HeadDim : v_end * HeadDim]
    orig_v   = inp_copy[:, :, v_start * HeadDim : v_end * HeadDim]
    assert torch.allclose(actual_v, orig_v, rtol=0, atol=0), "V-heads changed in in-place forward"

    ref_out, _ = _qk_norm_reference(inp_copy, q_wgt, k_wgt, Nq, Nkv, eps)
    rtol, atol = TOL[dtype]
    assert out.float().cpu().numpy() == pytest.approx(ref_out.float().cpu().numpy(), rel=rtol, abs=atol)


@pytest.mark.parametrize("B,T,Nq,Nkv,HeadDim", [
    (1, 4, 2, 1, 16),
    (2, 8, 4, 2, 64),
    (4, 16, 8, 4, 128),
])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("with_abs_max", [False, True])
def test_qk_norm_and_rope_forward(B, T, Nq, Nkv, HeadDim, dtype, with_abs_max):
    torch.manual_seed(0)
    device = "cuda"
    N = Nq + 2 * Nkv
    eps = 1e-6

    inp   = torch.randn((B, T, N * HeadDim), device=device, dtype=dtype)
    q_wgt = torch.randn((HeadDim,), device=device, dtype=dtype)
    k_wgt = torch.randn((HeadDim,), device=device, dtype=dtype)
    out   = torch.empty_like(inp)
    # NOTE: the kernel does not write rrms for value heads; init to 1 to match reference
    r_rms = torch.ones((B, T, N), device=device, dtype=torch.float32)

    freq_dtype = torch.float32 if dtype == torch.float32 else torch.float16
    freqs = _make_rope_freqs(T, HeadDim, 1_000_000.0, freq_dtype, device)

    abs_max = torch.zeros((1,), device=device, dtype=torch.float32) if with_abs_max else None

    K.qk_norm_and_rope_forward(out, r_rms, inp, q_wgt, k_wgt, freqs, abs_max, eps, Nq, Nkv)

    # reference: qk-norm first, then rope on the result
    ref_normed, ref_r_rms = _qk_norm_reference(inp, q_wgt, k_wgt, Nq, Nkv, eps)
    ref_out = _rope_python(ref_normed.view(B, T, N, HeadDim), freqs, Nq, Nkv).view(B, T, N * HeadDim)

    rtol, atol = TOL[dtype]
    assert r_rms.cpu().numpy() == pytest.approx(ref_r_rms.cpu().numpy(), rel=rtol, abs=atol)
    assert out.float().cpu().numpy() == pytest.approx(ref_out.float().cpu().numpy(), rel=rtol, abs=atol)

    if with_abs_max:
        # kernel takes abs-max over Q+K post-rope and V pass-through
        ref_abs_max = ref_out.float().abs().max().item()
        assert abs_max.item() == pytest.approx(ref_abs_max, rel=rtol, abs=atol)


# ============================================================
# qk_norm_backward
# ============================================================

@pytest.mark.parametrize("B,T,Nq,Nkv,HeadDim", [
    (1, 4, 2, 1, 16),
    (2, 8, 4, 2, 64),
    (4, 16, 8, 4, 64),
    (1, 1, 4, 2, 128),
])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("in_place", [True, False])
def test_qk_norm_backward(B, T, Nq, Nkv, HeadDim, dtype, in_place: bool):
    torch.manual_seed(0)
    device = "cuda"
    N = Nq + 2 * Nkv
    eps = 1e-6

    inp   = torch.randn((B, T, N * HeadDim), device=device, dtype=dtype)
    q_wgt = torch.randn((HeadDim,),          device=device, dtype=dtype)
    k_wgt = torch.randn((HeadDim,),          device=device, dtype=dtype)
    dout  = torch.randn((B, T, N * HeadDim), device=device, dtype=dtype)

    # Reference in fp32
    inp_f   = inp.float().detach().requires_grad_(True)
    q_wgt_f = q_wgt.float().detach().requires_grad_(True)
    k_wgt_f = k_wgt.float().detach().requires_grad_(True)
    ref_out_f, _ = _qk_norm_reference(inp_f, q_wgt_f, k_wgt_f, Nq, Nkv, eps)
    ref_out_f.backward(dout.float())
    ref_dinp   = inp_f.grad
    ref_dq_wgt = q_wgt_f.grad
    ref_dk_wgt = k_wgt_f.grad

    # Run forward to get the kernel's rstd.
    r_rms = torch.empty((B, T, N), device=device, dtype=torch.float32)
    r_rms.fill_(1.0)  # V-slot sentinel; kernel only writes QK slots
    K.qk_norm_forward(torch.empty_like(inp), r_rms, inp, q_wgt, k_wgt, eps, Nq, Nkv)

    scratch_size = K.get_qknorm_backward_scratch_size(Nq, Nkv, HeadDim, inp.dtype)
    scratch  = torch.zeros((scratch_size,), device=device, dtype=torch.uint8)
    dq_wgt_k = torch.zeros_like(q_wgt)
    dk_wgt_k = torch.zeros_like(k_wgt)

    if in_place:
        dinp_buf = dout.clone()
        K.qk_norm_backward(dinp_buf, dq_wgt_k, dk_wgt_k, scratch,
                           dinp_buf, inp, q_wgt, k_wgt, r_rms,
                           None, Nq, Nkv)
        dinp = dinp_buf
    else:
        dinp = torch.zeros_like(inp)
        K.qk_norm_backward(dinp, dq_wgt_k, dk_wgt_k, scratch,
                           dout, inp, q_wgt, k_wgt, r_rms,
                           None, Nq, Nkv)

    rtol, atol = TOL[dtype]

    assert dinp.float().cpu().numpy()     == pytest.approx(ref_dinp.cpu().numpy(),   rel=rtol, abs=atol)
    assert dq_wgt_k.float().cpu().numpy() == pytest.approx(ref_dq_wgt.cpu().numpy(), rel=rtol, abs=atol)
    assert dk_wgt_k.float().cpu().numpy() == pytest.approx(ref_dk_wgt.cpu().numpy(), rel=rtol, abs=atol)

@pytest.mark.parametrize("B,T,Nq,Nkv,HeadDim", [
    (1, 4, 2, 1, 16),
    (2, 8, 4, 2, 64),
    (4, 16, 8, 4, 64),
    (1, 1, 4, 2, 128),
])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("in_place", [True, False])
def test_qk_norm_and_rope_backward(B, T, Nq, Nkv, HeadDim, dtype, in_place: bool):
    torch.manual_seed(0)
    device = "cuda"
    N = Nq + 2 * Nkv
    eps = 1e-6

    inp   = torch.randn((B, T, N * HeadDim), device=device, dtype=dtype)
    q_wgt = torch.randn((HeadDim,),          device=device, dtype=dtype)
    k_wgt = torch.randn((HeadDim,),          device=device, dtype=dtype)
    dout  = torch.randn((B, T, N * HeadDim), device=device, dtype=dtype)

    freq_dtype = torch.float32 if dtype == torch.float32 else torch.float16
    freqs = _make_rope_freqs(T, HeadDim, 1_000_000.0, freq_dtype, device)

    # Reference in fp32: qk-norm then rope, backprop through both.
    inp_f   = inp.float().detach().requires_grad_(True)
    q_wgt_f = q_wgt.float().detach().requires_grad_(True)
    k_wgt_f = k_wgt.float().detach().requires_grad_(True)
    ref_normed_f, _ = _qk_norm_reference(inp_f, q_wgt_f, k_wgt_f, Nq, Nkv, eps)
    ref_out_f = _rope_python(
        ref_normed_f.view(B, T, N, HeadDim), freqs, Nq, Nkv
    ).view(B, T, N * HeadDim)
    ref_out_f.backward(dout.float())
    ref_dinp   = inp_f.grad
    ref_dq_wgt = q_wgt_f.grad
    ref_dk_wgt = k_wgt_f.grad

    # Run the fused forward to populate r_rms (kernel only writes QK slots).
    r_rms = torch.empty((B, T, N), device=device, dtype=torch.float32)
    r_rms.fill_(1.0)
    K.qk_norm_and_rope_forward(
        torch.empty_like(inp), r_rms, inp, q_wgt, k_wgt, freqs, None, eps, Nq, Nkv,
    )

    scratch_size = K.get_qknorm_and_rope_backward_scratch_size(Nq, Nkv, HeadDim, inp.dtype)
    scratch  = torch.zeros((scratch_size,), device=device, dtype=torch.uint8)
    dq_wgt_k = torch.zeros_like(q_wgt)
    dk_wgt_k = torch.zeros_like(k_wgt)

    v_slice = slice((Nq + Nkv) * HeadDim, N * HeadDim)

    if in_place:
        dinp_buf = dout.clone()
        v_before = dinp_buf[:, :, v_slice].clone()
        K.qk_norm_and_rope_backward(
            dinp_buf, dq_wgt_k, dk_wgt_k, scratch,
            dinp_buf, inp, q_wgt, k_wgt, r_rms, freqs,
            None, Nq, Nkv,
        )
        dinp = dinp_buf
        # In-place V pass-through: kernel must not write V slots
        # (mirrors qk_norm_backward_kernel's `if (dout_i != dinp_i)` guard).
        assert torch.equal(dinp[:, :, v_slice], v_before)
    else:
        dinp = torch.zeros_like(inp)
        K.qk_norm_and_rope_backward(
            dinp, dq_wgt_k, dk_wgt_k, scratch,
            dout, inp, q_wgt, k_wgt, r_rms, freqs,
            None, Nq, Nkv,
        )
        # Out-of-place V pass-through: dtype copy is bit-exact.
        assert torch.equal(dinp[:, :, v_slice], dout[:, :, v_slice])

    rtol, atol = TOL[dtype]
    assert dinp.float().cpu().numpy()     == pytest.approx(ref_dinp.cpu().numpy(),   rel=rtol, abs=atol)
    assert dq_wgt_k.float().cpu().numpy() == pytest.approx(ref_dq_wgt.cpu().numpy(), rel=rtol, abs=atol)
    assert dk_wgt_k.float().cpu().numpy() == pytest.approx(ref_dk_wgt.cpu().numpy(), rel=rtol, abs=atol)
