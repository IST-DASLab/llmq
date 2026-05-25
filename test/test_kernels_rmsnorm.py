import pytest
import torch
from pyllmq import kernels as K
from conftest import DTYPES, TOL

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


@pytest.mark.parametrize("B,T,C", [(2, 3, 16), (1, 4, 64), (4, 8, 256)])
@pytest.mark.parametrize("dtype", DTYPES)
def test_rmsnorm_backward_dresidual_accumulate(B, T, C, dtype):
    torch.manual_seed(1)
    inp = torch.randn((B, T, C), device="cuda", dtype=dtype)
    weight = torch.randn((C,), device="cuda", dtype=dtype)
    dout = torch.randn((B, T, C), device="cuda", dtype=dtype)
    eps = 1e-6

    _, rstd = _rmsnorm_reference(inp, weight, eps)
    rstd = rstd.to(torch.float32).cuda()

    dinp = torch.empty_like(inp)
    dweight = torch.zeros((C,), device="cuda", dtype=dtype)
    scratch = torch.zeros(K.get_rmsnorm_backward_scratch_size(C), device="cuda", dtype=torch.float32)

    # Pre-fill dresidual with a non-zero sentinel so accumulation is detectable.
    dresidual = torch.randn_like(inp)
    dresidual_initial = dresidual.clone()
    ref_dinp, _ = _rmsnorm_backward_reference(dout, inp, weight, rstd)
    expected_dinp = dresidual_initial.float() + ref_dinp.float()

    K.rmsnorm_backward(dinp, dweight, scratch, dresidual, dout, inp, weight, rstd, None)

    rtol, atol = TOL[dtype]
    if dtype == torch.bfloat16:
        # TODO check why we need so much tolerance
        atol = 8e-3
        rtol = 0.015
    assert dinp.float().cpu() == pytest.approx(
        expected_dinp.cpu(), rel=rtol, abs=atol
    )


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
