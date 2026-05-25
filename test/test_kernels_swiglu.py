import pytest
import torch
from pyllmq import kernels as K
from conftest import DTYPES, TOL


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

# ============================================================
# swiglu_backward
# ============================================================

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
