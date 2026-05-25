import pytest
import torch
from pyllmq import kernels as K
from conftest import DTYPES, TOL


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
    assert dequant.cpu() == pytest.approx(x.float().cpu(), rel=0.1, abs=1e-3)

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
