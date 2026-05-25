import pytest
import torch
from pyllmq import kernels as K
from conftest import DTYPES, TOL


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


# ============================================================
# matmul
# ============================================================

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
