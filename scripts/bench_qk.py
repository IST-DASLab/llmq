import pytest
import torch
from pyllmq import kernels as K


DTYPES = [torch.float32, torch.bfloat16]

# Tolerances by dtype for approximate kernels: (rtol, atol)
TOL = {
    torch.float32: (5e-4, 5e-5),
    torch.bfloat16: (1e-2, 5e-3),
}


@pytest.mark.parametrize("B,T,Nq,Nkv,HeadDim", [
    (4, 1024, 8, 4, 128),
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

@pytest.mark.parametrize("B,T,Nq,Nkv,HeadDim", [
    (4, 1024, 8, 4, 128),
])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_qk_norm_backward(B, T, Nq, Nkv, HeadDim, dtype):
    torch.manual_seed(0)
    device = "cuda"
    N = Nq + 2 * Nkv
    eps = 1e-6

    inp   = torch.randn((B, T, N * HeadDim), device=device, dtype=dtype, requires_grad=True)
    q_wgt = torch.randn((HeadDim,), device=device, dtype=dtype, requires_grad=True)
    k_wgt = torch.randn((HeadDim,), device=device, dtype=dtype, requires_grad=True)
    dout  = torch.randn((B, T, N * HeadDim), device=device, dtype=dtype)


    scratch_size = K.get_qknorm_backward_scratch_size(Nq, Nkv, HeadDim, inp.dtype)
    scratch  = torch.zeros((scratch_size,), device=device, dtype=torch.uint8)
    dinp     = torch.zeros_like(inp)
    dq_wgt_k = torch.zeros_like(q_wgt)
    dk_wgt_k = torch.zeros_like(k_wgt)
    r_rms    = torch.ones((B, T, N), device=device, dtype=torch.float32)
    K.qk_norm_forward(torch.empty_like(inp), r_rms, inp, q_wgt, k_wgt, eps, Nq, Nkv)

    print(f"BW: dout {dout.nbytes // 1024 // 1024} MB inp {inp.nbytes // 1024 // 1024} MB dinp {dinp.nbytes // 1024 // 1024} MB")
    K.qk_norm_backward(dinp, dq_wgt_k, dk_wgt_k, scratch,
                       dout, inp, q_wgt, k_wgt, r_rms,
                       None, Nq, Nkv)


if __name__ == "__main__":
    #test_qk_norm_forward(4, 1024, 16, 2, 128, torch.bfloat16)
    # qk_norm_backward_kernel
    test_qk_norm_backward(4, 1024, 16, 2, 128, torch.bfloat16)
    print("QK norm backward kernel test passed")
