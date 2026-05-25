import math

import pytest
import torch
from pyllmq import kernels as K
from conftest import DTYPES, TOL

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
