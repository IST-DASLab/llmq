import pytest
import torch
from pyllmq import kernels as K
from conftest import DTYPES, TOL


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
