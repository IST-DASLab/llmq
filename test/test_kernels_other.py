import pytest
import torch
import torch.nn.functional as F
from pyllmq import kernels as K
from conftest import DTYPES, TOL


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

@pytest.mark.parametrize("mean,std", [(0.0, 1.0), (1.5, 0.25)])
def test_fill_normal_subsequence_independence(mean, std):
    """Different subsequence offsets with the same seed must produce
    non-identical, uncorrelated draws."""
    N = 64 * 1024
    seed = 0xABCDEF01

    tensors = []
    for subseq in range(4):
        x = torch.empty((N,), device="cuda", dtype=torch.float32)
        K.fill_normal(x, float(mean), float(std), seed, subseq)
        tensors.append(x.cpu())

    for i in range(len(tensors)):
        for j in range(i + 1, len(tensors)):
            # Bit-identity: the subsequence argument must actually be used.
            assert not torch.equal(tensors[i], tensors[j]), (
                f"Subsequences {i} and {j} produced identical output"
            )

            # Pearson |r| < 0.05 is a very conservative bound at N=64k
            # (3-sigma ≈ 0.012), so a genuine failure stands out clearly.
            a = tensors[i] - tensors[i].mean()
            b = tensors[j] - tensors[j].mean()
            r = (a * b).mean() / (a.std() * b.std())
            assert abs(r.item()) < 0.05, (
                f"Subsequences {i} and {j} are correlated: r={r.item():.4f}"
            )

@pytest.mark.parametrize("mean,std", [(0.0, 1.0), (1.5, 0.25), (-1.0, 2.0)])
def test_fill_normal_stats_bf16(mean, std):
    """BF16 should satisfy the same mean/std checks as float32 with
    tolerances relaxed to reflect its ~0.8 % per-value precision."""
    N = 256 * 1024
    seed = 0xABCDEF01
    x = torch.empty((N,), device="cuda", dtype=torch.bfloat16)
    K.fill_normal(x, float(mean), float(std), seed, 0)

    # Upcast before reducing — BF16 accumulation is itself lossy.
    xf = x.float()
    m = xf.mean().item()
    s = xf.std(unbiased=True).item()

    assert abs(m - mean) < max(1e-2, 0.05 * abs(std))
    assert abs(s - std) / max(std, 1e-12) < 0.05

    # Determinism check mirrors the float32 test.
    y = torch.empty_like(x)
    K.fill_normal(y, float(mean), float(std), seed, 0)
    assert torch.equal(x.cpu(), y.cpu())

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
