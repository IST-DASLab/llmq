import pytest
import torch
from pyllmq import kernels as K
from conftest import DTYPES, TOL


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
