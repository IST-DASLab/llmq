import torch


DTYPES = [torch.float32, torch.bfloat16]

# Tolerances by dtype for approximate kernels: (rtol, atol)
TOL = {
    torch.float32: (5e-4, 5e-5),
    torch.bfloat16: (1e-2, 5e-3),
}
