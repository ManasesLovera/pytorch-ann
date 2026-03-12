# CUDA basics with PyTorch

This workspace has a CUDA-capable NVIDIA GPU available through PyTorch.

## Run

```powershell
uv run test.py
uv run cuda_basics.py
```

## What `cuda_basics.py` shows

- Moving tensors onto the GPU
- Running parallel tensor math
- Timing a large matrix multiplication
- Using autograd on CUDA tensors

## Good next steps

- Build a tiny image classifier on GPU
- Compare CPU vs GPU timings for the same workload
- Learn batching, `DataLoader`, and mixed precision
