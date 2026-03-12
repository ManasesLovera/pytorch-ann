import time

import torch


def header(title: str) -> None:
    print(f"\n=== {title} ===")


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available on this machine.")

    device = torch.device("cuda")
    print("CUDA available:", True)
    print("GPU:", torch.cuda.get_device_name(0))
    return device


def tensor_math_demo(device: torch.device) -> None:
    header("1. Tensor Math On GPU")
    a = torch.arange(1, 6, device=device, dtype=torch.float32)
    b = torch.arange(10, 15, device=device, dtype=torch.float32)

    print("a:", a)
    print("b:", b)
    print("a + b:", a + b)
    print("a * b:", a * b)
    print("mean(a * b):", (a * b).mean().item())


def matmul_demo(device: torch.device) -> None:
    header("2. Matrix Multiply Timing")
    size = 2048
    x = torch.randn(size, size, device=device)
    y = torch.randn(size, size, device=device)

    torch.cuda.synchronize()
    start = time.perf_counter()
    z = x @ y
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    print(f"matrix size: {size}x{size}")
    print(f"GPU matmul time: {elapsed:.4f} seconds")
    print("result checksum:", z.mean().item())


def autograd_demo(device: torch.device) -> None:
    header("3. Tiny Autograd Example")
    x = torch.tensor([2.0], device=device, requires_grad=True)
    y = (x**3) + 4 * x
    y.backward()

    print("x:", x.item())
    print("y = x^3 + 4x:", y.item())
    print("dy/dx:", x.grad.item())


def main() -> None:
    device = require_cuda()
    tensor_math_demo(device)
    matmul_demo(device)
    autograd_demo(device)


if __name__ == "__main__":
    main()
