"""CUDA launch vs. finish. Needs a GPU; otherwise prints a skip message.

Run::

    uv sync --extra torch
    uv run python scripts/cuda_async.py
"""

from __future__ import annotations

import time

try:
    import torch
except ImportError:
    raise SystemExit("install torch: uv sync --extra torch") from None


def main() -> None:
    if not torch.cuda.is_available():
        print("no CUDA device (check driver / nvidia-smi / a cu* torch wheel)")
        return

    print("device", torch.cuda.get_device_name(0))
    x = torch.randn(2048, 4096, device="cuda")
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    y = x @ x.T
    t_launch = time.perf_counter() - t0
    torch.cuda.synchronize()
    t_done = time.perf_counter() - t0
    print(f"python returned after {t_launch * 1e3:.2f} ms")
    print(f"GPU finished          {t_done * 1e3:.2f} ms after launch")

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    y = y @ y
    end.record()
    torch.cuda.synchronize()
    print(f"cuda Event elapsed     {start.elapsed_time(end):.2f} ms")

    t0 = time.perf_counter()
    _ = float(y[0, 0].item())
    print(f".item() (host sync)    {(time.perf_counter() - t0) * 1e3:.2f} ms")

    n = y.shape[0]
    ids_list = list(range(0, n, 64))
    ids_cuda = torch.arange(0, n, 64, device="cuda")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = y[ids_list]
    torch.cuda.synchronize()
    print(f"index with python list {(time.perf_counter() - t0) * 1e3:.2f} ms")
    t0 = time.perf_counter()
    _ = y[ids_cuda]
    torch.cuda.synchronize()
    print(f"index with cuda tensor  {(time.perf_counter() - t0) * 1e3:.2f} ms")


if __name__ == "__main__":
    main()
