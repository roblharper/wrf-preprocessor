"""Device selection for the compute path: pick the array module and allocator.
CPU uses numpy; a GPU device uses torch. Only the swap points live here."""

from __future__ import annotations

import os
import time

import numpy as np

_DEBUG = bool(os.environ.get("PREPROC_DEBUG"))


def sync(device: str) -> None:
    """Block until queued GPU work finishes, so timers measure real cost."""
    if device != "cpu":
        import torch
        torch.cuda.synchronize()


def step(label: str, device: str = "cpu"):
    """Context manager: print '[t] label: X.XXs' when PREPROC_DEBUG is set. GPU
    work is synced first so the time is real, not just kernel-launch."""
    class _S:
        def __enter__(self):
            self.t0 = time.perf_counter()
            return self
        def __exit__(self, *a):
            if _DEBUG:
                sync(device)
                print(f"      [t] {label}: {time.perf_counter() - self.t0:.3f}s", flush=True)
    return _S()


def array_module(device: str):
    """Return the array module for ``device`` ('cpu' -> numpy, else torch)."""
    if device == "cpu":
        return np
    import torch
    return torch


def to_device(arr, device: str):
    """Move a raw numpy array onto ``device`` (no-op for cpu)."""
    if device == "cpu":
        return np.asarray(arr, dtype=np.float64)
    import torch
    return torch.as_tensor(np.asarray(arr), dtype=torch.float64, device=device)


def make_empty(device: str):
    """Return an ``empty(shape)`` allocator producing float64 arrays on ``device``."""
    if device == "cpu":
        return lambda shape: np.empty(shape, dtype=np.float64)
    import torch
    return lambda shape: torch.empty(shape, dtype=torch.float64, device=device)


def to_numpy(arr) -> np.ndarray:
    """Bring an array back to host numpy (for spilling / writing)."""
    return arr.detach().cpu().numpy() if hasattr(arr, "detach") else np.asarray(arr)
