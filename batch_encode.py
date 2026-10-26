"""Batched torch normalize + POD encode: many sub-domains in one pass, on any device.

Same math and output keys as conditional_cases._normalize_case + _encode_case
(per-case basis only), but every case in the batch shares one shape, so the
normalization is one broadcast and every POD is one batched torch.linalg.svd.
Built for a carve that already holds a window's sub-domains in memory: no raw
.npz write or read-back.

Input: dict of RAW_MEMBERS stacked on a leading batch axis (numpy or torch).
Output: one dict per case, numpy, ready for np.savez.
"""

from __future__ import annotations

import numpy as np
import torch

from conditional_cases import COORD_NAMES, STATE_VARS
from device import step

NC = len(COORD_NAMES)


def _t(arr, device: str) -> torch.Tensor:
    return torch.as_tensor(np.asarray(arr) if not torch.is_tensor(arr) else arr,
                           dtype=torch.float32, device=device)


def _svd_encode(mat: torch.Tensor, k: int):
    """Batched POD over the second-to-last axis of mat (..., n_samples, n_space).
    Returns mean (..., n_space), modes (..., n_space, k), coeffs (..., n_samples, k)."""
    mean = mat.mean(dim=-2, keepdim=True)
    U, S, Vh = torch.linalg.svd(mat - mean, full_matrices=False)
    ke = min(k, Vh.shape[-2])
    return mean.squeeze(-2), Vh[..., :ke, :].transpose(-1, -2), U[..., :ke] * S[..., None, :ke]


def normalize_batch(raw: dict, recipe: dict, device: str = "cpu") -> dict[str, torch.Tensor]:
    """_normalize_case over a batch: (arr - offset) / scale per member."""
    rv = {k: torch.tensor(recipe[k], dtype=torch.float32, device=device) for k in
          ("state_offset", "state_scale", "coord_offset", "coord_scale",
           "flux_offset", "flux_scale")}
    s, c, f = ((rv[f"{n}_offset"], rv[f"{n}_scale"]) for n in ("state", "coord", "flux"))
    norm = lambda a, rec: (a - rec[0]) / rec[1]

    initial = _t(raw["initial"], device)
    initial = torch.cat([norm(initial[..., :NC], c), norm(initial[..., NC:], s)], dim=-1)
    targets = _t(raw["targets"], device)
    mask = torch.isfinite(targets)
    targets = torch.where(mask, norm(torch.where(mask, targets, s[0]), s), 0.0)
    surface = _t(raw["surface"], device)
    xyt = c[0][[0, 1, 3]], c[1][[0, 1, 3]]
    surface = torch.cat([norm(surface[..., :3], xyt), norm(surface[..., 3:], f)], dim=-1)
    return {
        "initial": initial,
        "boundary": norm(_t(raw["boundary"], device), s),
        "boundary_coords": norm(_t(raw["boundary_coords"], device), c),
        "terrain": _t(raw["terrain"], device),
        "interior": norm(_t(raw["interior"], device), c),
        "targets": targets, "target_mask": mask.float(),
        "times": _t(raw["times"], device), "surface": surface,
    }


def encode_batch(raw: dict, recipe: dict, ks_initial: dict, ks_boundary: dict,
                 grid: tuple[int, int, int], keep_raw: bool = False,
                 device: str = "cpu") -> list[dict[str, np.ndarray]]:
    """Normalize + per-case POD encode a batch of same-shape sub-domains.
    grid = (nx, ny, nz) of the initial field (x,y-slow, z-fast rows)."""
    with step("normalize", device):
        m = normalize_batch(raw, recipe, device)
    nx, ny, nz = grid
    B = m["initial"].shape[0]
    out = {k: m[k] for k in ("boundary_coords", "terrain", "interior", "targets",
                             "target_mask", "times", "surface")}
    out["initial_coords"] = m["initial"][..., :NC]

    # boundary (B, nf, nt, fl, nv) -> (B, nf, nv, nt, fl): time-SVD per case,face,var
    with step("boundary svd", device):
        bnd = m["boundary"]
        bmat = bnd.permute(0, 1, 4, 2, 3)
        for vj, var in enumerate(STATE_VARS):
            mean, modes, coeffs = _svd_encode(bmat[:, :, vj], ks_boundary[var])
            for face in range(bnd.shape[1]):
                out[f"bnd_mean_{var}_{face}"] = mean[:, face]
                out[f"bnd_modes_{var}_{face}"] = modes[:, face]
                out[f"bnd_coeffs_{var}_{face}"] = coeffs[:, face]

    # initial (B, nx*ny*nz, nv) -> (B, nv, nz, nx*ny): spatial-SVD over z-levels
    with step("initial svd", device):
        grids = (m["initial"][..., NC:].reshape(B, nx, ny, nz, -1)
                 .permute(0, 4, 3, 1, 2).reshape(B, -1, nz, nx * ny))
        for vj, var in enumerate(STATE_VARS):
            mean, modes, coeffs = _svd_encode(grids[:, vj], ks_initial[var])
            out[f"ini_mean_{var}"] = mean
            out[f"ini_modes_{var}"] = modes
            out[f"ini_coeffs_{var}"] = coeffs
    if keep_raw:
        out["ini_raw"] = grids
        out["bnd_raw"] = bnd

    with step("to host", device):
        host = {k: v.contiguous().cpu().numpy() for k, v in out.items()}
    cases = [{k: v[b] for k, v in host.items()} for b in range(B)]
    for c in cases:
        c["initial_grid"] = np.array([nx, ny, nz], dtype=np.int64)
    return cases
