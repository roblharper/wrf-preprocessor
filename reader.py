"""Generic streaming NetCDF reader, driven by a SourceType record."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import numpy as np
import netCDF4

from config import SourceType, match_source
from device import array_module, to_device

log = logging.getLogger(__name__)

#: name -> flat 1-D array (canonical columns and raw derive-inputs).
Chunk = dict[str, np.ndarray]


def discover_files(input_root: Path) -> list[tuple[Path, SourceType]]:
    """Walk the tree, match each .nc/.cdf file to a SourceType by filename.

    Any folder layout works. Unmatched files are reported and skipped.
    """
    if not input_root.is_dir():
        raise NotADirectoryError(f"Input root does not exist: {input_root}")

    matched: list[tuple[Path, SourceType]] = []
    unmatched: list[Path] = []
    for p in sorted(input_root.rglob("*")):
        if not (p.is_file() and p.suffix in (".nc", ".cdf")):
            continue
        source = match_source(p.name)
        if source is not None:
            matched.append((p, source))
            log.debug("matched %s -> %s", p.name, source.name)
        else:
            unmatched.append(p)

    if unmatched:
        names = ", ".join(sorted({u.name for u in unmatched})[:6])
        log.info("%d file(s) matched no type, skipped: %s", len(unmatched), names)
    if not matched:
        raise FileNotFoundError(f"No recognized NetCDF files under {input_root}.")
    return matched


def read_file(path: Path, source: SourceType, *, chunk_size: int = 1,
              device: str = "cpu") -> Iterator[Chunk]:
    """Stream a file in blocks along ``chunk_dim`` (never loaded whole). ``device``
    selects where the block math runs: 'cpu' (numpy) or a GPU device (torch)."""
    ds = netCDF4.Dataset(path)
    try:
        if source.chunk_dim not in ds.dimensions:
            raise KeyError(
                f"{path.name}: chunk_dim '{source.chunk_dim}' not a dimension "
                f"(has: {', '.join(ds.dimensions)})."
            )
        n = len(ds.dimensions[source.chunk_dim])
        for start in range(0, n, chunk_size):
            yield _read_block(ds, source, start, min(start + chunk_size, n), device)
    finally:
        ds.close()


def _read_block(ds: "netCDF4.Dataset", source: SourceType, start: int, stop: int,
                device: str = "cpu") -> Chunk:
    """Read the column_map + derive_inputs variables for [start, stop) onto device."""
    xp = array_module(device)
    wanted = list(source.column_map.items()) + [(v, v) for v in source.derive_inputs]
    sliced: dict[str, tuple[np.ndarray, tuple[str, ...]]] = {}
    for out_key, varname in wanted:
        if varname in ds.variables:
            arr, dims = _slice_along(ds.variables[varname], source.chunk_dim, start, stop)
            if arr.dtype.kind == "S":                       # WRF Times char array
                arr, dims = _parse_time_strings(arr, dims, source.chunk_dim)
            arr, dims = _destagger(to_device(arr, device), dims, xp)
            sliced[out_key] = (arr, dims)
    return _broadcast_to_rows(sliced, xp)


def _parse_time_strings(
    arr: np.ndarray, dims: tuple[str, ...], chunk_dim: str
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Decode a WRF ``Times`` char array (N, 19) to epoch seconds, shape (N,)."""
    from datetime import datetime, timezone

    rows = arr.reshape(arr.shape[0], -1)                    # (N, DateStrLen)
    epochs = np.empty(rows.shape[0], dtype=np.float64)
    for i, row in enumerate(rows):
        stamp = b"".join(np.asarray(row).ravel()).decode().replace("_", " ")
        dt = datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        epochs[i] = dt.timestamp()
    return epochs, (chunk_dim,)


#: WRF Arakawa-C staggered dims -> the mass-grid dim they interpolate onto.
_STAGGER_MAP = {
    "west_east_stag": "west_east",
    "south_north_stag": "south_north",
    "bottom_top_stag": "bottom_top",
}


def _destagger(arr: np.ndarray, dims: tuple[str, ...], xp=np) -> tuple[np.ndarray, tuple[str, ...]]:
    """Average staggered (cell-face) axes onto the mass grid so dim names align.
    Winds live on faces (N+1), scalars on centres (N). ``xp`` is numpy or torch."""
    out_dims = list(dims)
    for i, d in enumerate(dims):
        if d in _STAGGER_MAP:
            lo = arr[_axis_slice(arr.ndim, i, 0, arr.shape[i] - 1)]
            hi = arr[_axis_slice(arr.ndim, i, 1, arr.shape[i])]
            arr = 0.5 * (lo + hi)
            out_dims[i] = _STAGGER_MAP[d]
    return arr, tuple(out_dims)


def _axis_slice(ndim: int, axis: int, start: int, stop: int) -> tuple:
    """Index tuple selecting [start:stop] on one axis, full slices elsewhere."""
    idx = [slice(None)] * ndim
    idx[axis] = slice(start, stop)
    return tuple(idx)


def _slice_along(
    var: "netCDF4.Variable", chunk_dim: str, start: int, stop: int
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Read a variable, slicing chunk_dim if present; return array + dim names."""
    if chunk_dim in var.dimensions:
        index: list[object] = [slice(None)] * var.ndim
        index[var.dimensions.index(chunk_dim)] = slice(start, stop)
        return np.asarray(var[tuple(index)]), tuple(var.dimensions)
    return np.asarray(var[...]), tuple(var.dimensions)


def _broadcast_to_rows(sliced: dict[str, tuple[np.ndarray, tuple[str, ...]]], xp=np) -> Chunk:
    """Expand every column onto a common grid by dimension name, then flatten.
    Broadcasting by name (not trailing-aligned) keeps each value on its axis."""
    if not sliced:
        return {}

    axis_order: list[str] = []
    for _, dims in sorted(sliced.values(), key=lambda ad: -len(ad[1])):
        for d in dims:
            if d not in axis_order:
                axis_order.append(d)

    target_shape = _target_shape(sliced, axis_order)
    return {
        column: xp.broadcast_to(_place_on_axes(arr, dims, axis_order), target_shape).reshape(-1)
        for column, (arr, dims) in sliced.items()
    }


def _target_shape(
    sliced: dict[str, tuple[np.ndarray, tuple[str, ...]]], axis_order: list[str]
) -> tuple[int, ...]:
    length: dict[str, int] = {}
    for arr, dims in sliced.values():
        for d, n in zip(dims, arr.shape):
            length[d] = n
    return tuple(length.get(d, 1) for d in axis_order)


def _place_on_axes(arr: np.ndarray, dims: tuple[str, ...], axis_order: list[str]) -> np.ndarray:
    """Reshape so each dimension sits on its axis in axis_order; others size-1."""
    shape = [arr.shape[dims.index(d)] if d in dims else 1 for d in axis_order]
    return arr.reshape(shape)
