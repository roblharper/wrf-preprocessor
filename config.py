"""Canonical schema and the instrument-type registry.

Adding an instrument is one SourceType record here; no other file changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


#: Output columns, in order. The writer emits and the PINN reads exactly this.
CANONICAL_COLUMNS: tuple[str, ...] = ("x", "y", "z", "t", "u", "v", "w", "T", "P")
COLUMN_INDEX: dict[str, int] = {name: i for i, name in enumerate(CANONICAL_COLUMNS)}

#: Half-width of the time-match window: rows within +/- this of a snapshot time
#: are synced to it. Sensors are high-rate, so an exact match is too strict.
TIME_TOLERANCE_SECONDS = 1800.0   # +/- 30 min (half of hourly HRRR cadence)

# --- derive hooks: compute canonical columns that are not plain renames -------
Raw = dict[str, np.ndarray]


def _time_from_base_offset(raw: Raw) -> Raw:
    """ARM absolute time = base_time (epoch) + time_offset (s)."""
    if "base_time" in raw and "time_offset" in raw:
        return {"t": raw["base_time"] + raw["time_offset"]}
    return {}


def _uv_from_speed_dir(raw: Raw) -> Raw:
    """Wind speed + direction-from (deg) -> eastward u, northward v."""
    if "wspd" in raw and "wdir" in raw:
        theta = np.deg2rad(raw["wdir"])
        return {"u": -raw["wspd"] * np.sin(theta), "v": -raw["wspd"] * np.cos(theta)}
    return {}


def _smos(raw: Raw) -> Raw:
    return {**_time_from_base_offset(raw), **_uv_from_speed_dir(raw)}


def _tower_temp(raw: Raw) -> Raw:
    """ARM time, plus air temperature Celsius -> Kelvin."""
    out = _time_from_base_offset(raw)
    if "temp" in raw:
        out["T"] = raw["temp"] + 273.15
    return out


@dataclass(frozen=True)
class SourceType:
    """How one instrument's NetCDF files map onto the canonical schema.

    ``column_map`` is direct renames; ``derive`` computes the rest from
    ``derive_inputs``. ``canonical=False`` registers a type with no canonical
    fields (recognized, but contributes no rows).
    """

    name: str
    match: str                          # filename substring identifying the type
    is_anchor: bool = False             # True only for HRRR (defines the snapshots)
    column_map: dict[str, str] = field(default_factory=dict)
    derive: Callable[[Raw], Raw] | None = None
    derive_inputs: tuple[str, ...] = ()
    chunk_dim: str = "time"
    canonical: bool = True
    note: str = ""


# Add an instrument = add a record. Names verified against the on-disk files.
_ARM_TIME = ("base_time", "time_offset")

REGISTRY: tuple[SourceType, ...] = (
    SourceType(
        name="HRRR forecast tile", match="hrrr", is_anchor=True,
        column_map={
            "x": "longitude", "y": "latitude", "z": "geopotential_height",
            "t": "valid_time",
            "u": "u_wind", "v": "v_wind", "w": "vertical_velocity",
            "T": "temperature", "P": "pressure",
        },
        note="lat/lon degrees on hybrid pressure levels; the inlet condition",
    ),
    SourceType(
        name="FastEddy / LASSO LES", match="les",
        column_map={
            "x": "x0", "y": "y0", "z": "z4", "t": "time",
            "u": "uu", "v": "vv", "w": "ww", "P": "pressure",  # theta->T later
        },
        note="high-res interior truth; metres frame (reconciliation pending)",
    ),
    SourceType(
        name="ecor sonic wind", match="ecorsfwind",
        column_map={
            "x": "lon", "y": "lat", "z": "alt",
            "u": "wind_u", "v": "wind_v", "w": "wind_w",
        },
        derive=_time_from_base_offset, derive_inputs=_ARM_TIME,
        note="single-point 3-D sonic wind at ~10 Hz",
    ),
    SourceType(
        name="soil-met surface wx", match="smos",
        derive=_smos, derive_inputs=_ARM_TIME + ("wspd", "wdir"),
        note="wind as speed/direction; no lat/lon stored",
    ),
    SourceType(
        name="tower met (T/RH)", match="twr",
        column_map={"x": "lon", "y": "lat", "z": "alt"},
        derive=_tower_temp, derive_inputs=_ARM_TIME + ("temp",),
        note="tower temperature/humidity; no wind",
    ),
    SourceType(
        name="flux + met", match="co2flx",
        column_map={"P": "bar_pres"},
        derive=_time_from_base_offset, derive_inputs=_ARM_TIME,
        note="mostly fluxes; little canonical state",
    ),
    SourceType(
        name="best-estimate profiles", match="armbeatm",
        column_map={"u": "u_wind_sfc", "v": "v_wind_sfc"},
        derive=_time_from_base_offset, derive_inputs=_ARM_TIME,
        note="merged vertical profiles; surface wind mapped for now",
    ),
    SourceType(
        name="doppler lidar aux", match="dlaux",
        canonical=False,
        note="lidar housekeeping; registered but unmapped",
    ),
)


def match_source(filename: str) -> SourceType | None:
    """First registry type whose ``match`` appears in ``filename``, else None."""
    for source in REGISTRY:
        if source.match in filename:
            return source
    return None
