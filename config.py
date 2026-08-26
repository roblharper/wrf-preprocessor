"""Canonical schema and the instrument-type registry.

Adding an instrument is one SourceType record here; no other file changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


#: Physical columns read from the files, in order.
PHYSICAL_COLUMNS: tuple[str, ...] = ("x", "y", "z", "t", "u", "v", "w",
                                     "theta", "p_prime")

#: Columns a row MUST have. A row is dropped only if one of these is NaN (it has
#: no usable position/time). Everything else (u,v,w,theta,p_prime) may be NaN for
#: a source that does not measure it; missing optional columns are logged, kept.
REQUIRED_COLUMNS: tuple[str, ...] = ("x", "y", "z", "t")

#: Full output schema: the physical columns plus a per-row integer source tag
#: (which SourceType the row came from), so the loss can weight sources.
CANONICAL_COLUMNS: tuple[str, ...] = PHYSICAL_COLUMNS + ("source",)
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


# Source category codes written to the 'source' column, so the loss can weight
# simulation vs sensor. HRRR is NOT a data source: each .npy IS an HRRR snapshot
# condition, so HRRR only anchors the case (time + bbox) and contributes no rows.
SRC_SIM = 0         # LES (LASSO)
SRC_SENSOR = 1      # all ground-observation streams


@dataclass(frozen=True)
class SourceType:
    """How one instrument's NetCDF files map onto the canonical schema.

    ``column_map`` is direct renames; ``derive`` computes the rest from
    ``derive_inputs``. ``canonical=False`` registers a type with no canonical
    fields (recognized, but contributes no rows).
    """

    name: str
    match: str                          # filename substring identifying the type
    source_code: int = -1               # per-row tag written to the 'source' column
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
        column_map={"x": "longitude", "y": "latitude", "t": "valid_time"},
        note="anchor only: defines each snapshot's time + x/y bbox, not a data source",
    ),
    SourceType(
        name="LASSO WRF-LES", match="wrfout", source_code=SRC_SIM,
        column_map={
            "x": "XLONG", "y": "XLAT", "z": "HGT", "t": "valid_time",
            "u": "U", "v": "V", "w": "W",
            "theta": "T", "p_prime": "P",   # wrfout T = pert. theta, P = pert. pressure
        },
        chunk_dim="Time",
        note="wrfout in lon/lat degrees, m; theta/p' as perturbations",
    ),
    SourceType(
        name="FastEddy LES", match="FE_NBL", source_code=SRC_SIM,
        column_map={
            "x": "x0", "y": "y0", "z": "z4", "t": "time",
            "u": "uu", "v": "vv", "w": "ww",
            "theta": "theta", "p_prime": "pressure",   # theta (K), pressure is a perturbation
        },
        note="standalone LES; local x/y/z in metres, theta/p' as perturbations",
    ),
    SourceType(
        name="ecor sonic wind", match="ecorsfwind", source_code=SRC_SENSOR,
        column_map={
            "x": "lon", "y": "lat", "z": "alt",
            "u": "wind_u", "v": "wind_v", "w": "wind_w",
        },
        derive=_time_from_base_offset, derive_inputs=_ARM_TIME,
        note="single-point 3-D sonic wind at ~10 Hz",
    ),
    SourceType(
        name="soil-met surface wx", match="smos", source_code=SRC_SENSOR,
        derive=_smos, derive_inputs=_ARM_TIME + ("wspd", "wdir"),
        note="wind as speed/direction; no lat/lon stored",
    ),
    SourceType(
        name="tower met (T/RH)", match="twr", source_code=SRC_SENSOR,
        column_map={"x": "lon", "y": "lat", "z": "alt"},
        derive=_time_from_base_offset, derive_inputs=_ARM_TIME,
        note="tower met; no wind components",
    ),
    SourceType(
        name="flux + met", match="co2flx", source_code=SRC_SENSOR,
        derive=_time_from_base_offset, derive_inputs=_ARM_TIME,
        note="mostly fluxes; little canonical state",
    ),
    SourceType(
        name="best-estimate profiles", match="armbeatm", source_code=SRC_SENSOR,
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
