"""Generate a folder-per-case phony input tree for testing.

Each case folder holds one HRRR file (tags the case time) plus one of every
recognized LES/sensor type. Every file in a folder contributes rows
unconditionally; there is no time/bbox matching to test.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import netCDF4

warnings.filterwarnings(
    "ignore",
    message="Setting the shape on a NumPy array has been deprecated",
    category=DeprecationWarning,
)

SYNC_TIME = 1_720_580_400.0
REGION_LON = (-97.60, -97.35)
REGION_LAT = (36.50, 36.70)
REGION_Z = (0.0, 1000.0)
_DAY0 = SYNC_TIME - (SYNC_TIME % 86400.0)      # midnight base for ARM offsets


# --- anchor + interior (gridded snapshots) ----------------------------------
def _write_hrrr(path: Path, *, valid_time: float, seed: int) -> None:
    rng = np.random.default_rng(seed)
    ny = nx = 4; nlev = 3
    ds = netCDF4.Dataset(path, "w")
    ds.createDimension("time", 1); ds.createDimension("hybrid_level", nlev)
    ds.createDimension("y", ny); ds.createDimension("x", nx)
    lon = np.linspace(*REGION_LON, nx); lat = np.linspace(*REGION_LAT, ny)
    lon2d, lat2d = np.meshgrid(lon, lat)
    ds.createVariable("longitude", "f4", ("y", "x"))[:, :] = lon2d
    ds.createVariable("latitude", "f4", ("y", "x"))[:, :] = lat2d
    ds.createVariable("valid_time", "f8", ("time",))[:] = [valid_time]
    shape = (1, nlev, ny, nx)
    for name, base in (("u_wind", 5.0), ("v_wind", 2.0), ("vertical_velocity", 0.1),
                       ("temperature", 290.0), ("pressure", 95000.0),
                       ("geopotential_height", 500.0)):
        ds.createVariable(name, "f4", ("time", "hybrid_level", "y", "x"))[:, :, :, :] = \
            base + rng.normal(0, 0.1, size=shape)
    ds.close()


def _wrf_times(valid_time: float) -> np.ndarray:
    """WRF Times char array (1, 19) for one epoch: 'YYYY-MM-DD_HH:MM:SS'."""
    from datetime import datetime, timezone

    stamp = datetime.fromtimestamp(valid_time, tz=timezone.utc).strftime("%Y-%m-%d_%H:%M:%S")
    return np.array([list(stamp)], dtype="S1")


def _write_les(path: Path, *, valid_time: float, seed: int, lon_c: float, lat_c: float) -> None:
    """Real LASSO wrfout shape: Times char stamp + Arakawa-C staggered U/V/W."""
    rng = np.random.default_rng(seed)
    n = 5; nz = 3
    ds = netCDF4.Dataset(path, "w")
    ds.createDimension("Time", 1); ds.createDimension("DateStrLen", 19)
    ds.createDimension("bottom_top", nz); ds.createDimension("bottom_top_stag", nz + 1)
    ds.createDimension("south_north", n); ds.createDimension("south_north_stag", n + 1)
    ds.createDimension("west_east", n); ds.createDimension("west_east_stag", n + 1)
    lon = np.linspace(lon_c - 0.08, lon_c + 0.08, n)
    lat = np.linspace(lat_c - 0.06, lat_c + 0.06, n)
    lon2d, lat2d = np.meshgrid(lon, lat)
    ds.createVariable("XLONG", "f4", ("Time", "south_north", "west_east"))[:, :, :] = lon2d
    ds.createVariable("XLAT", "f4", ("Time", "south_north", "west_east"))[:, :, :] = lat2d
    ds.createVariable("HGT", "f4", ("Time", "south_north", "west_east"))[:, :, :] = 300.0
    ds.createVariable("Times", "S1", ("Time", "DateStrLen"))[:, :] = _wrf_times(valid_time)
    # winds on staggered (cell-face) grids; T/P scalars on the mass grid
    for name, base, dims in (
        ("U", 5.0, ("Time", "bottom_top", "south_north", "west_east_stag")),
        ("V", 2.0, ("Time", "bottom_top", "south_north_stag", "west_east")),
        ("W", 0.1, ("Time", "bottom_top_stag", "south_north", "west_east")),
        ("T", 1.5, ("Time", "bottom_top", "south_north", "west_east")),
        ("P", 20.0, ("Time", "bottom_top", "south_north", "west_east")),
    ):
        shape = tuple(len(ds.dimensions[d]) for d in dims)
        ds.createVariable(name, "f4", dims)[:] = base + rng.normal(0, 0.1, size=shape)
    ds.close()


# --- ARM observation streams (base_time + time_offset) ----------------------
def _arm_time(ds, times: np.ndarray) -> None:
    """Write ARM-style time: base_time (scalar epoch) + time_offset (s)."""

    ds.createVariable("base_time", "f8", ())[...] = _DAY0
    ds.createVariable("time_offset", "f8", ("time",))[:] = times - _DAY0


def _write_ecor(path: Path, *, times: np.ndarray, seed: int, lon, lat, alt) -> None:
    rng = np.random.default_rng(seed)
    ds = netCDF4.Dataset(path, "w"); ds.createDimension("time", len(times))
    _arm_time(ds, times)
    ds.createVariable("lon", "f8", ())[...] = lon
    ds.createVariable("lat", "f8", ())[...] = lat
    ds.createVariable("alt", "f8", ())[...] = alt
    for name, base in (("wind_u", 5.0), ("wind_v", 2.0), ("wind_w", 0.1)):
        ds.createVariable(name, "f4", ("time",))[:] = base + rng.normal(0, 0.2, len(times))
    ds.close()


def _write_smos(path: Path, *, times: np.ndarray, seed: int) -> None:
    """Soil-met: wind as speed/direction (derive -> u,v); no lat/lon in file."""

    rng = np.random.default_rng(seed)
    ds = netCDF4.Dataset(path, "w"); ds.createDimension("time", len(times))
    _arm_time(ds, times)
    ds.createVariable("wspd", "f4", ("time",))[:] = 5.0 + rng.normal(0, 0.3, len(times))
    ds.createVariable("wdir", "f4", ("time",))[:] = 270.0 + rng.normal(0, 5, len(times))
    ds.createVariable("temp", "f4", ("time",))[:] = 20.0 + rng.normal(0, 0.5, len(times))
    ds.close()


def _write_twr(path: Path, *, times: np.ndarray, seed: int, lon, lat, alt) -> None:
    """Tower met: Celsius temperature (derive -> T), lat/lon/alt, no wind."""

    rng = np.random.default_rng(seed)
    ds = netCDF4.Dataset(path, "w"); ds.createDimension("time", len(times))
    _arm_time(ds, times)
    ds.createVariable("lon", "f8", ())[...] = lon
    ds.createVariable("lat", "f8", ())[...] = lat
    ds.createVariable("alt", "f8", ())[...] = alt
    ds.createVariable("temp", "f4", ("time",))[:] = 18.0 + rng.normal(0, 0.4, len(times))
    ds.createVariable("rh", "f4", ("time",))[:] = 60.0 + rng.normal(0, 2, len(times))
    ds.close()


def _write_co2flx(path: Path, *, times: np.ndarray, seed: int) -> None:
    """Flux + met: mostly fluxes; a little canonical state (bar_pres -> P)."""

    rng = np.random.default_rng(seed)
    ds = netCDF4.Dataset(path, "w"); ds.createDimension("time", len(times))
    _arm_time(ds, times)
    ds.createVariable("bar_pres", "f4", ("time",))[:] = 95.0 + rng.normal(0, 0.1, len(times))
    ds.createVariable("h", "f4", ("time",))[:] = rng.normal(100, 10, len(times))     # sensible heat
    ds.createVariable("le", "f4", ("time",))[:] = rng.normal(150, 15, len(times))    # latent heat
    ds.close()


def _write_armbeatm(path: Path, *, times: np.ndarray, seed: int) -> None:
    """Best-estimate profiles: vertical dims + surface wind (mapped)."""

    rng = np.random.default_rng(seed)
    nheight = 6
    ds = netCDF4.Dataset(path, "w")
    ds.createDimension("time", len(times)); ds.createDimension("height", nheight)
    _arm_time(ds, times)
    ds.createVariable("height", "f8", ("height",))[:] = np.linspace(10, 2000, nheight)
    ds.createVariable("u_wind_sfc", "f4", ("time",))[:] = 5.0 + rng.normal(0, 0.3, len(times))
    ds.createVariable("v_wind_sfc", "f4", ("time",))[:] = 2.0 + rng.normal(0, 0.3, len(times))
    ds.close()


def _write_dlaux(path: Path, *, times: np.ndarray, seed: int) -> None:
    """Doppler lidar housekeeping: no canonical fields (registered, unmapped)."""

    rng = np.random.default_rng(seed)
    ds = netCDF4.Dataset(path, "w"); ds.createDimension("time", len(times))
    _arm_time(ds, times)
    ds.createVariable("battery_voltage", "f4", ("time",))[:] = 12.0 + rng.normal(0, 0.1, len(times))
    ds.createVariable("pitch", "f4", ("time",))[:] = rng.normal(0, 0.5, len(times))
    ds.close()


def _write_case(folder: Path, *, valid_time: float, seed: int) -> None:
    """Populate one case folder: HRRR tag + one file of every recognized type."""
    folder.mkdir(parents=True, exist_ok=True)
    lon_c, lat_c = float(np.mean(REGION_LON)), float(np.mean(REGION_LAT))
    times = valid_time + np.linspace(-600, 600, 20)   # sensor sampling around the tag

    _write_hrrr(folder / "hrrr.nc", valid_time=valid_time, seed=seed + 1)
    _write_les(folder / "wrfout_d01.nc", valid_time=valid_time, seed=seed + 4,
               lon_c=lon_c, lat_c=lat_c)
    _write_ecor(folder / "sgpecorsfwindC1.b1.nc", times=times, seed=seed + 7,
                lon=lon_c - 0.02, lat=lat_c + 0.01, alt=320.0)
    _write_smos(folder / "sgpsmosC1.b1.nc", times=times, seed=seed + 11)
    _write_twr(folder / "sgptwr25mC1.b1.nc", times=times, seed=seed + 12,
               lon=lon_c - 0.01, lat=lat_c + 0.02, alt=316.0)
    _write_co2flx(folder / "sgpco2flx4mmetC1.b1.nc", times=times, seed=seed + 13)
    _write_armbeatm(folder / "sgparmbeatmC1.c1.nc", times=times, seed=seed + 14)
    _write_dlaux(folder / "sgpdlauxC1.b1.nc", times=times, seed=seed + 15)


def generate(root: Path, *, seed: int = 0) -> dict:
    """Build the folder-per-case tree under ``root``; return a manifest for tests."""

    case_times = {"case_morning": SYNC_TIME, "case_afternoon": SYNC_TIME + 7 * 3600.0}
    for i, (name, vt) in enumerate(case_times.items()):
        _write_case(root / name, valid_time=vt, seed=seed + 100 * i)

    return {
        "case_ids": sorted(case_times),
        "hrrr_times": case_times,
        "region_lon": REGION_LON,
        "region_lat": REGION_LAT,
        "unmapped_file": "sgpdlaux",
        # every non-HRRR type placed in each case folder contributes rows
        "data_types": ["wrfout", "ecorsfwind", "smos", "twr", "co2flx", "armbeatm"],
    }


if __name__ == "__main__":
    import sys
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("fixtures/generated")
    manifest = generate(dest)
    print(f"wrote phony data to {dest}")
    for k, v in manifest.items():
        print(f"  {k}: {v}")
