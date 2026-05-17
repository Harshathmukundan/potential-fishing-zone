"""
data_processor.py — NetCDF Preprocessing for Inference
AI-Based Geospatial Decision Support System
SRMIST — Harshath Mukundan & Praveen N
"""

import numpy as np
import xarray as xr
import glob
import os
from skimage.transform import resize

TARGET   = 256
LAT_MIN, LAT_MAX = 5.0,  14.5
LON_MIN, LON_MAX = 77.0, 84.92

_das       = None
_N_TIMES   = None
_land_mask = None


def clean(arr):
    arr = arr.astype(np.float32)
    arr[arr == -32768] = np.nan
    arr[arr < -1e10]   = np.nan
    m = np.nanmean(arr)
    return np.where(np.isnan(arr), m, arr)


def norm(arr):
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-8:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def to_target(arr):
    return resize(arr, (TARGET, TARGET), anti_aliasing=True).astype(np.float32)


def safe_open(path):
    try:
        return xr.open_dataset(path, engine='netcdf4')
    except Exception:
        return xr.open_dataset(path, engine='h5netcdf')


def load_data(data_dir):
    """Load CMEMS datasets from directory. Returns (das, N_TIMES)."""
    global _das, _N_TIMES
    if _das is not None:
        return _das, _N_TIMES

    all_nc = sorted(glob.glob(os.path.join(data_dir, '*.nc')))
    # Also search subdirectories if no files found at top level
    if not all_nc:
        all_nc = sorted(glob.glob(os.path.join(data_dir, '**', '*.nc'), recursive=True))

    def find(pattern):
        m = [f for f in all_nc if pattern in f]
        if not m:
            raise FileNotFoundError(f"No file matching '{pattern}'")
        return m[0]

    ds_sst = safe_open(find('8369287'))
    ds_vo  = safe_open(find('8416191'))
    ds_zos = safe_open(find('8480927'))
    ds_uo  = safe_open(find('8817739'))
    ds_chl = safe_open(find('obs-oc'))

    def surface(da, dim='depth'):
        return da.isel(**{dim: 0}) if dim in da.dims else da

    _das = {
        'sst': surface(ds_sst['thetao']),
        'vo':  surface(ds_vo['vo']),
        'uo':  surface(ds_uo['uo']),
        'ssh': ds_zos['zos'],
        'chl': ds_chl.get('CHL', ds_chl.get('CHL_uncertainty'))
    }
    _N_TIMES = min(len(v.time) for v in _das.values())

    # Pre-compute land mask from raw SST (NaN = land)
    global _land_mask
    raw_sst = _das['sst'].isel(time=0).values
    land_raw = np.isnan(raw_sst) | (raw_sst < -1e10) | (raw_sst == -32768)
    _land_mask = resize(land_raw.astype(float), (TARGET, TARGET),
                        order=0, anti_aliasing=False) > 0.5

    return _das, _N_TIMES


def get_land_mask():
    """Return (TARGET, TARGET) boolean mask: True = land pixel."""
    global _land_mask
    if _land_mask is None:
        return np.zeros((TARGET, TARGET), dtype=bool)
    return _land_mask


def get_date_for_index(das, t):
    """Return the calendar date string for time index t."""
    try:
        ts = das['sst'].time.values[int(t)]
        return str(ts)[:10]   # e.g. '2020-09-27'
    except (IndexError, KeyError):
        return f'Day {t}'


def build_sample(das, t):
    """Build 8-channel feature map for time t."""
    t = int(np.clip(t, 0, min(len(v.time) for v in das.values()) - 1))

    sst = norm(clean(das['sst'].isel(time=t).values))
    ssh = norm(clean(das['ssh'].isel(time=t).values))
    uo  = norm(clean(das['uo'].isel(time=t).values))
    vo  = norm(clean(das['vo'].isel(time=t).values))
    chl = norm(clean(das['chl'].isel(time=t).values))

    gy, gx   = np.gradient(sst)
    sst_grad = norm(np.sqrt(gx**2 + gy**2))
    curr_spd = norm(np.sqrt(uo**2 + vo**2))
    div      = norm(np.abs(np.gradient(uo)[1] + np.gradient(vo)[0]))

    channels = [sst, ssh, uo, vo, chl, sst_grad, curr_spd, div]
    return np.stack([to_target(c) for c in channels], axis=-1)


def build_temporal_sample(das, t, N_TIMES):
    """3-day stack → (TARGET, TARGET, 24)."""
    frames = []
    for dt in [-1, 0, 1]:
        tt = int(np.clip(t + dt, 0, N_TIMES - 1))
        frames.append(build_sample(das, tt))
    return np.concatenate(frames, axis=-1)


def build_lstm_sequence(das, t, N_TIMES, seq_len=7):
    """7-day sequence → (7, TARGET, TARGET, 8)."""
    return np.stack([
        build_sample(das, int(np.clip(t - seq_len + k, 0, N_TIMES - 1)))
        for k in range(seq_len)
    ], axis=0)


def make_label(mc):
    """Domain-informed 3-class PFZ label."""
    sst, ssh, chl = mc[:,:,0], mc[:,:,1], mc[:,:,4]
    sst_grad, curr_spd, div = mc[:,:,5], mc[:,:,6], mc[:,:,7]

    score = np.zeros_like(sst)
    score += (sst_grad > 0.25).astype(np.float32) * 2.0
    score += (chl < 0.4).astype(np.float32)       * 1.5
    score += ((curr_spd > 0.2) & (curr_spd < 0.7)).astype(np.float32) * 1.0
    score += (div < 0.3).astype(np.float32)        * 1.0
    score += ((ssh < 0.4) & (sst < 0.5)).astype(np.float32) * 1.5

    lbl = np.zeros_like(score, dtype=np.int8)
    lbl[score >= 2.5] = 1
    lbl[score >= 4.0] = 2
    return lbl


def get_channel_stats(das, t):
    """Return per-channel statistics for dashboard display."""
    mc = build_sample(das, t)
    names = ['SST', 'SSH', 'U-Current', 'V-Current', 'Chlorophyll',
             'SST Gradient', 'Current Speed', 'Divergence']
    stats = {}
    for i, name in enumerate(names):
        ch = mc[:,:,i]
        stats[name] = {
            'min':  float(ch.min()),
            'max':  float(ch.max()),
            'mean': float(ch.mean()),
            'std':  float(ch.std())
        }
    return stats


def pfz_to_gps(pred_map, confidence_map, min_confidence=0.7):
    """Convert High-PFZ pixels to GPS coordinates."""
    LAT = np.linspace(LAT_MIN, LAT_MAX, TARGET)
    LON = np.linspace(LON_MIN, LON_MAX, TARGET)

    rows, cols = np.where((pred_map == 2) & (confidence_map >= min_confidence))
    records = []
    for r, c in zip(rows, cols):
        records.append({
            'lat':        float(np.round(LAT[r], 4)),
            'lon':        float(np.round(LON[c], 4)),
            'confidence': float(np.round(confidence_map[r, c], 4))
        })
    records.sort(key=lambda x: -x['confidence'])
    return records
