"""
dataset.py — Data Generator & Preprocessing
AI-Based Geospatial Decision Support System
SRMIST — Harshath Mukundan & Praveen N
"""

import numpy as np
import xarray as xr
import os
import glob
import tensorflow as tf
from keras.utils import to_categorical
from skimage.transform import resize


TARGET = 256


# ─────────────────────────────────────────────────────────────
# Preprocessing utilities
# ─────────────────────────────────────────────────────────────

def clean(arr):
    """Replace fill values and NaNs with channel mean."""
    arr = arr.astype(np.float32)
    arr[arr == -32768] = np.nan
    arr[arr < -1e10]   = np.nan
    m = np.nanmean(arr)
    return np.where(np.isnan(arr), m, arr)


def norm(arr):
    """Min-max normalize to [0, 1]."""
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-8:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def to_target(arr, size=TARGET):
    """Resize to (size, size)."""
    return resize(arr, (size, size), anti_aliasing=True).astype(np.float32)


# ─────────────────────────────────────────────────────────────
# NetCDF loading
# ─────────────────────────────────────────────────────────────

def safe_open(path):
    try:
        return xr.open_dataset(path, engine='netcdf4')
    except Exception:
        return xr.open_dataset(path, engine='h5netcdf')


def load_datasets(data_dir):
    """
    Auto-detect and load 5 CMEMS NetCDF files.
    Returns dict of DataArrays and N_TIMES.
    """
    all_nc = sorted(glob.glob(os.path.join(data_dir, '*.nc')))
    if not all_nc:
        all_nc = sorted(glob.glob(os.path.join(data_dir, '**', '*.nc'), recursive=True))

    def find(pattern):
        matches = [f for f in all_nc if pattern in f]
        if not matches:
            raise FileNotFoundError(f"No file matching '{pattern}' in {data_dir}")
        return matches[0]

    ds_sst = safe_open(find('8369287'))
    ds_vo  = safe_open(find('8416191'))
    ds_zos = safe_open(find('8480927'))
    ds_uo  = safe_open(find('8817739'))
    ds_chl = safe_open(find('obs-oc'))

    def surface(da, dim='depth'):
        return da.isel(**{dim: 0}) if dim in da.dims else da

    thetao = surface(ds_sst['thetao'])
    vo     = surface(ds_vo['vo'])
    uo     = surface(ds_uo['uo'])
    zos    = ds_zos['zos']
    chl    = ds_chl.get('CHL', ds_chl.get('CHL_uncertainty'))

    chl_time_len = len(chl.time) if chl is not None and hasattr(chl, 'time') else 999999
    
    N_TIMES = int(min(len(thetao.time), len(vo.time), len(uo.time),
                  len(zos.time), chl_time_len))

    print(f"Loaded datasets. Common N_TIMES = {N_TIMES}")

    return {
        'sst': thetao, 'vo': vo, 'uo': uo,
        'ssh': zos, 'chl': chl
    }, N_TIMES


# ─────────────────────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────────────────────

def build_sample(das, t):
    """
    Build 8-channel feature map for time index t.
    das: dict from load_datasets()
    Returns: (TARGET, TARGET, 8)
    """
    t = int(np.clip(t, 0, len(das['sst'].time) - 1))

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
    """Concatenate 3 days → (TARGET, TARGET, 24)."""
    frames = []
    for dt in [-1, 0, 1]:
        tt = int(np.clip(t + dt, 0, N_TIMES - 1))
        frames.append(build_sample(das, tt))
    return np.concatenate(frames, axis=-1)


def make_label(mc):
    """Domain-informed 3-class PFZ label generation."""
    sst      = mc[:, :, 0]
    ssh      = mc[:, :, 1]
    chl      = mc[:, :, 4]
    sst_grad = mc[:, :, 5]
    curr_spd = mc[:, :, 6]
    div      = mc[:, :, 7]

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


# ─────────────────────────────────────────────────────────────
# Keras Generators
# ─────────────────────────────────────────────────────────────

class PFZGenerator(tf.keras.utils.Sequence): # type: ignore
    """Generator for U-Net training."""

    def __init__(self, das, indices, N_TIMES, batch_size=4, shuffle=True):
        self.das        = das
        self.indices    = indices.copy()
        self.N_TIMES    = N_TIMES
        self.batch_size = batch_size
        self.shuffle    = shuffle
        if shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return len(self.indices) // self.batch_size

    def __getitem__(self, idx):
        batch = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        X, Y = [], []
        for t in batch:
            x = build_temporal_sample(self.das, t, self.N_TIMES)
            y = make_label(build_sample(self.das, t))
            X.append(x)
            Y.append(to_categorical(y, 3))
        return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


class ConvLSTMGenerator(tf.keras.utils.Sequence): # type: ignore
    """Generator for ConvLSTM training."""

    def __init__(self, das, indices, N_TIMES, seq_len=7, batch_size=4):
        self.das        = das
        self.indices    = np.array([i for i in indices if i >= seq_len and i < N_TIMES - 1])
        self.N_TIMES    = N_TIMES
        self.seq_len    = seq_len
        self.batch_size = batch_size

    def __len__(self):
        return len(self.indices) // self.batch_size

    def __getitem__(self, idx):
        batch = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        X, Y = [], []
        for t in batch:
            seq = np.stack([
                build_sample(self.das, t - self.seq_len + k)
                for k in range(self.seq_len)
            ], axis=0)
            label = make_label(build_sample(self.das, t))
            X.append(seq)
            Y.append(to_categorical(label, 3))
        return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)
