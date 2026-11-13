"""
foot_geometry.py - shared foot shape, sensor-array grid and colour maps (numpy only, no Qt).
Used by the 3D model, the simulated thermal camera, the demo camera and the heat maps.
"""
import numpy as np

from config import FOOT_LEN, ZONES

KT = [0, .03, .09, .20, .40, .55, .68, .78, .88, .94, .98, 1]
KLAT = [0, 2.8, 3.4, 3.5, 3.9, 4.5, 4.8, 4.4, 3.3, 2.2, 0.9, -1.4]             # lateral edge x (cm)
KMED = [0, -2.6, -3.2, -3.0, -2.4, -3.6, -4.8, -4.6, -4.2, -3.9, -3.2, -1.45]   # medial edge x (cm)
KH = [0, 2.5, 5.0, 8.5, 7.5, 5.5, 3.8, 3.0, 2.4, 1.8, 1.4, 0]                  # height (cm)


def _smooth(a, n=15):
    return np.convolve(np.pad(a, n // 2, mode="edge"), np.ones(n) / n, mode="valid")


_DT = np.linspace(0, 1, 400)
_LAT = _smooth(np.interp(_DT, KT, KLAT))
_MED = _smooth(np.interp(_DT, KT, KMED))


def edges(t):
    """(medial_x, lateral_x) of the outline at normalised length t (0 heel .. 1 toe tip). Right-foot frame."""
    t = np.asarray(t, dtype=float)
    return np.interp(t, _DT, _MED), np.interp(t, _DT, _LAT)


def outline(t):
    med, lat = edges(t)
    return (lat - med) / 2.0, (lat + med) / 2.0          # half width, centre line


# ---- sensor array grid (FSR / NTC arrays are mapped onto a grid like this) -----
GR, GC = 27, 11
GX = np.linspace(-5, 5, GC)
GY = np.linspace(0.3, FOOT_LEN - 0.3, GR)
_hw, _c = outline(GY / FOOT_LEN)
MASK_GRID = (np.abs(GX[None, :] - _c[:, None]) <= _hw[:, None]).astype(np.float32)
_GXX, _GYY = np.meshgrid(GX, GY)
ZG = np.stack([np.exp(-(((_GXX - zx) ** 2 + (_GYY - zy) ** 2) / (2 * 2.3 ** 2))) for _, (zx, zy) in ZONES])


def pressure_grid(p):
    return np.tensordot(np.asarray(p, dtype=float), ZG, 1)


def temp_grid(t, base=31.0, w0=0.05):
    return (np.tensordot(np.asarray(t, dtype=float), ZG, 1) + w0 * base) / (ZG.sum(0) + w0)


# ---- colour map (blue -> cyan -> green -> yellow -> red) ----------------------
CS = [0, .25, .5, .75, 1.0]
CR = [0.10, 0.00, 0.20, 1.00, 0.90]
CG = [0.20, 0.80, 0.80, 0.90, 0.10]
CB = [0.80, 0.90, 0.30, 0.10, 0.10]


def cmap(n):
    n = np.clip(n, 0, 1)
    return np.stack([np.interp(n, CS, CR), np.interp(n, CS, CG), np.interp(n, CS, CB)], axis=-1)
