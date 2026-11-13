"""
thermal_camera.py - thermal camera code for diabetic-foot monitoring.

Sources (all return a 2-D numpy array of temperatures in C, left foot in the left half,
right foot in the right half, toes up):
    SimThermalCamera   simulated, follows the shared scenario        (default)
    MLX90640Camera     real MLX90640 (24x32) on the Pi's I2C bus     (UNTESTED skeleton)

analyse_thermal() implements the standard diabetic-foot screening idea: compare each foot with the
SAME place on the other foot. A persistent difference of about 2.2 C or more is a warning sign
of inflammation / pre-ulcer. It is decision support, not a diagnosis.
"""
import math
import time

import numpy as np
import cv2

import config as C
from config import FOOT_LEN, TMIN, TMAX, WATCH_DELTA, HOT_DELTA, CRITICAL_DELTA
from foot_geometry import edges


class SimThermalCamera:
    """Simulated 4:3 thermal frame (60 x 80) covering a 40 x 30 cm mat, toes up."""
    label = "simulated thermal camera"
    ROWS, COLS = 60, 80

    def __init__(self, scenario):
        self.scenario = scenario
        yy, xx = np.mgrid[0:self.ROWS, 0:self.COLS]
        X = ((xx + 0.5) / self.COLS - 0.5) * 40.0         # cm from the centre of the mat
        Y = (yy + 0.5) / self.ROWS * 30.0                 # cm from the top of the mat
        fy = 28.0 - Y                                     # foot length coordinate (toes at the top)
        t = np.clip(fy / FOOT_LEN, 0, 1)
        med, lat = edges(t)
        inside_y = (fy > 0) & (fy < FOOT_LEN)
        self.fy = fy
        self.xl = {"R": X - 6.5, "L": -(X + 6.5)}          # local x: medial negative for both feet
        mask = np.zeros((self.ROWS, self.COLS), np.float32)
        for xl in self.xl.values():
            mask += ((xl >= med) & (xl <= lat) & inside_y).astype(np.float32)
        self.mask = cv2.GaussianBlur(np.clip(mask, 0, 1), (0, 0), 1.0)
        self.t0 = time.time()
        self.rng = np.random.default_rng()

    def read(self):
        t = time.time() - self.t0
        T = 24.0 + 8.0 * self.mask
        T -= 1.5 * np.clip((self.fy - 20.0) / 6.0, 0, 1) * self.mask          # toes slightly cooler
        amp = self.scenario.hot_amp
        if amp > 0:                                                          # hotspot: left first metatarsal
            xl = self.xl["L"]
            hot = amp * np.exp(-(((xl + 1.6) ** 2 + (self.fy - 19.0) ** 2) / (2 * 1.8 ** 2)))
            T += hot * self.mask * (0.95 + 0.05 * math.sin(t))
        T += self.rng.normal(0, 0.12, T.shape)
        return T.astype(np.float32)


class MLX90640Camera:
    """
    REAL HARDWARE skeleton (not tested here):
        pip install adafruit-circuitpython-mlx90640     and enable I2C on the Pi.
    Mount it so both feet are in view, toes up, left foot on the left of the image.
    """
    label = "MLX90640 thermal camera"

    def __init__(self):
        import board, busio, adafruit_mlx90640          # noqa: imported late on purpose
        i2c = busio.I2C(board.SCL, board.SDA, frequency=800000)
        self.mlx = adafruit_mlx90640.MLX90640(i2c)
        self.mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_4_HZ
        self.buf = [0.0] * 768
        self.last = None

    def read(self):
        try:
            self.mlx.getFrame(self.buf)
            self.last = np.array(self.buf, dtype=np.float32).reshape(24, 32)
        except (ValueError, RuntimeError):               # the sensor sometimes drops a frame
            pass
        return self.last


def open_thermal(mode, scenario):
    if mode == "mlx90640":
        return MLX90640Camera()
    return SimThermalCamera(scenario)


def analyse_thermal(T):
    """
    Returns (BGR image, result dict). Result keys:
      found, status, color, asym (max left-right difference, C), side ('Left'/'Right' = warmer foot),
      region, mean_l, mean_r, max_l, max_r
    """
    h = C.CAM_H
    w = max(2, int(round(h * T.shape[1] / T.shape[0])))
    big = cv2.resize(T, (w, h), interpolation=cv2.INTER_CUBIC)
    g = (np.clip((big - TMIN) / (TMAX - TMIN), 0, 1) * 255).astype(np.uint8)
    img = cv2.applyColorMap(g, cv2.COLORMAP_INFERNO)

    half = w // 2
    foot = big > float(np.percentile(big, 5)) + 3.0
    fL, fR = foot[:, :half], foot[:, w - half:][:, ::-1]
    if fL.sum() < 200 or fR.sum() < 200:
        return img, dict(found=False, status="NO FEET", color="#8b949e")

    TL, TR = big[:, :half], big[:, w - half:][:, ::-1]
    both = fL & fR
    diff = np.where(both, TL - TR, 0.0)
    iy, ix = np.unravel_index(np.argmax(np.abs(diff)), diff.shape)
    asym = float(abs(diff[iy, ix]))
    warmer_left = diff[iy, ix] > 0

    hot = np.zeros_like(foot)
    hot[:, :half] = both & (diff > HOT_DELTA)
    hot[:, w - half:] = (both & (-diff > HOT_DELTA))[:, ::-1]

    cs, _ = cv2.findContours(foot.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, cs, -1, (80, 200, 80), 1)
    hcs, _ = cv2.findContours(hot.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, hcs, -1, (255, 255, 0), 2)                 # cyan outline = hotspot
    mx = int(ix) if warmer_left else w - 1 - int(ix)
    cv2.drawMarker(img, (mx, int(iy)), (255, 255, 255), cv2.MARKER_CROSS, 16, 2)
    cv2.line(img, (half, 0), (half, h), (90, 90, 90), 1)
    cv2.putText(img, "L", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, "R", (w - 24, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    wf = fL if warmer_left else fR
    rows = np.where(wf.any(axis=1))[0]
    frac = (iy - rows[0]) / max(1, rows[-1] - rows[0])
    region = "Toes" if frac < 0.2 else "Forefoot" if frac < 0.55 else "Midfoot" if frac < 0.75 else "Heel"

    if asym >= CRITICAL_DELTA:
        status, color = "CRITICAL", "#f85149"
    elif asym >= HOT_DELTA:
        status, color = "WARNING", "#f0883e"
    elif asym >= WATCH_DELTA:
        status, color = "WATCH", "#d29922"
    else:
        status, color = "NORMAL", "#3fb950"

    return img, dict(found=True, status=status, color=color, asym=asym,
                     side="Left" if warmer_left else "Right", region=region,
                     mean_l=float(TL[fL].mean()), mean_r=float(TR[fR].mean()),
                     max_l=float(TL[fL].max()), max_r=float(TR[fR].max()))