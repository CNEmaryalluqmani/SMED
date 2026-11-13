"""
sensors.py - models of every sensor (simulated values for now).

Each sensor is its own small class with a read() method, so later you replace ONE class
with real Arduino data and nothing else changes:

    PressureSensors     FSR array under each foot            -> 6 zones per foot (kPa)
    WeightSensor        load cell + HX711 amplifier          -> body weight (kg), also tells if a person is on the mat
    PPGSensor           MAX30102 optical pulse sensor        -> waveform, pulse rate, SpO2, perfusion index
    DHT11Sensor         ambient temperature / humidity       -> integers, updates once per second
    LRAActuator         vibration motor (output, not sensor)
    UVActuator          UV-C disinfection lamp (output) with a SAFETY LOCK: never on while a person is on the mat

SensorHub.read() returns ONE dict used by the whole dashboard:
    {"fsr": float (kPa),                                            <- ONE value per sensor
     "L": {"pressure": [6]}, "R": {...},                            <- simulated array view used by the map / 3D
     "weight": float (kg), "uv_state": "locked|waiting|disinfecting|clean", "uv_on": bool,
     "uv_progress": 0..1, "uv_left": seconds,
     "ppg": float (latest sample), "ppg_samples": [(t, value), ...] (new samples since the last read, 50 per second),
     "bpm": float, "spo2": float, "pi": float, "dht_t": float, "dht_h": float, "t": seconds}
"""
import math
import os
import time

import numpy as np

import config as C

SCENARIOS = {
    #            hot_amp (C)  overload  wound
    "normal":   (0.0, False, False),
    "hotspot":  (3.0, False, False),
    "overload": (0.0, True, False),
    "ulcer":    (4.4, True, True),
}


class SimScenario:
    """One switch shared by every simulated source, so all of them tell the same story."""
    def __init__(self, name="hotspot"):
        self.person = True             # simulated: is somebody standing on the mat?
        self.set(name)

    def set_person(self, on):
        self.person = bool(on)

    def set(self, name):
        self.name = name
        self.hot_amp, self.overload, self.wound = SCENARIOS[name]

    @property
    def red(self):                     # visible redness on the left foot
        return self.hot_amp > 0


class PressureSensors:
    """FSR array. Simulates a person standing and shifting weight between the feet."""
    BASE = {"L": [145, 58, 150, 125, 95, 105], "R": [150, 60, 145, 130, 100, 110]}
    OFFS = [0.0, 0.8, 1.6, 1.8, 1.8, 2.4]

    def __init__(self, rng, scenario):
        self.rng, self.scenario = rng, scenario
        self.k = 1.0                                              # 1 = person on the mat, 0 = empty (changes smoothly)

    def read(self, t):
        self.k += ((1.0 if self.scenario.person else 0.0) - self.k) * 0.08
        if self.k < 0.002:
            self.k = 0.0
        share = {"L": 0.5 + 0.12 * math.sin(t / 4), "R": 0.5 - 0.12 * math.sin(t / 4)}
        out = {}
        for side in "LR":
            vals = []
            for i, (base, off) in enumerate(zip(self.BASE[side], self.OFFS)):
                if side == "L" and i == 2 and self.scenario.overload:
                    base *= 2.6                                   # overloaded first metatarsal
                load = share[side] * 2 * (0.85 + 0.15 * math.sin(1.6 * t - off))
                vals.append(max(0.0, base * load * self.k * (1 + self.rng.normal(0, 0.02))))
            out[side] = vals
        return out


class WeightSensor:
    """
    Load cell + HX711 amplifier under the mat: body weight in kg.
    REAL HARDWARE: read the HX711 on the Arduino, calibrate with a known weight, send kg to the PC.
    """
    def __init__(self, rng, scenario):
        self.rng, self.scenario, self.kg = rng, scenario, 0.0

    def read(self, t):
        target = 0.0
        if self.scenario.person:
            target = C.SIM_PERSON_KG + 0.6 * math.sin(1.3 * t) + self.rng.normal(0, 0.15)   # small sway
        self.kg += (target - self.kg) * 0.08                       # the person steps on / off smoothly
        if self.kg < 0.3:
            self.kg = 0.0
        return float(self.kg)


class UVActuator:
    """
    UV-C disinfection lamp (an OUTPUT). SAFETY LOCK: the lamp can only be on while the mat is EMPTY (weight below
    WEIGHT_PRESENT_KG), because UV-C harms skin and eyes.
        locked        a person is on the mat -> lamp OFF
        waiting       mat just became empty -> wait UV_START_DELAY seconds
        disinfecting  lamp ON for UV_CYCLE_SECONDS
        clean         cycle finished (stays so until the next person steps on)
    REAL HARDWARE: switch a relay from the Arduino (e.g. ser.write(b"UV,1\\n")) ONLY when uv_on is True, and also
    wire a hardware interlock (door / pressure switch) so the lamp can never be on if the software fails.
    """
    def __init__(self):
        self.state, self.t_empty, self.t_start = "locked", None, None

    def update(self, kg):
        now = time.monotonic()
        if kg >= C.WEIGHT_PRESENT_KG:
            self.state, self.t_empty, self.t_start = "locked", None, None
        else:
            if self.state == "locked":
                self.state, self.t_empty = "waiting", now
            if self.state == "waiting" and now - self.t_empty >= C.UV_START_DELAY:
                self.state, self.t_start = "disinfecting", now
            if self.state == "disinfecting" and now - self.t_start >= C.UV_CYCLE_SECONDS:
                self.state = "clean"
        progress, left = 0.0, 0.0
        if self.state == "disinfecting":
            e = now - self.t_start
            progress, left = min(1.0, e / C.UV_CYCLE_SECONDS), max(0.0, C.UV_CYCLE_SECONDS - e)
        elif self.state == "waiting":
            left = max(0.0, C.UV_START_DELAY - (now - self.t_empty))
        elif self.state == "clean":
            progress = 1.0
        return dict(uv_state=self.state, uv_on=self.state == "disinfecting", uv_progress=progress, uv_left=left)


class PPGSensor:
    """
    MAX30102 optical PPG: waveform + pulse rate + SpO2 + perfusion index.
    The waveform is produced at a fixed 50 samples per second (like the real sensor), so it looks the same
    however slowly the window redraws.
    """
    RATE = 50.0

    def __init__(self, rng):
        self.rng, self.phase, self.acc = rng, 0.0, 0.0

    def read(self, t, dt):
        bpm = 77 + 3 * math.sin(t / 15)
        self.acc += min(dt, 1.0) * self.RATE
        n = int(self.acc)
        self.acc -= n
        samples = []
        for k in range(n):
            self.phase = (self.phase + bpm / 60.0 / self.RATE) % 1.0
            ph = self.phase
            v = (math.exp(-((ph - 0.18) / 0.07) ** 2) + 0.45 * math.exp(-((ph - 0.45) / 0.10) ** 2)
                 + self.rng.normal(0, 0.01))
            samples.append((t - (n - 1 - k) / self.RATE, v))
        if samples:
            self.last = samples[-1][1]
        return dict(ppg=getattr(self, "last", 0.0), ppg_samples=samples, bpm=bpm,
                    spo2=97 + 0.6 * math.sin(t / 9), pi=2.2 + 0.4 * math.sin(t / 10))


class DHT11Sensor:
    """DHT11: 1 C and 1 % RH resolution, new value only once per second (like the real part)."""
    def __init__(self, rng):
        self.rng, self.last, self.t, self.h = rng, -9.0, 26.0, 45.0

    def read(self, t):
        if t - self.last >= 1.0:
            self.last = t
            self.t = float(round(26.5 + 0.3 * math.sin(t / 30) + self.rng.normal(0, 0.2)))
            self.h = float(round(45 + 3 * math.sin(t / 25)))
        return dict(dht_t=self.t, dht_h=self.h)


class LRAActuator:
    """
    Vibration motor (LRA = linear resonant actuator, the kind that makes a phone buzz). An OUTPUT, not a sensor.
    send(intensity, ms, pattern) starts a buzz; telemetry() returns what a real motor would show while it runs
    (simulated: resonant frequency ~175 Hz, acceleration up to ~1.5 g, current up to ~80 mA).

    REAL HARDWARE: also send the command to the Arduino, e.g.  ser.write(f"LRA,{intensity},{ms},{pattern}\\n".encode())
    (an LRA is normally driven through a DRV2605L haptic driver over I2C), and fill telemetry() from your measurements.
    """
    FREQ = 175.0
    PATTERNS = ("continuous", "pulses", "double")

    def __init__(self):
        self.rng = np.random.default_rng()
        self.t_start, self.t_end = None, 0.0
        self.strength, self.pattern, self.count, self.last_time = 0, "continuous", 0, None
        self.level = 0.0                                                # simulated reading: vibration level 0-100 %

    def send(self, intensity, ms, pattern="continuous"):
        now = time.monotonic()
        self.t_start, self.t_end = now, now + ms / 1000.0
        self.strength, self.pattern = intensity, pattern
        self.count += 1
        self.last_time = time.strftime("%H:%M:%S")

    def telemetry(self):
        now = time.monotonic()
        window = self.t_start is not None and now < self.t_end          # a buzz command is running
        on = False
        if window:
            e = now - self.t_start
            if self.pattern == "continuous":
                on = True
            elif self.pattern == "pulses":
                on = (e % 0.30) < 0.15
            else:                                                       # double tap
                c = e % 1.0
                on = c < 0.12 or 0.22 <= c < 0.34
        k = self.strength / 100.0
        r = self.rng
        # SIMULATED SENSOR VALUE: vibration level in %. It rises smoothly while the motor buzzes and falls back to 0
        # when it stops (like a real motor spinning up / down). REAL HARDWARE: replace with the measured value.
        target = self.strength * (1 + r.normal(0, 0.02)) if on else 0.0
        self.level += (target - self.level) * 0.35
        if self.level < 0.5:
            self.level = 0.0
        level = float(min(100.0, max(0.0, self.level)))
        if on:
            return dict(on=True, window=True, level=level, freq=self.FREQ + r.normal(0, 0.8),
                        accel=1.5 * k * (1 + r.normal(0, 0.03)),
                        current=15 + 65 * k * (1 + r.normal(0, 0.03)), count=self.count, last=self.last_time)
        return dict(on=False, window=window, level=level, freq=0.0, accel=0.0, current=0.0,
                    count=self.count, last=self.last_time)


class SensorHub:
    """Collects every sensor into the single dict the dashboard uses."""
    def __init__(self, scenario=None):
        self.scenario = scenario or SimScenario()
        rng = np.random.default_rng()
        self.pressure = PressureSensors(rng, self.scenario)
        self.weight = WeightSensor(rng, self.scenario)
        self.uv = UVActuator()
        self.ppg = PPGSensor(rng)
        self.dht = DHT11Sensor(rng)
        # where each value comes from. When you connect the real Arduino change these to "LIVE".
        self.sources = {"fsr": "SIMULATED", "weight": "SIMULATED", "uv": "SIMULATED", "ppg": "SIMULATED",
                        "dht": "SIMULATED", "lra": "SIMULATED"}
        self.t0 = self.last = time.time()

    def read(self):
        now = time.time()
        dt, self.last = now - self.last, now
        t = now - self.t0
        p = self.pressure.read(t)
        out = {side: dict(pressure=p[side]) for side in "LR"}
        out["t"] = t
        # ONE reading per sensor. Simulated here as the monitored (left) foot:
        out["fsr"] = max(p["L"])                 # pressure under the foot, kPa      <- put your real FSR value here
        out["weight"] = self.weight.read(t)         # body weight, kg  <- put your real load-cell value here
        out.update(self.uv.update(out["weight"]))    # UV lamp state (safety lock by weight)
        out.update(self.ppg.read(t, dt))
        out.update(self.dht.read(t))
        return out


# ----------------------------------------------------------------------------
# One tile per sensor for the side panel. ONE reading per sensor.
# ctx = {"data": SensorHub.read() dict, "thermal": thermal result or None, "ai": AI result or None,
#        "camera": {"ok": bool, "spec": str, "size": (w, h)}, "lra": {"vibrating": bool, "intensity": int}}
# To connect a real sensor: make SensorHub.read() return the real value under the same key (data["fsr"],
# data["bpm"], data["dht_t"] ...) and set the badge to LIVE in SensorHub.sources - the tile follows automatically.
# ----------------------------------------------------------------------------
GREEN, YELLOW, ORANGE, RED, BLUE = "#3fb950", "#d29922", "#f0883e", "#f85149", "#58a6ff"


def _safe(fn):
    def g(ctx):
        try:
            return fn(ctx)
        except Exception:
            return None
    return g


def build_sensor_tiles():
    from config import PRESSURE_ALERT, PRESSURE_MAX, WATCH_DELTA, HOT_DELTA, CRITICAL_DELTA, WEIGHT_MAX, WEIGHT_PRESENT_KG
    PURPLE = "#a371f7"
    uv_text = {"locked": "LOCKED", "waiting": "WAITING", "disinfecting": "UV ON", "clean": "CLEAN"}
    uv_col = {"locked": RED, "waiting": YELLOW, "disinfecting": PURPLE, "clean": GREEN}

    def uv_sub(c):
        d = c["data"]
        return {"locked": "Safety lock: person on the mat, lamp OFF",
                "waiting": "Mat is empty - lamp starts in {:.0f} s".format(d["uv_left"]),
                "disinfecting": "Lamp ON - mat empty, {:.0f} s left".format(d["uv_left"]),
                "clean": "Cycle finished - mat disinfected"}[d["uv_state"]]

    return [
        dict(key="fsr", title="Pressure (FSR)", unit="kPa", fmt="{:.0f}", spark=True,
             get=_safe(lambda c: c["data"]["fsr"]), lo=0, hi=PRESSURE_MAX,
             bands=[(PRESSURE_ALERT * 0.75, GREEN), (PRESSURE_ALERT, YELLOW), (PRESSURE_MAX, RED)],
             sub=_safe(lambda c: "force under the foot")),
        dict(key="weight", title="Weight (load cell HX711)", unit="kg", fmt="{:.1f}", spark=True, weight_icon=True,
             get=_safe(lambda c: c["data"]["weight"]), lo=0, hi=WEIGHT_MAX,
             bands=[(WEIGHT_PRESENT_KG, "#6e7681"), (WEIGHT_MAX, BLUE)],
             sub=_safe(lambda c: "Person on the mat" if c["data"]["weight"] >= WEIGHT_PRESENT_KG else "Mat empty")),
        dict(key="ppg", title="Pulse (MAX30102 PPG)", unit="BPM", fmt="{:.0f}", spark=True,
             get=_safe(lambda c: c["data"]["bpm"]), lo=40, hi=140,
             bands=[(55, YELLOW), (100, GREEN), (140, YELLOW)],
             sub=_safe(lambda c: "SpO2 {:.0f} %    Perfusion {:.1f} %".format(c["data"]["spo2"], c["data"]["pi"]))),
        dict(key="dht", title="Ambient (DHT11)", unit="", fmt=None, spark=False, ambient=True,
             get=lambda c: None, text=_safe(lambda c: "{:.0f} C".format(c["data"]["dht_t"])),
             sub=_safe(lambda c: "room temperature and humidity")),
        dict(key="thermal", title="Thermal camera (left - right)", unit="\u00b0C", fmt="{:.1f}", spark=True,
             get=_safe(lambda c: c["thermal"]["asym"]), lo=0, hi=6,
             bands=[(WATCH_DELTA, GREEN), (HOT_DELTA, YELLOW), (CRITICAL_DELTA, ORANGE), (6, RED)],
             sub=_safe(lambda c: "Warmer: {} foot ({})".format(c["thermal"]["side"], c["thermal"]["region"]))),
        dict(key="camera", title="Camera (AI analysis)", unit="", fmt=None, spark=False,
             get=lambda c: None,
             text=_safe(lambda c: c["ai"]["status"] if c["camera"]["ok"] else "no picture"),
             vcolor=_safe(lambda c: c["ai"]["color"]),
             sub=_safe(lambda c: "Red {:.1f} %   Dark {:.1f} %  ({}x{} px)".format(
                 max(f["red_pct"] for f in c["ai"]["feet"]), max(f["dark_pct"] for f in c["ai"]["feet"]),
                 *c["camera"]["size"]))),
        dict(key="lra", title="Vibration motor (LRA)", unit="%", fmt="{:.0f}", spark=False, feet_icon=True,
             get=_safe(lambda c: c["lra"]["level"]), lo=0, hi=100,
             bands=[(40, GREEN), (70, YELLOW), (100, ORANGE)],
             sub=_safe(lambda c: "Buzzing" if c["lra"]["vibrating"] else "Standby")),
        dict(key="uv", title="UV disinfection (UV-C)", unit="", fmt=None, spark=False, uv_icon=True,
             get=_safe(lambda c: c["data"]["uv_progress"] * 100.0), lo=0, hi=100, bands=[(100, PURPLE)],
             text=_safe(lambda c: uv_text[c["data"]["uv_state"]]),
             vcolor=_safe(lambda c: uv_col[c["data"]["uv_state"]]),
             sub=_safe(uv_sub)),
    ]