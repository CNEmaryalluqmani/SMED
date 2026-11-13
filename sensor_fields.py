"""
sensor_fields.py ("حقول الحساسات") - the on-screen field for each sensor.

    PressurePanel     FSR array   -> heat map per foot + peak value
    PPGPanel          MAX30102    -> column beside the pressure map: live pulse waveform with the 3 values under it
    MatPanel          dashboard column "Weight and UV": scale picture, UV lamp picture and a simple vibration (LRA) reading
    ScaleIcon / UVIcon  the pictures on the Weight and UV tiles (dial scale / UV lamp with safety lock)
    LRAPanel          vibration motor: one simulated value (%), a bar, the state and two test buttons
    SensorsDock       the side panel: ONE tile per sensor (big value, range bar, history line, source badge)

(The sensor MODELS with the simulated values are in sensors.py.)
"""
import math
import random
import time
from collections import deque

import numpy as np
import cv2
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

import config as C
from foot_geometry import MASK_GRID, cmap, pressure_grid, edges
from sensors import build_sensor_tiles
from widgets import ImageView


def small_caption(text=""):
    lbl = QtWidgets.QLabel(text)
    lbl.setAlignment(QtCore.Qt.AlignCenter)
    lbl.setWordWrap(True)
    lbl.setStyleSheet(f"color:#8b949e;font-size:{C.px(12)}px;")
    return lbl


def render_foot_map(grid, mirror, vmin, vmax):
    """Top-down heat map of one foot (toes up). mirror=True for the left foot."""
    m = MASK_GRID
    num = cv2.GaussianBlur((grid * m).astype(np.float32), (0, 0), 1.5)
    den = cv2.GaussianBlur(m, (0, 0), 1.5) + 1e-6
    v = np.where(m > 0, grid, num / den).astype(np.float32)
    v, mm = np.flipud(v), np.flipud(m)
    if mirror:
        v, mm = v[:, ::-1], mm[:, ::-1]
    v = cv2.resize(np.ascontiguousarray(v), (C.MAP_W, C.MAP_H), interpolation=cv2.INTER_CUBIC)
    mm = cv2.resize(np.ascontiguousarray(mm), (C.MAP_W, C.MAP_H), interpolation=cv2.INTER_LINEAR)
    mm = cv2.GaussianBlur(mm, (0, 0), 4) > 0.5            # smooth the stair-stepped sensor-grid edge
    rgb = (cmap((v - vmin) / (vmax - vmin)) * 255).astype(np.uint8)
    rgb[~mm] = (13, 17, 23)
    rgb = np.ascontiguousarray(rgb)
    return QtGui.QPixmap.fromImage(
        QtGui.QImage(rgb.data, C.MAP_W, C.MAP_H, 3 * C.MAP_W, QtGui.QImage.Format_RGB888).copy())


class _FeetMapPanel(QtWidgets.QGroupBox):
    def __init__(self, title):
        super().__init__(title)
        grid = QtWidgets.QGridLayout(self)
        self.maps, self.caps = {}, {}
        for col, side in enumerate("LR"):
            grid.addWidget(small_caption("Left" if side == "L" else "Right"), 0, col)
            lbl = ImageView(70, 140, bg="#0d1117")
            self.maps[side] = lbl
            grid.addWidget(lbl, 1, col)
            self.caps[side] = small_caption("--")
            grid.addWidget(self.caps[side], 2, col)
        grid.setRowStretch(1, 1)


class PressurePanel(_FeetMapPanel):
    def __init__(self):
        super().__init__("Pressure map (FSR array)")

    def update_data(self, data):
        for side in "LR":
            p = data[side]["pressure"]
            self.maps[side].setPixmap(render_foot_map(pressure_grid(p), side == "L", 0, C.PRESSURE_MAX))
            pk = int(np.argmax(p))
            self.caps[side].setText(f"peak {max(p):.0f} kPa\n({C.ZONES[pk][0]})")


class PPGPanel(QtWidgets.QGroupBox):
    """Pulse column: the live waveform (last 6 seconds) on top, the three PPG values directly under it."""
    WINDOW = 6.0

    def __init__(self):
        super().__init__("Pulse (PPG) - live waveform")
        self.buf, self.now = deque(), 0.0
        self.setMinimumWidth(C.px(250))
        v = QtWidgets.QVBoxLayout(self)
        self.plot = pg.PlotWidget()
        self.plot.setBackground("#161b22")
        self.plot.hideAxis("bottom")
        self.plot.hideAxis("left")
        self.plot.setMouseEnabled(False, False)
        self.plot.setYRange(-0.1, 1.5, padding=0)
        self.plot.setXRange(-self.WINDOW, 0, padding=0)
        self.plot.setMinimumHeight(C.px(110))
        self.plot.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)   # its big default hint must not steal space
        self.curve = self.plot.plot(pen=pg.mkPen("#3fb950", width=3))
        self.lbl = {}

        def card(caption, big, color="#3fb950"):
            fr = QtWidgets.QFrame()
            fr.setStyleSheet("QFrame{background:#161b22;border:1px solid #30363d;border-radius:10px;}QLabel{border:0;}")
            c = QtWidgets.QVBoxLayout(fr)
            c.setSpacing(C.px(0))
            cap = QtWidgets.QLabel(caption)
            cap.setStyleSheet(f"color:#8b949e;font-size:{C.px(12)}px;font-weight:bold;")
            cap.setWordWrap(True)
            val = QtWidgets.QLabel("--")
            val.setStyleSheet(f"font-size:{C.px(big)}px;font-weight:bold;color:{color};")
            c.addWidget(cap)
            c.addWidget(val)
            return fr, val

        pulse_card, self.lbl["bpm"] = card("PULSE  (beats per minute)", 40)
        spo2_card, self.lbl["spo2"] = card("OXYGEN (SpO2)", 26)
        pi_card, self.lbl["pi"] = card("PERFUSION INDEX (PI)", 26)
        vals = QtWidgets.QGridLayout()                      # directly under the waveform: pulse, then SpO2 | PI
        vals.setSpacing(C.px(8))
        vals.addWidget(pulse_card, 0, 0, 1, 2)
        vals.addWidget(spo2_card, 1, 0)
        vals.addWidget(pi_card, 1, 1)
        v.addWidget(self.plot, 1)
        v.addLayout(vals)

    def push(self, data):                      # every tick: collect the new samples
        for t, val in data["ppg_samples"]:
            self.buf.append((t, val))
        self.now = data["t"]
        while self.buf and self.buf[0][0] < self.now - self.WINDOW - 1:
            self.buf.popleft()

    def redraw(self):                          # ~15 times per second
        if len(self.buf) > 1:
            arr = np.asarray(self.buf)
            self.curve.setData(arr[:, 0] - self.now, arr[:, 1])

    def update_data(self, data):               # a few times per second
        self.lbl["bpm"].setText(f"{data['bpm']:.0f}")
        self.lbl["spo2"].setText(f"{data['spo2']:.0f} %")
        self.lbl["pi"].setText(f"{data['pi']:.1f} %")


# ----------------------------------------------------------------------------
# Vibration motor: picture + alert cards
# ----------------------------------------------------------------------------
class FootVibrationView(QtWidgets.QWidget):
    """
    A picture of the two feet with the vibration motor under each heel.
    The foot that is being warned shakes, turns orange and sends waves out of its motor.
    Click a foot to test its motor (signal foot_clicked with "L" or "R").
    """
    foot_clicked = QtCore.Signal(str)

    def mousePressEvent(self, e):
        if not self.compact:
            self.foot_clicked.emit("L" if e.position().x() < self.width() / 2 else "R")

    def __init__(self, compact=False):
        super().__init__()
        if not compact:
            self.setCursor(QtCore.Qt.PointingHandCursor)
        self.active, self.level, self.phase, self.sides = False, 0.6, 0.0, {"L", "R"}
        self.compact = compact
        t = np.linspace(0, 1, 50)
        self.med, self.lat = edges(t)
        self.setMinimumSize(C.px(110 if compact else 230), C.px(70 if compact else 100))
        pol = QtWidgets.QSizePolicy.Fixed if compact else QtWidgets.QSizePolicy.Expanding
        self.setSizePolicy(pol, QtWidgets.QSizePolicy.Fixed if compact else QtWidgets.QSizePolicy.Preferred)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._step)
        self.timer.start(40)

    def set_active(self, on, level=0.6, sides=None):
        self.active, self.level = on, level
        if sides:
            self.sides = set(sides)
        self.update()

    def _step(self):
        if self.active:
            self.phase = (self.phase + 0.06) % 1.0
            self.update()

    def paintEvent(self, _):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        w, h = self.width(), self.height()
        s = min(h * 0.84 / C.FOOT_LEN, w * 0.92 / 24.0)             # pixels per cm
        base_y = h * 0.93
        for side, sign in (("L", -1), ("R", 1)):
            lit = self.active and side in self.sides
            dx = dy = 0.0
            if lit:
                shake = 0.35 * s * self.level
                dx, dy = random.uniform(-shake, shake), random.uniform(-shake, shake)
            cx = w / 2 + sign * 6.5 * s + dx
            def pt(x_cm, t):
                return QtCore.QPointF(cx + sign * x_cm * s, base_y - t * C.FOOT_LEN * s + dy)
            ts = np.linspace(0, 1, 50)
            poly = QtGui.QPolygonF([pt(m, t) for m, t in zip(self.med, ts)] +
                                   [pt(l, t) for l, t in zip(self.lat[::-1], ts[::-1])])
            fill = QtGui.QColor("#f0883e") if lit else QtGui.QColor("#21262d")
            fill.setAlpha(120 if lit else 255)
            p.setPen(QtGui.QPen(QtGui.QColor("#f0883e" if lit else "#484f58"), 2.5 if lit else 1.5))
            p.setBrush(fill)
            p.drawPolygon(poly)
            # the motor under the heel
            mx = (self.med[6] + self.lat[6]) / 2
            m = pt(mx, ts[6])
            if lit:
                for i in range(3):                                   # waves leaving the motor
                    k = (self.phase + i / 3.0) % 1.0
                    col = QtGui.QColor("#f0883e")
                    col.setAlpha(int(200 * (1 - k)))
                    p.setPen(QtGui.QPen(col, 2.5))
                    p.setBrush(QtCore.Qt.NoBrush)
                    r = s * (1.3 + 4.2 * k * (0.6 + 0.6 * self.level))
                    p.drawEllipse(m, r, r)
            p.setPen(QtGui.QPen(QtGui.QColor("#ffd2a8" if lit else "#8b949e"), 2))
            p.setBrush(QtGui.QColor("#f0883e" if lit else "#6e7681"))
            p.drawEllipse(m, s * 1.25, s * 1.25)
            if not self.compact:                                     # foot letter
                p.setPen(QtGui.QColor("#e6edf3" if lit else "#6e7681"))
                f_ = p.font()
                f_.setBold(True)
                f_.setPixelSize(max(10, int(s * 3.0)))
                p.setFont(f_)
                p.drawText(QtCore.QRectF(cx - 2.5 * s, base_y - C.FOOT_LEN * s * 0.55, 5 * s, 4 * s),
                           QtCore.Qt.AlignCenter, side)


class AmbientWidget(QtWidgets.QWidget):
    """DHT11 as two small pictures: a thermometer (filled to the temperature) and a droplet (filled to the humidity)."""
    def __init__(self, big=False):
        super().__init__()
        self.big, self.t, self.h = big, None, None
        self.setFixedHeight(C.px(62 if big else 38))
        self.setMinimumWidth(C.px(250 if big else 200))
        self.setToolTip("DHT11: room temperature and humidity")

    def set_values(self, t, h):
        self.t, self.h = t, h
        self.update()

    @staticmethod
    def _heat(t):
        return "#58a6ff" if t < 18 else "#3fb950" if t < 26 else "#f0883e" if t < 32 else "#f85149"

    def paintEvent(self, _):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        H, w = float(self.height()), self.width()
        ttxt = "--" if self.t is None else f"{self.t:.0f}\u00b0C"
        htxt = "--" if self.h is None else f"{self.h:.0f}%"
        px = int(H * 0.52)                                       # shrink the numbers until both pictures + numbers fit
        while True:
            f_ = p.font()
            f_.setBold(True)
            f_.setPixelSize(px)
            fm = QtGui.QFontMetricsF(f_)
            tw, hw = fm.horizontalAdvance(ttxt), fm.horizontalAdvance(htxt)
            if px <= 9 or H * 1.62 + tw + hw + H * 0.35 <= w:
                break
            px -= 1
        p.setFont(f_)
        grey = QtGui.QColor("#6e7681")

        # thermometer
        t = self.t if self.t is not None else 20.0
        col = QtGui.QColor(self._heat(t))
        tx = H * 0.30
        tube_w, top, bottom = H * 0.16, H * 0.08, H * 0.68
        level = min(max((t - 10.0) / 30.0, 0.0), 1.0)
        p.setPen(QtGui.QPen(grey, 1.6))
        p.setBrush(QtCore.Qt.NoBrush)
        p.drawRoundedRect(QtCore.QRectF(tx - tube_w / 2, top, tube_w, bottom - top), tube_w / 2, tube_w / 2)
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(col)
        fill_top = bottom - (bottom - top) * level
        p.drawRoundedRect(QtCore.QRectF(tx - tube_w * 0.28, fill_top, tube_w * 0.56, bottom - fill_top), 2, 2)
        p.drawEllipse(QtCore.QPointF(tx, H * 0.80), H * 0.17, H * 0.17)
        p.setPen(QtGui.QColor("#e6edf3"))
        p.drawText(QtCore.QRectF(H * 0.62, 0, tw + 4, H), QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft, ttxt)

        # droplet
        hx = H * 0.62 + tw + H * 0.35 + H * 0.30
        r = H * 0.25
        yc = H * 0.62
        drop = QtGui.QPainterPath()
        drop.addEllipse(QtCore.QPointF(hx, yc), r, r)
        tri = QtGui.QPainterPath()
        tri.addPolygon(QtGui.QPolygonF([QtCore.QPointF(hx, H * 0.06), QtCore.QPointF(hx - r * 0.87, yc - r * 0.5),
                                        QtCore.QPointF(hx + r * 0.87, yc - r * 0.5)]))
        drop = drop.united(tri)
        hum = min(max((self.h if self.h is not None else 0.0) / 100.0, 0.0), 1.0)
        p.save()
        p.setClipPath(drop)
        d_top, d_bottom = H * 0.06, yc + r
        p.fillRect(QtCore.QRectF(hx - r * 1.2, d_bottom - (d_bottom - d_top) * hum, r * 2.4, H), QtGui.QColor("#58a6ff"))
        p.restore()
        p.setPen(QtGui.QPen(grey, 1.6))
        p.setBrush(QtCore.Qt.NoBrush)
        p.drawPath(drop)
        p.setPen(QtGui.QColor("#e6edf3"))
        p.drawText(QtCore.QRectF(hx + r + H * 0.12, 0, hw + 6, H), QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft, htxt)


# ----------------------------------------------------------------------------
# Vibration motor (LRA): ONE simple panel - a big simulated value, a bar, the state and two test buttons
# ----------------------------------------------------------------------------
LRA_TEXT = {
    "en": dict(title="Vibration motor (LRA)", level="Vibration level", idle="Standby", vib="Buzzing",
               test_l="Test left foot", test_r="Test right foot",
               tip="The motor warns the patient when a danger is found. The value is SIMULATED: it rises while the "
                   "motor buzzes and returns to 0 when it stops."),
    "ar": dict(title="محرك الاهتزاز (LRA)", level="مستوى الاهتزاز", idle="جاهز", vib="يهتز",
               test_l="تجربة القدم اليسرى", test_r="تجربة القدم اليمنى",
               tip="ينبّه المحرك المريض عند اكتشاف خطر. القيمة محاكاة: ترتفع أثناء الاهتزاز وترجع إلى 0 عند التوقف."),
}


class LevelBar(QtWidgets.QWidget):
    """A plain horizontal bar filled to a value 0-100 (green - yellow - orange)."""
    def __init__(self):
        super().__init__()
        self.v = 0.0
        self.setFixedHeight(C.px(14))
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

    def set_value(self, v):
        self.v = min(max(float(v), 0.0), 100.0)
        self.update()

    def paintEvent(self, _):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        w, h = self.width(), self.height()
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(QtGui.QColor("#21262d"))
        p.drawRoundedRect(0, 0, w, h, h / 2, h / 2)
        if self.v > 0:
            col = "#3fb950" if self.v < 40 else "#d29922" if self.v < 70 else "#f0883e"
            p.setBrush(QtGui.QColor(col))
            p.drawRoundedRect(0, 0, max(h, int(w * self.v / 100.0)), h, h / 2, h / 2)


class LRAPanel(QtWidgets.QGroupBox):
    """
    The vibration motor, simple: one big (simulated) vibration level in %, a bar, the state (Standby / Buzzing) and two
    buttons to test the left / right motor. An alert from the risk engine makes the motor buzz by itself.
    The buzz settings are plain attributes: strength, pattern, duration_ms.
    """
    def __init__(self, actuator):
        super().__init__()
        self.actuator = actuator
        self.strength, self.pattern, self.duration_ms = 60, "continuous", 2000    # fixed buzz settings
        self.vibrating, self.last_vib, self.lang = False, 0.0, "en"
        self.sides, self.tel = {"L", "R"}, {}

        self.cap = QtWidgets.QLabel()
        self.cap.setAlignment(QtCore.Qt.AlignCenter)
        self.cap.setStyleSheet(f"color:#8b949e;font-size:{C.px(13)}px;font-weight:bold;")
        self.value = QtWidgets.QLabel("0 %")
        self.value.setAlignment(QtCore.Qt.AlignCenter)
        self.state = QtWidgets.QLabel()
        self.state.setAlignment(QtCore.Qt.AlignCenter)
        self.bar = LevelBar()
        self.btn_l, self.btn_r = QtWidgets.QPushButton(), QtWidgets.QPushButton()
        self.btn_l.clicked.connect(lambda: self.vibrate("manual", "L"))
        self.btn_r.clicked.connect(lambda: self.vibrate("manual", "R"))
        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(self.btn_l)
        buttons.addWidget(self.btn_r)

        v = QtWidgets.QVBoxLayout(self)
        v.addStretch(1)
        v.addWidget(self.cap)
        v.addWidget(self.value)
        v.addWidget(self.bar)
        v.addWidget(self.state)
        v.addStretch(1)
        v.addLayout(buttons)
        self._apply_lang()

        self.poll = QtCore.QTimer(self)                          # watches the motor 25 times per second
        self.poll.timeout.connect(self._poll)
        self.poll.start(40)

    # ---- texts ----
    def _apply_lang(self):
        t = LRA_TEXT[self.lang]
        self.setLayoutDirection(QtCore.Qt.RightToLeft if self.lang == "ar" else QtCore.Qt.LeftToRight)
        self.setTitle(t["title"])
        self.setToolTip(t["tip"])
        self.cap.setText(t["level"])
        self.btn_l.setText(t["test_l"])
        self.btn_r.setText(t["test_r"])
        self._show()

    def set_lang(self, lang):
        self.lang = lang if lang in ("en", "ar") else "en"
        self._apply_lang()

    # ---- the motor ----
    def status(self):
        tel = self.tel or {}
        return dict(vibrating=self.vibrating, intensity=self.strength, sides=self.sides,
                    level=tel.get("level", 0.0), freq=tel.get("freq", 0.0), accel=tel.get("accel", 0.0),
                    current=tel.get("current", 0.0), count=tel.get("count", 0), last=tel.get("last"))

    def vibrate(self, reason, side="?"):
        self.sides = {side} if side in ("L", "R") else {"L", "R"}     # which foot is being warned
        self.actuator.send(self.strength, self.duration_ms, self.pattern)
        self._poll()

    def raise_alert(self, ev):
        """An alert from the risk engine: buzz the motor of that foot (at most once per LRA_COOLDOWN seconds)."""
        side = ev.get("side", "?") if isinstance(ev, dict) else "?"
        if time.time() - self.last_vib > C.LRA_COOLDOWN:
            self.last_vib = time.time()
            self.vibrate("auto", side)

    def _poll(self):
        self.tel = self.actuator.telemetry()
        self.vibrating = self.tel["window"]
        self._show()

    def _show(self):
        t, tel = LRA_TEXT[self.lang], self.tel or {}
        level, on = tel.get("level", 0.0), bool(tel.get("window"))
        col = "#f0883e" if on else "#8b949e"
        self.value.setText(f"{level:.0f} %")
        self.value.setStyleSheet(f"font-size:{C.px(44)}px;font-weight:bold;color:{col};")
        self.bar.set_value(level)
        self.state.setText(t["vib"] if on else t["idle"])
        self.state.setStyleSheet(f"font-size:{C.px(16)}px;font-weight:bold;color:{col};")


BADGE = {"SIMULATED": "#d29922", "LIVE": "#3fb950", "FILE": "#58a6ff", "FROM CAMERA": "#58a6ff"}


class ScaleIcon(QtWidgets.QWidget):
    """A scale: a platform with a dial and a needle (0 .. WEIGHT_MAX kg). Two foot prints appear when a person stands on it."""
    def __init__(self, w=120, h=74):
        super().__init__()
        self.kg = 0.0
        self.setFixedSize(C.px(w), C.px(h))

    def set_value(self, kg):
        self.kg = kg or 0.0
        self.update()

    def paintEvent(self, _):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        w, h = self.width(), self.height()
        present = self.kg >= C.WEIGHT_PRESENT_KG
        f = min(max(self.kg / C.WEIGHT_MAX, 0.0), 1.0)
        accent = QtGui.QColor("#58a6ff" if present else "#6e7681")
        cx, cy, r = w / 2, h * 0.70, min(w * 0.40, h * 0.62)
        rect = QtCore.QRectF(cx - r, cy - r, 2 * r, 2 * r)
        p.setPen(QtGui.QPen(QtGui.QColor("#30363d"), C.px(7), QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
        p.drawArc(rect, 0, 180 * 16)                                       # track
        if f > 0:
            p.setPen(QtGui.QPen(accent, C.px(7), QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
            p.drawArc(rect, 180 * 16, -int(f * 180 * 16))                  # filled part (left to right)
        ang = math.pi * (1.0 - f)
        p.setPen(QtGui.QPen(QtGui.QColor("#e6edf3"), C.px(3), QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
        p.drawLine(QtCore.QPointF(cx, cy), QtCore.QPointF(cx + 0.78 * r * math.cos(ang), cy - 0.78 * r * math.sin(ang)))
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(QtGui.QColor("#e6edf3"))
        p.drawEllipse(QtCore.QPointF(cx, cy), C.px(4), C.px(4))
        plat = QtCore.QRectF(w * 0.06, h * 0.80, w * 0.88, h * 0.16)       # the platform
        p.setPen(QtGui.QPen(QtGui.QColor("#6e7681"), 1))
        p.setBrush(QtGui.QColor("#21262d"))
        p.drawRoundedRect(plat, h * 0.06, h * 0.06)
        if present:                                                        # two foot prints
            p.setPen(QtCore.Qt.NoPen)
            p.setBrush(accent)
            for dx in (-0.16, 0.16):
                p.drawEllipse(QtCore.QPointF(w * (0.5 + dx), plat.center().y()), w * 0.055, h * 0.045)


class UVIcon(QtWidgets.QWidget):
    """
    A UV lamp over the mat. ON: the tube glows purple and rays fall onto the mat. LOCKED: a red padlock.
    WAITING: dim yellow tube. CLEAN: green tick.
    """
    COL = {"locked": "#f85149", "waiting": "#d29922", "disinfecting": "#a371f7", "clean": "#3fb950"}

    def __init__(self, w=120, h=74):
        super().__init__()
        self.state, self.phase = "locked", 0.0
        self.setFixedSize(C.px(w), C.px(h))
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._step)
        self.timer.start(60)

    def set_state(self, state):
        self.state = state
        self.update()

    def _step(self):
        if self.state == "disinfecting":
            self.phase = (self.phase + 0.08) % 1.0
            self.update()

    def paintEvent(self, _):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        w, h = self.width(), self.height()
        col = QtGui.QColor(self.COL.get(self.state, "#8b949e"))
        on = self.state == "disinfecting"
        tube = QtCore.QRectF(w * 0.12, h * 0.10, w * 0.76, h * 0.14)
        if on:                                                             # glow + falling rays
            glow = QtGui.QColor(col)
            glow.setAlpha(70)
            p.setPen(QtCore.Qt.NoPen)
            p.setBrush(glow)
            p.drawRoundedRect(tube.adjusted(-w * 0.04, -h * 0.05, w * 0.04, h * 0.05), h * 0.12, h * 0.12)
            p.setPen(QtGui.QPen(col, C.px(2)))
            for i in range(6):
                x = w * (0.18 + 0.128 * i)
                y0 = h * 0.30 + (self.phase * h * 0.14)
                p.drawLine(QtCore.QPointF(x, y0), QtCore.QPointF(x, min(h * 0.80, y0 + h * 0.18)))
        p.setPen(QtGui.QPen(QtGui.QColor("#6e7681"), 1))
        p.setBrush(col if on else QtGui.QColor("#3d444d"))
        p.drawRoundedRect(tube, h * 0.07, h * 0.07)
        plat = QtCore.QRectF(w * 0.06, h * 0.84, w * 0.88, h * 0.10)       # the mat
        p.setBrush(QtGui.QColor("#21262d"))
        p.drawRoundedRect(plat, h * 0.04, h * 0.04)
        cx, cy = w / 2, h * 0.56
        if self.state == "locked":                                         # padlock
            p.setPen(QtGui.QPen(col, C.px(3)))
            p.setBrush(QtCore.Qt.NoBrush)
            p.drawArc(QtCore.QRectF(cx - w * 0.07, cy - h * 0.26, w * 0.14, h * 0.26), 0, 180 * 16)
            p.setPen(QtCore.Qt.NoPen)
            p.setBrush(col)
            p.drawRoundedRect(QtCore.QRectF(cx - w * 0.11, cy - h * 0.12, w * 0.22, h * 0.24), 3, 3)
        elif self.state == "clean":                                        # tick
            p.setPen(QtGui.QPen(col, C.px(4), QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
            p.drawPolyline([QtCore.QPointF(cx - w * 0.10, cy), QtCore.QPointF(cx - w * 0.02, cy + h * 0.10),
                            QtCore.QPointF(cx + w * 0.12, cy - h * 0.12)])
        elif self.state == "waiting":                                      # clock
            p.setPen(QtGui.QPen(col, C.px(3)))
            p.setBrush(QtCore.Qt.NoBrush)
            p.drawEllipse(QtCore.QPointF(cx, cy), h * 0.15, h * 0.15)
            p.drawLine(QtCore.QPointF(cx, cy), QtCore.QPointF(cx, cy - h * 0.10))
            p.drawLine(QtCore.QPointF(cx, cy), QtCore.QPointF(cx + h * 0.08, cy))


class GaugeBar(QtWidgets.QWidget):
    """Coloured range bar with a marker at the current value."""
    def __init__(self, lo, hi, bands):
        super().__init__()
        self.lo, self.hi, self.bands, self.value = lo, hi, bands, None
        self.setFixedHeight(C.px(26))
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

    def set_value(self, v):
        self.value = v
        self.update()

    def paintEvent(self, _):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        w, bar_h = self.width(), C.px(10)
        y = (self.height() - bar_h) // 2 + C.px(2)
        x0 = 0.0
        for limit, col in self.bands:                           # dim coloured zones
            x1 = (min(max(limit, self.lo), self.hi) - self.lo) / (self.hi - self.lo) * w
            c = QtGui.QColor(col)
            c.setAlpha(110)
            p.fillRect(QtCore.QRectF(x0, y, max(0.0, x1 - x0), bar_h), c)
            x0 = x1
        if self.value is not None:
            x = (min(max(self.value, self.lo), self.hi) - self.lo) / (self.hi - self.lo) * w
            p.setPen(QtGui.QPen(QtGui.QColor("#ffffff"), 3))
            p.drawLine(QtCore.QPointF(x, y - C.px(5)), QtCore.QPointF(x, y + bar_h + C.px(5)))


class Sparkline(QtWidgets.QWidget):
    """The last ~20 seconds of one value."""
    def __init__(self, lo, hi, n=140):
        super().__init__()
        self.lo, self.hi, self.vals = lo, hi, deque(maxlen=n)
        self.setFixedHeight(C.px(34))
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

    def push(self, v):
        self.vals.append(v)
        self.update()

    def paintEvent(self, _):
        if len(self.vals) < 2:
            return
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        w, h = self.width(), self.height() - 4
        n = self.vals.maxlen
        pts = [QtCore.QPointF(w - (len(self.vals) - 1 - i) * w / (n - 1),
                              2 + h - (min(max(v, self.lo), self.hi) - self.lo) / (self.hi - self.lo) * h)
               for i, v in enumerate(self.vals)]
        p.setPen(QtGui.QPen(QtGui.QColor("#58a6ff"), 2))
        p.drawPolyline(pts)


class MatPanel(QtWidgets.QGroupBox):
    """
    Dashboard column for the two mat sensors, one simple card each:
        Weight: scale picture + big kg value + "Person on the mat / Mat empty"
        UV:     lamp picture (padlock / clock / purple rays / tick) + big state + a progress bar while it runs
        LRA:    vibration level in % + state (Standby / Buzzing) + a level bar
    """
    UV_TEXT = {"locked": "LOCKED", "waiting": "WAITING", "disinfecting": "UV ON", "clean": "CLEAN"}
    UV_COL = {"locked": "#f85149", "waiting": "#d29922", "disinfecting": "#a371f7", "clean": "#3fb950"}

    def __init__(self):
        super().__init__("Weight and UV")
        self.setMinimumWidth(C.px(190))
        v = QtWidgets.QVBoxLayout(self)

        def card(caption):
            fr = QtWidgets.QFrame()
            fr.setStyleSheet("QFrame{background:#161b22;border:1px solid #30363d;border-radius:10px;}QLabel{border:0;}")
            lay = QtWidgets.QVBoxLayout(fr)
            lay.setSpacing(C.px(2))
            cap = QtWidgets.QLabel(caption)
            cap.setStyleSheet(f"color:#8b949e;font-size:{C.px(12)}px;font-weight:bold;")
            lay.addWidget(cap)
            return fr, lay

        def centred(text, size, color="#e6edf3", bold=True):
            lb = QtWidgets.QLabel(text)
            lb.setAlignment(QtCore.Qt.AlignCenter)
            lb.setWordWrap(True)
            lb.setStyleSheet(f"font-size:{C.px(size)}px;{'font-weight:bold;' if bold else ''}color:{color};")
            return lb

        wcard, wl = card("WEIGHT  (load cell)")
        self.scale = ScaleIcon(100, 60)
        self.w_val = centred("-- kg", 28)
        self.w_sub = centred("", 13, "#8b949e", False)
        wl.addWidget(self.scale, 0, QtCore.Qt.AlignHCenter)
        wl.addWidget(self.w_val)
        wl.addWidget(self.w_sub)

        ucard, ul = card("UV DISINFECTION  (UV-C)")
        self.lamp = UVIcon(100, 60)
        self.u_val = centred("--", 24)
        self.u_sub = centred("", 13, "#8b949e", False)
        self.u_bar = QtWidgets.QProgressBar()
        self.u_bar.setRange(0, 100)
        self.u_bar.setTextVisible(False)
        self.u_bar.setFixedHeight(C.px(8))
        self.u_bar.setStyleSheet("QProgressBar{background:#21262d;border:0;border-radius:4px;}"
                                 "QProgressBar::chunk{background:#a371f7;border-radius:4px;}")
        ul.addWidget(self.lamp, 0, QtCore.Qt.AlignHCenter)
        ul.addWidget(self.u_val)
        ul.addWidget(self.u_sub)
        ul.addWidget(self.u_bar)

        lcard, ll = card("VIBRATION MOTOR  (LRA)")                  # simple: level in %, state, bar
        row = QtWidgets.QHBoxLayout()
        self.l_val = centred("0 %", 24, "#8b949e")
        self.l_state = centred("Standby", 13, "#8b949e")
        row.addWidget(self.l_val)
        row.addWidget(self.l_state)
        self.l_bar = LevelBar()
        ll.addLayout(row)
        ll.addWidget(self.l_bar)

        v.addWidget(wcard, 3)
        v.addWidget(ucard, 3)
        v.addWidget(lcard, 2)

    def update_data(self, d, lra=None):
        kg = d.get("weight", 0.0)
        present = kg >= C.WEIGHT_PRESENT_KG
        self.scale.set_value(kg)
        self.w_val.setText(f"{kg:.1f} kg")
        self.w_val.setStyleSheet(f"font-size:{C.px(28)}px;font-weight:bold;color:{'#58a6ff' if present else '#8b949e'};")
        self.w_sub.setText("Person on the mat" if present else "Mat empty")

        st = d.get("uv_state", "locked")
        col = self.UV_COL[st]
        self.lamp.set_state(st)
        self.u_val.setText(self.UV_TEXT[st])
        self.u_val.setStyleSheet(f"font-size:{C.px(24)}px;font-weight:bold;color:{col};")
        left = d.get("uv_left", 0.0)
        self.u_sub.setText({"locked": "Lamp OFF - person on the mat",
                            "waiting": f"Mat is empty - starts in {left:.0f} s",
                            "disinfecting": f"Lamp ON - {left:.0f} s left",
                            "clean": "Mat disinfected"}[st])
        self.u_bar.setValue(int(round(100 * d.get("uv_progress", 0.0))))

        lra = lra or {}
        level, on = lra.get("level", 0.0), bool(lra.get("vibrating"))
        lcol = "#f0883e" if on else "#8b949e"
        self.l_val.setText(f"{level:.0f} %")
        self.l_val.setStyleSheet(f"font-size:{C.px(24)}px;font-weight:bold;color:{lcol};")
        self.l_state.setText("Buzzing" if on else "Standby")
        self.l_state.setStyleSheet(f"font-size:{C.px(13)}px;font-weight:bold;color:{lcol};")
        self.l_bar.set_value(level)


class SensorTile(QtWidgets.QFrame):
    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        self.setStyleSheet("SensorTile{background:#161b22;border:1px solid #30363d;border-radius:10px;}")
        v = QtWidgets.QVBoxLayout(self)
        v.setSpacing(C.px(3))
        top = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel(spec["title"])
        title.setStyleSheet(f"color:#58a6ff;font-weight:bold;font-size:{C.px(15)}px;")
        self.badge = QtWidgets.QLabel("--")
        top.addWidget(title)
        top.addStretch(1)
        top.addWidget(self.badge)
        v.addLayout(top)
        mid = QtWidgets.QHBoxLayout()
        self.value = QtWidgets.QLabel("--")
        self.value.setStyleSheet(f"font-size:{C.px(40)}px;font-weight:bold;")
        self.unit = QtWidgets.QLabel(spec["unit"])
        self.unit.setStyleSheet(f"color:#8b949e;font-size:{C.px(18)}px;")
        mid.addWidget(self.value)
        mid.addWidget(self.unit, 0, QtCore.Qt.AlignBottom)
        mid.addStretch(1)
        self.amb = None
        if spec.get("ambient"):                                 # DHT11: thermometer + droplet instead of a big number
            self.amb = AmbientWidget(big=True)
            mid.addWidget(self.amb, 1)
            self.value.hide()
            self.unit.hide()
        self.icon = None
        if spec.get("feet_icon"):                               # the vibration tile: the same two-feet picture, small
            self.icon = FootVibrationView(compact=True)
            mid.addWidget(self.icon)
        self.wicon = self.uvicon = None
        if spec.get("weight_icon"):                             # the weight tile: a scale with a dial
            self.wicon = ScaleIcon()
            mid.addWidget(self.wicon)
        if spec.get("uv_icon"):                                 # the UV tile: a lamp over the mat (with the safety lock)
            self.uvicon = UVIcon()
            mid.addWidget(self.uvicon)
        v.addLayout(mid)
        self.sub = QtWidgets.QLabel("")
        self.sub.setStyleSheet(f"color:#c9d1d9;font-size:{C.px(14)}px;")
        self.sub.setWordWrap(True)
        v.addWidget(self.sub)
        self.gauge = self.spark = None
        if spec.get("lo") is not None:
            self.gauge = GaugeBar(spec["lo"], spec["hi"], spec["bands"])
            v.addWidget(self.gauge)
        if spec.get("spark"):
            self.spark = Sparkline(spec["lo"], spec["hi"])
            v.addWidget(self.spark)

    def update_data(self, ctx, source):
        s = self.spec
        col = BADGE.get(source, "#8b949e")
        self.badge.setText(source)
        self.badge.setStyleSheet(f"background:{col};color:#000;font-weight:bold;border-radius:4px;"
                                 f"padding:1px 8px;font-size:{C.px(11)}px;")
        if self.amb is not None:
            self.amb.set_values(ctx["data"]["dht_t"], ctx["data"]["dht_h"])
        if self.icon is not None:
            lra = ctx["lra"]
            self.icon.set_active(lra["vibrating"], lra["intensity"] / 100.0, lra.get("sides"))
        if self.wicon is not None:
            self.wicon.set_value(ctx["data"].get("weight", 0.0))
        if self.uvicon is not None:
            self.uvicon.set_state(ctx["data"].get("uv_state", "locked"))
        num = s["get"](ctx)
        text = s["text"](ctx) if s.get("text") else (s["fmt"].format(num) if num is not None else "--")
        colour = s["vcolor"](ctx) if s.get("vcolor") else "#e6edf3"
        self.value.setText(text if text is not None else "--")
        self.value.setStyleSheet(f"font-size:{C.px(40)}px;font-weight:bold;color:{colour or '#e6edf3'};")
        sub = s["sub"](ctx) if s.get("sub") else ""
        self.sub.setText(sub or "no reading")
        if self.gauge is not None:
            self.gauge.set_value(num)
        if self.spark is not None and num is not None:
            self.spark.push(num)


class SensorsDock(QtWidgets.QDockWidget):
    """
    The independent Sensors panel on the side of the window (hide / show with F3):
    ONE tile per sensor - big value, range bar, history line and where the value comes from.
    """
    def __init__(self):
        super().__init__("Sensors")
        self.setObjectName("sensors_dock")
        self.tiles = {s["key"]: SensorTile(s) for s in build_sensor_tiles()}
        inner = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(inner)
        for tile in self.tiles.values():
            v.addWidget(tile)
        v.addStretch(1)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll.setWidget(inner)
        self.setWidget(scroll)
        self.inner = inner
        self.setMinimumWidth(C.px(300))

    def fit_width(self):
        """Make the panel at least as wide as its widest tile (+ scroll bar): no value can be cut off."""
        need = self.inner.minimumSizeHint().width() + C.px(30)
        self.setMinimumWidth(max(C.px(300), need))
        return self.minimumWidth()

    def update_data(self, ctx, sources):
        for key, tile in self.tiles.items():
            tile.update_data(ctx, sources.get(key, "SIMULATED"))