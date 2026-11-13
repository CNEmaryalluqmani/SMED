"""
camera.py - camera input + the camera panel with two slots:
      [ Thermal analysis ]   [ AI analysis ]

Visible-light camera source (--camera):
    demo                           synthetic feet that follow the simulation scenario   (default)
    serial:/dev/ttyUSB0[@BAUD]     ESP32-CAM over USB (esp32_cam_usb_stream.ino), default 921600
    http://host:port/stream        MJPEG stream (ESP32-CAM WiFi example uses :81/stream)
    webcam:0                       any normal USB webcam
    file:foot.jpg                  an image or a video file
Thermal source (--thermal): sim (default) | mlx90640 | pseudo
    pseudo = no thermal camera: show the visible camera's REDNESS map instead (not degrees).
"""
import math
import os
import struct
import threading
import time

import numpy as np
import cv2
from PySide6 import QtCore, QtGui, QtWidgets

import config as C
from foot_geometry import edges
from thermal_camera import open_thermal, analyse_thermal
from ai_analysis import FootAI
from widgets import ImageView

HDR = b"\xAA\x55\xAA\x55"
MAX_JPEG = 200_000


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def bgr_to_pixmap(img):
    rgb = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    h, w, _ = rgb.shape
    return QtGui.QPixmap.fromImage(QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy())


def fit_to(img, w, h):
    """Resize keeping aspect ratio and pad with black to exactly w x h."""
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    r = cv2.resize(img, (max(1, int(iw * s)), max(1, int(ih * s))),
                   interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    out = np.zeros((h, w, 3), np.uint8)
    y, x = (h - r.shape[0]) // 2, (w - r.shape[1]) // 2
    out[y:y + r.shape[0], x:x + r.shape[1]] = r
    return out


# ----------------------------------------------------------------------------
# camera sources
# ----------------------------------------------------------------------------
def pull_frame(buf):
    """Extract one JPEG from a bytearray (consumes it). Returns bytes or None."""
    while True:
        i = buf.find(HDR)
        if i < 0:
            del buf[:max(0, len(buf) - 3)]
            return None
        if i > 0:
            del buf[:i]
        if len(buf) < 8:
            return None
        n = struct.unpack("<I", bytes(buf[4:8]))[0]
        if n == 0 or n > MAX_JPEG:
            del buf[:1]
            continue
        if len(buf) < 8 + n:
            return None
        jpg = bytes(buf[8:8 + n])
        if jpg[:2] != b"\xff\xd8":
            del buf[:1]
            continue
        del buf[:8 + n]
        return jpg


class _Threaded:
    def __init__(self):
        self.frame, self.err, self._run = None, None, True

    def start(self):
        threading.Thread(target=self._loop, daemon=True).start()

    def read(self):
        f = self.frame
        return None if f is None else f.copy()

    def close(self):
        self._run = False


class SerialCamera(_Threaded):
    def __init__(self, port, baud=921600):
        super().__init__()
        self.port, self.baud = port, baud
        self.start()

    def _loop(self):
        import serial                                      # pip install pyserial
        while self._run:
            try:
                with serial.Serial(self.port, self.baud, timeout=0.2) as ser:
                    buf = bytearray()
                    while self._run:
                        data = ser.read(ser.in_waiting or 1)
                        if data:
                            buf += data
                        while True:
                            jpg = pull_frame(buf)
                            if jpg is None:
                                break
                            img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                            if img is not None:
                                self.frame = img
            except Exception as e:                         # port missing / unplugged -> retry
                self.err = str(e)
                time.sleep(1.0)


class CvCamera(_Threaded):
    """HTTP MJPEG stream, webcam index, or video file."""
    def __init__(self, src, is_file=False):
        super().__init__()
        self.src, self.is_file = src, is_file
        self.start()

    def _loop(self):
        while self._run:
            cap = cv2.VideoCapture(self.src)
            if not cap.isOpened():
                self.err = f"cannot open {self.src}"
                time.sleep(1.0)
                continue
            delay = 1.0 / (cap.get(cv2.CAP_PROP_FPS) or 25.0)      # play video files at their own speed
            while self._run:
                ok, f = cap.read()
                if not ok:
                    break
                self.frame = f
                if self.is_file:
                    time.sleep(delay)
            cap.release()


class StaticImage:
    def __init__(self, img):
        self.img, self.err = img, None

    def read(self):
        return self.img.copy()

    def close(self):
        pass


class DemoFootCamera:
    """Synthetic top-down view of both feet (toes up). Follows the shared simulation scenario."""
    W, H, PPC = 640, 480, 14.2                             # pixels, pixels, pixels per cm
    SKIN = (140, 175, 215)

    def __init__(self, scenario):
        self.scenario, self.err = scenario, None
        self.rng = np.random.default_rng()
        self.t0 = time.time()
        t = np.linspace(0, 1, 60)
        self.med, self.lat = edges(t)
        self.fy = t * C.FOOT_LEN

    def _px(self, side, x_local, fy):
        X = (6.5 + x_local) if side == "R" else (-6.5 - x_local)
        return self.W / 2 + X * self.PPC, (self.H - 55) - fy * self.PPC

    def read(self, t=None):
        img = np.full((self.H, self.W, 3), (120, 120, 125), np.uint8)
        for side in "LR":
            pts = [self._px(side, m, f) for m, f in zip(self.med, self.fy)]
            pts += [self._px(side, l, f) for l, f in zip(self.lat[::-1], self.fy[::-1])]
            cv2.fillPoly(img, [np.array(pts, np.int32)], self.SKIN)
        sc = self.scenario
        if sc.red:                                        # inflamed area, left first metatarsal
            t = (time.time() - self.t0) if t is None else t
            x, y = self._px("L", -1.6, 19.0)
            layer = img.copy()
            cv2.circle(layer, (int(x), int(y)), 24, (90, 90, 215), -1)
            a = 0.85 + 0.15 * math.sin(t)
            img = cv2.addWeighted(layer, a, img, 1 - a, 0)
            if sc.wound:
                cv2.circle(img, (int(x), int(y)), 14, (40, 50, 70), -1)
        img = cv2.GaussianBlur(img, (7, 7), 0)
        noise = self.rng.normal(0, 2.0, img.shape)
        return np.clip(img + noise, 0, 255).astype(np.uint8)

    def close(self):
        pass


def open_camera(spec, scenario):
    spec = spec or "demo"
    if spec == "demo":
        return DemoFootCamera(scenario)
    if spec.startswith(("http://", "https://", "rtsp://")):
        return CvCamera(spec)
    kind, _, rest = spec.partition(":")
    if kind == "serial":
        port, _, baud = rest.partition("@")
        return SerialCamera(port, int(baud or 921600))
    if kind == "webcam":
        return CvCamera(int(rest or 0))
    if kind == "file":
        img = cv2.imread(rest)
        return StaticImage(img) if img is not None else CvCamera(rest, is_file=True)
    raise ValueError(f"unknown camera spec: {spec}")


# ----------------------------------------------------------------------------
# the panel: source selector + two slots
# ----------------------------------------------------------------------------
SAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")

# label, kind, value
SOURCES = [
    ("Demo (synthetic feet)",          "demo",   "demo"),
    ("Sample video (feet)",            "spec",   "file:" + os.path.join(SAMPLES_DIR, "demo_feet.mp4")),
    ("Sample image - normal",          "spec",   "file:" + os.path.join(SAMPLES_DIR, "feet_normal.jpg")),
    ("Sample image - inflamed",        "spec",   "file:" + os.path.join(SAMPLES_DIR, "feet_inflamed.jpg")),
    ("Sample image - wound",           "spec",   "file:" + os.path.join(SAMPLES_DIR, "feet_wound.jpg")),
    ("My image file...",               "image",  ""),
    ("My video file...",               "video",  ""),
    ("Live - USB camera (ESP32-CAM)",  "serial", "/dev/ttyUSB0"),
    ("Live - webcam",                  "webcam", "0"),
    ("Live - network stream",          "http",   "http://192.168.1.50:8000/stream"),
]
THERMAL_MODES = ["sim", "pseudo", "mlx90640"]
THERMAL_LABELS = ["Simulated thermal camera", "Redness map from the camera", "MLX90640 thermal camera"]


class CameraPanel(QtWidgets.QWidget):
    def __init__(self, scenario, camera_spec="demo", thermal_mode="sim", model_path=None):
        super().__init__()
        self.scenario = scenario
        self.ai = FootAI(model_path)
        self.thermal_res, self.ai_res = None, None
        self.rgb, self.thermal, self.thermal_mode, self.camera_spec = None, None, "sim", "demo"
        self.thermal_msg = ""

        # ---- controls row
        self.src_combo = QtWidgets.QComboBox()
        self.src_combo.addItems([s[0] for s in SOURCES])
        self.src_edit = QtWidgets.QLineEdit()
        self.src_btn = QtWidgets.QPushButton("Connect")
        self.th_combo = QtWidgets.QComboBox()
        self.th_combo.addItems(THERMAL_LABELS)
        self.hint = QtWidgets.QLabel("Tip: with an image / video / live camera, choose 'Redness map from the camera' for the thermal slot.")
        self.hint.setStyleSheet(f"color:#8b949e;font-size:{C.px(12)}px;")
        self.hint.setWordWrap(True)
        ctl = QtWidgets.QHBoxLayout()
        ctl.addWidget(QtWidgets.QLabel("Camera source:"))
        ctl.addWidget(self.src_combo)
        ctl.addWidget(self.src_edit, 1)
        ctl.addWidget(self.src_btn)
        ctl.addSpacing(14)
        ctl.addWidget(QtWidgets.QLabel("Thermal slot:"))
        ctl.addWidget(self.th_combo)

        # ---- two slots
        self.thermal_img, self.ai_img = ImageView(280, 210), ImageView(280, 210)      # both 4:3, same size
        self.last_size = None
        self.thermal_txt, self.ai_txt = QtWidgets.QLabel("--"), QtWidgets.QLabel("--")
        for lbl in (self.thermal_txt, self.ai_txt):
            lbl.setTextFormat(QtCore.Qt.RichText)
            lbl.setWordWrap(True)
            lbl.setMinimumHeight(C.px(100))
            lbl.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
            lbl.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        slots = QtWidgets.QHBoxLayout()
        self.g_thermal = QtWidgets.QGroupBox()
        self.g_ai = QtWidgets.QGroupBox("AI analysis (camera)")
        for g, img, txt in ((self.g_thermal, self.thermal_img, self.thermal_txt),
                            (self.g_ai, self.ai_img, self.ai_txt)):
            v = QtWidgets.QVBoxLayout(g)
            v.addWidget(img, 1)
            v.addWidget(txt)
            slots.addWidget(g, 1)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(ctl)
        lay.addWidget(self.hint)
        lay.addLayout(slots, 1)

        # ---- initial state from the command line, then connect the signals
        self.set_source(camera_spec or "demo")
        self._select_for_spec(camera_spec or "demo")
        self.set_thermal(thermal_mode)
        self.th_combo.setCurrentIndex(THERMAL_MODES.index(self.thermal_mode))
        self.src_combo.currentIndexChanged.connect(self._src_changed)
        self.src_btn.clicked.connect(self._src_button)
        self.src_edit.returnPressed.connect(self._src_button)
        self.th_combo.currentIndexChanged.connect(lambda i: self.set_thermal(THERMAL_MODES[i]))

    # ---------------- source selection ----------------
    def set_source(self, spec):
        """Switch the visible-light camera to demo / image / video / live. Safe to call any time."""
        old = self.rgb
        try:
            self.rgb = open_camera(spec, self.scenario)
            self.camera_spec = spec
        except Exception as e:                                   # bad spec: keep the old source
            self.rgb = old
            self.ai_img.setText(f"Cannot open source:\n{e}")
            return
        if old is not None and old is not self.rgb:
            old.close()
        self.ai_res = None

    def _select_for_spec(self, spec):
        """Make the combo / text box show the source that was set from the command line."""
        idx, text = 0, ""
        for i, (_, kind, val) in enumerate(SOURCES):
            if kind in ("demo", "spec") and val == spec:
                idx = i
                break
        else:
            kind, _, rest = spec.partition(":")
            if spec.startswith(("http://", "https://", "rtsp://")):
                idx, text = 9, spec
            elif kind == "serial":
                idx, text = 7, rest
            elif kind == "webcam":
                idx, text = 8, rest or "0"
            elif kind == "file":
                idx, text = (5 if rest.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")) else 6), rest
        self.src_combo.blockSignals(True)
        self.src_combo.setCurrentIndex(idx)
        self.src_combo.blockSignals(False)
        self._show_controls(idx, text)

    def _show_controls(self, idx, text=None):
        kind, default = SOURCES[idx][1], SOURCES[idx][2]
        has_edit = kind in ("image", "video", "serial", "webcam", "http")
        self.src_edit.setVisible(has_edit)
        self.src_btn.setVisible(has_edit)
        self.src_btn.setText("Browse..." if kind in ("image", "video") else "Connect")
        if has_edit:
            self.src_edit.setText(text if text is not None else default)
            self.src_edit.setPlaceholderText({"serial": "port, e.g. COM5 or /dev/ttyUSB0",
                                              "webcam": "camera number, e.g. 0",
                                              "http": "http://pi-ip:8000/stream"}.get(kind, "file path"))

    def _src_changed(self, idx):
        kind, val = SOURCES[idx][1], SOURCES[idx][2]
        self._show_controls(idx)
        if kind in ("demo", "spec"):
            self.set_source(val)
        elif kind in ("image", "video"):
            self._browse(kind)

    def _browse(self, kind):
        filt = ("Images (*.jpg *.jpeg *.png *.bmp)" if kind == "image"
                else "Videos (*.mp4 *.avi *.mov *.mkv)")
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Choose a file", "", filt)
        if path:
            self.src_edit.setText(path)
            self.set_source("file:" + path)

    def _src_button(self):
        kind = SOURCES[self.src_combo.currentIndex()][1]
        text = self.src_edit.text().strip()
        if kind in ("image", "video"):
            self._browse(kind)
        elif kind == "serial" and text:
            self.set_source("serial:" + text)
        elif kind == "webcam":
            self.set_source("webcam:" + (text or "0"))
        elif kind == "http" and text:
            self.set_source(text)

    def set_thermal(self, mode):
        """Thermal slot: simulated camera / redness map from the visible camera / MLX90640."""
        self.thermal_msg = ""
        if mode == "pseudo":
            self.thermal = None
        else:
            try:
                self.thermal = open_thermal(mode, self.scenario)
            except Exception as e:                               # e.g. MLX90640 library or sensor missing
                self.thermal_msg = f"{THERMAL_LABELS[THERMAL_MODES.index(mode)]} not available: {e}"
                mode = "sim"
                self.thermal = open_thermal("sim", self.scenario)
                self.th_combo.blockSignals(True)
                self.th_combo.setCurrentIndex(0)
                self.th_combo.blockSignals(False)
        self.thermal_mode = mode
        self.thermal_res = None
        self.g_thermal.setTitle("Thermal analysis (redness map - not degrees)" if mode == "pseudo"
                                else "Thermal analysis (left vs right)")

    def status(self):
        return dict(ok=self.ai_res is not None, spec=self.camera_spec, size=self.last_size or (0, 0))

    def source_label(self):
        spec = self.camera_spec
        return "SIMULATED" if spec == "demo" else "FILE" if spec.startswith("file:") else "LIVE"

    def thermal_label(self):
        return {"sim": "SIMULATED", "pseudo": "FROM CAMERA", "mlx90640": "LIVE"}[self.thermal_mode]

    # ---------------- called ~8 times per second by main.py ----------------
    def refresh(self):
        frame = self.rgb.read() if self.rgb else None
        if frame is None:
            err = getattr(self.rgb, "err", None)
            self.ai_img.setText(f"Waiting for camera...\n{self.camera_spec}" + (f"\n({err})" if err else ""))
            self.ai_txt.setText("--")
            self.ai_res = None
        else:
            self.last_size = (frame.shape[1], frame.shape[0])
            self.ai_res = self.ai.analyse(frame)
            self.ai_img.setPixmap(bgr_to_pixmap(self.ai_res["annotated"]))
            self.ai_txt.setText(self._ai_text(self.ai_res))

        if self.thermal_mode == "pseudo":
            self.thermal_res = None
            if self.ai_res is not None:
                self.thermal_img.setPixmap(bgr_to_pixmap(self.ai_res["redness_img"]))
                self.thermal_txt.setText("Brighter = redder than the rest of the foot.<br>"
                                         "<span style='color:#8b949e'>Relative colour index, NOT temperature.</span>")
            else:
                self.thermal_img.setText("Waiting for the camera...")
        else:
            T = self.thermal.read()
            if T is None:
                self.thermal_img.setText("Waiting for thermal camera...")
                self.thermal_res = None
            else:
                img, self.thermal_res = analyse_thermal(T)
                self.thermal_img.setPixmap(bgr_to_pixmap(img))
                self.thermal_txt.setText(self._thermal_text(self.thermal_res)
                                         + (f"<br><span style='color:#f0883e'>{self.thermal_msg}</span>" if self.thermal_msg else ""))
        return self.thermal_res, self.ai_res

    @staticmethod
    def _thermal_text(r):
        if not r.get("found"):
            return "<b style='color:#8b949e;font-size:{C.px(17)}px;'>Feet not detected</b>"
        return (f"<b style='color:{r['color']};font-size:{C.px(18)}px;'>{r['status']}</b> "
                f"&nbsp; L-R difference: <b>{r['asym']:.1f} C</b><br>"
                f"Warmer: {r['side']} foot ({r['region']})<br>"
                f"Left avg/max: {r['mean_l']:.1f} / {r['max_l']:.1f} C &nbsp; "
                f"Right: {r['mean_r']:.1f} / {r['max_r']:.1f} C")

    @staticmethod
    def _ai_text(r):
        if not r["found"]:
            return (f"<b style='color:#8b949e;font-size:{C.px(17)}px;'>No foot detected</b><br>"
                    "Check lighting and that the foot is in view.")
        names = {"L": "Left", "R": "Right", "?": "Foot"}
        lines = [f"<b style='color:{r['color']};font-size:{C.px(18)}px;'>{r['status']}</b>"]
        for f in r["feet"]:
            lines.append(f"{names[f['side']]}: red {f['red_pct']:.1f}% | dark {f['dark_pct']:.1f}% "
                         f"<span style='color:{f['color']}'>({f['status']})</span>")
        for d in r.get("detections", []):
            lines.append(f"Model: {d['label']} {d['conf']:.0%}")
        lines.append(f"<span style='color:#8b949e;font-size:{C.px(12)}px;'>Screening aid only - not a diagnosis</span>")
        return "<br>".join(lines)