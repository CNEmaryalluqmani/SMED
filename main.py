"""
MEDthon - Diabetic Foot Monitor.  Run this file:

    python main.py                                   everything simulated (no hardware needed)
    python main.py --scenario ulcer                  start with a different simulated situation
    python main.py --camera serial:/dev/ttyUSB0      real ESP32-CAM over USB for the AI analysis
    python main.py --camera webcam:0                 test with a normal webcam
    python main.py --thermal pseudo                  no thermal camera: show the redness map instead
    python main.py --model my_foot_model.pt          add a trained YOLO model to the AI analysis
    python main.py --lang ar                         start with Arabic advice
    python main.py --windowed                        maximised window instead of full screen (F11 toggles)

Layout: the Sensors panel (ONE tile per sensor) sits on the left side, independent of the dashboard
(F3 hides / shows it). The pulse column (waveform with its 3 values under it) sits beside the pressure map.

Files:
    config.py          sizes, thresholds, zones            sensors.py        sensor models (simulated values)
    camera.py          camera input + 2-slot panel         sensor_fields.py  on-screen field for each sensor
    thermal_camera.py  thermal camera + analysis           services.py       risk stage, status and advice
    ai_analysis.py     AI / image analysis of the foot     foot_3d.py        interactive 3D feet
    foot_geometry.py   shared foot shape and colour maps   widgets.py        picture widget that fills its space
"""
import argparse
import sys
import time

import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

import config as C
from sensors import SimScenario, SensorHub, LRAActuator, SCENARIOS
from camera import CameraPanel
from sensor_fields import PressurePanel, PPGPanel, MatPanel, LRAPanel, SensorsDock, AmbientWidget
from services import RiskEngine, ServicesPanel
from foot_3d import Feet3D

STYLE = """
QWidget{background:#0d1117;color:#e6edf3;font-size:@FS@px;}
QLabel{background:transparent;}
QGroupBox{border:1px solid #30363d;border-radius:8px;margin-top:14px;padding-top:12px;font-weight:bold;}
QGroupBox::title{subcontrol-origin:margin;left:10px;padding:0 4px;color:#58a6ff;}
QPushButton{background:#21262d;border:1px solid #30363d;border-radius:6px;padding:6px 12px;}
QPushButton:hover{background:#30363d;}
QComboBox{background:#21262d;border:1px solid #30363d;border-radius:6px;padding:4px 8px;}
QPlainTextEdit{background:#161b22;border:1px solid #30363d;border-radius:6px;}
QDockWidget{font-weight:bold;}
QDockWidget::title{background:#161b22;padding:6px 10px;border:1px solid #30363d;color:#58a6ff;}
QScrollBar:vertical{background:#0d1117;width:12px;}
QScrollBar::handle:vertical{background:#30363d;border-radius:5px;min-height:30px;}
QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{height:0;}
"""


def ui_scale(width, height):
    """1.0 = design size 1920 x 1080. Text and panels follow the screen."""
    return max(0.85, min(1.6, min(width / 1920.0, height / 1080.0)))


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, args, fs=15, fullscreen=True):
        super().__init__()
        self.setWindowTitle("MEDthon - Diabetic Foot Monitor")
        self.setStyleSheet(STYLE.replace("@FS@", str(fs)))
        self.fullscreen = fullscreen
        QtGui.QShortcut(QtGui.QKeySequence("F11"), self, activated=self.toggle_fullscreen)
        QtGui.QShortcut(QtGui.QKeySequence("Esc"), self, activated=lambda: self.set_fullscreen(False))
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Q"), self, activated=self.close)
        QtGui.QShortcut(QtGui.QKeySequence("F3"), self, activated=self.toggle_sensors)

        self.scenario = SimScenario(args.scenario)
        self.hub = SensorHub(self.scenario)
        self.engine = RiskEngine()
        self.thermal_res = self.ai_res = None
        self.t_ppg = self.t_cam = self.t_slow = 0.0          # when each part was last refreshed

        self.camera = CameraPanel(self.scenario, args.camera, args.thermal, args.model)
        self.feet3d = Feet3D()
        self.pressure, self.ppg, self.mat = PressurePanel(), PPGPanel(), MatPanel()
        self.lra = LRAPanel(LRAActuator())                     # motor logic only: NOT shown on the dashboard, its value is in the Sensors panel
        self.services = ServicesPanel(args.lang)
        self.services.lang_changed.connect(self.lra.set_lang)
        self.lra.set_lang(args.lang)

        # ---- header
        title = QtWidgets.QLabel("MEDthon - Diabetic Foot Monitor")
        title.setStyleSheet(f"font-size:{C.px(22)}px;font-weight:bold;")
        keys = QtWidgets.QLabel("F3 sensors panel | F11 full screen | Esc exit full screen | Ctrl+Q quit")
        keys.setStyleSheet(f"color:#6e7681;font-size:{C.px(12)}px;")
        keys.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)   # may shrink on small screens
        self.ambient = AmbientWidget()
        self.vitals = QtWidgets.QLabel("Pulse --")
        self.vitals.setStyleSheet(f"color:#3fb950;font-weight:bold;font-size:{C.px(18)}px;")
        self.scn_combo = QtWidgets.QComboBox()
        self.scn_combo.addItems(list(SCENARIOS))
        self.scn_combo.setCurrentText(args.scenario)
        self.scn_combo.currentTextChanged.connect(self.scenario.set)
        self.person_chk = QtWidgets.QCheckBox("Person on the mat")          # simulation control: drives weight, pressure and the UV safety lock
        self.person_chk.setChecked(True)
        self.person_chk.toggled.connect(self.scenario.set_person)
        mode = QtWidgets.QLabel("Mode: SIMULATION")
        mode.setStyleSheet("color:#d29922;font-weight:bold;")
        head = QtWidgets.QHBoxLayout()
        head.addWidget(title)
        head.addSpacing(16)
        head.addWidget(keys)
        head.addStretch(1)
        head.addWidget(self.vitals)
        head.addSpacing(18)
        head.addWidget(QtWidgets.QLabel("Simulated situation:"))
        head.addWidget(self.scn_combo)
        head.addSpacing(12)
        head.addWidget(self.person_chk)
        head.addSpacing(16)
        head.addWidget(self.ambient)
        head.addSpacing(16)
        head.addWidget(mode)

        # ---- dashboard: left = cameras + maps + vibration, right = 3D + services
        bottom = QtWidgets.QHBoxLayout()                       # FSR map | pulse column (waveform + 3 values) | weight / UV / vibration column
        bottom.addWidget(self.pressure, 3)
        bottom.addWidget(self.ppg, 3)                          # the pulse sits right beside the pressure map
        bottom.addWidget(self.mat, 2)                          # weight, UV disinfection and vibration motor: a simple reading each
        left = QtWidgets.QVBoxLayout()
        left.addWidget(self.camera, 1)                         # top row: thermal | AI   (same height as ...)
        left.addLayout(bottom, 1)                              # ... bottom row: pressure | pulse | weight, UV, LRA

        g3 = QtWidgets.QGroupBox("3D feet")
        QtWidgets.QVBoxLayout(g3).addWidget(self.feet3d)
        gs = QtWidgets.QGroupBox("Services - foot status, risk stage and advice")
        QtWidgets.QVBoxLayout(gs).addWidget(self.services)
        right = QtWidgets.QVBoxLayout()
        right.addWidget(g3, 4)
        right.addWidget(gs, 5)

        dash = QtWidgets.QWidget()
        body = QtWidgets.QHBoxLayout(dash)
        body.setContentsMargins(0, 0, 0, 0)
        body.addLayout(left, 3)
        body.addLayout(right, 2)

        sa = QtWidgets.QScrollArea()                           # tiny screens get scroll bars instead of cut-off
        sa.setWidgetResizable(True)
        sa.setFrameShape(QtWidgets.QFrame.NoFrame)
        sa.setWidget(dash)

        central = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(central)
        root.addLayout(head)
        root.addWidget(sa, 1)
        self.setCentralWidget(central)

        # ---- the independent Sensors panel on the left side
        self.sensors = SensorsDock()
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.sensors)
        self.resizeDocks([self.sensors], [C.px(400)], QtCore.Qt.Horizontal)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(20)

    def showEvent(self, e):
        super().showEvent(e)
        QtCore.QTimer.singleShot(0, self._fit_dock)            # once the real font sizes are known

    def _fit_dock(self):
        need = self.sensors.fit_width()
        self.resizeDocks([self.sensors], [max(C.px(400), need)], QtCore.Qt.Horizontal)

    def set_fullscreen(self, on):
        self.fullscreen = on
        self.showFullScreen() if on else self.showMaximized()

    def toggle_fullscreen(self):
        self.set_fullscreen(not self.isFullScreen())

    def toggle_sensors(self):
        self.sensors.setVisible(not self.sensors.isVisible())

    def source_badges(self):
        s = dict(self.hub.sources)
        s["thermal"], s["camera"] = self.camera.thermal_label(), self.camera.source_label()
        return s

    def tick(self):
        """Fast loop (50 Hz) collects sensor data; every part is redrawn at its own pace (by the clock)."""
        now = time.monotonic()
        d = self.hub.read()
        self.ppg.push(d)
        if now - self.t_ppg >= 0.066:                          # pulse waveform ~15 fps
            self.t_ppg = now
            self.ppg.redraw()
        if now - self.t_cam >= 0.15:                           # cameras ~7 fps
            self.t_cam = now
            self.thermal_res, self.ai_res = self.camera.refresh()
        if now - self.t_slow >= 0.15:                          # everything else ~7 times per second
            dt = min(0.5, now - self.t_slow) if self.t_slow else 0.15
            self.t_slow = now
            self.ambient.set_values(d["dht_t"], d["dht_h"])
            self.vitals.setText(f"Pulse {d['bpm']:.0f} BPM  |  SpO2 {d['spo2']:.0f} %")
            feet, general, overall, events = self.engine.update(d, self.thermal_res, self.ai_res, dt)
            self.services.show(feet, general, overall)
            for e in events:
                self.lra.raise_alert(e)
            if d["weight"] >= C.WEIGHT_PRESENT_KG:                 # a person stands on the mat: while a foot stays at WARNING or worse,
                for s in "LR":                                     # the motor repeats its buzz (once per LRA_COOLDOWN seconds)
                    if feet[s].stage >= 2:
                        self.lra.raise_alert(dict(side=s))
            self.pressure.update_data(d)
            self.ppg.update_data(d)
            self.mat.update_data(d, self.lra.status())
            self.feet3d.set_data(d)
            if self.sensors.isVisible():
                ctx = dict(data=d, thermal=self.thermal_res, ai=self.ai_res, camera=self.camera.status(),
                           lra=self.lra.status())
                self.sensors.update_data(ctx, self.source_badges())


def main():
    ap = argparse.ArgumentParser(description="MEDthon - Diabetic Foot Monitor")
    ap.add_argument("--camera", default="demo", help="demo | serial:PORT[@BAUD] | http://... | webcam:N | file:PATH")
    ap.add_argument("--thermal", default="sim", choices=["sim", "mlx90640", "pseudo"])
    ap.add_argument("--scenario", default="hotspot", choices=list(SCENARIOS))
    ap.add_argument("--model", default=None, help="optional trained YOLO model (.pt) for the AI analysis")
    ap.add_argument("--lang", default=C.LANGUAGE, choices=["en", "ar"])
    ap.add_argument("--windowed", action="store_true", help="maximised window instead of full screen")
    args = ap.parse_args()

    pg.setConfigOptions(antialias=True)
    app = QtWidgets.QApplication(sys.argv[:1])
    geo = app.primaryScreen().geometry()
    C.UI_SCALE = ui_scale(geo.width(), geo.height())            # before any widget is built
    w = MainWindow(args, int(round(15 * C.UI_SCALE)), fullscreen=not args.windowed)
    w.show()
    w.set_fullscreen(not args.windowed)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()