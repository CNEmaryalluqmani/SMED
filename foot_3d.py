"""
foot_3d.py - the interactive 3D foot model (this file does ONLY that).

Two solid, light-beige feet standing on a sensor mat (a grid floor). The mat under each foot shows the
measurement as a smooth colour map, like a clinical plantar-pressure plate: blue = low ... red = high
(pressure). It also shows as a coloured rim around each foot; choose
"Feet: See-through" (or the Under button) to see the whole map under the foot.
Because the sensors are under the sole, nothing is painted on the top of the foot. Left foot on the left.
Drag = rotate, wheel = zoom, Shift+drag = pan.
Call set_data(data) with the dict from SensorHub.read(). Lighting is baked into the vertex colours.
Needs OpenGL (pyqtgraph.opengl). If the picture stays black over remote desktop try:
    LIBGL_ALWAYS_SOFTWARE=1 python main.py
"""
import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PySide6 import QtWidgets

import config as C
from foot_geometry import KT, KH, outline, cmap


def build_foot(nt=80, nth=44):
    """Y = heel->toe (cm), X = medial->lateral (right-foot frame), Z = up. Sole on Z=0."""
    t = np.linspace(0, 1, nt)
    th = np.linspace(0, 2 * np.pi, nth, endpoint=False)
    hw, center = outline(t)
    h = np.interp(t, KT, KH)
    h = np.convolve(np.pad(h, 2, mode="edge"), np.ones(5) / 5, mode="valid")
    hw[0] = hw[-1] = 0.0                      # close the mesh at heel and toe tip
    h[0] = h[-1] = 0.0
    arch = 1.0 * np.exp(-((t - 0.40) / 0.12) ** 2)

    c = np.cos(th)[None, :]
    s = np.sin(th)[None, :]
    cx = np.sign(c) * np.abs(c) ** 0.8
    sz = np.where(s >= 0, np.abs(s) ** 0.8, -np.abs(s) ** 0.35)

    X = center[:, None] + hw[:, None] * cx
    Y = np.repeat((t * C.FOOT_LEN)[:, None], nth, axis=1)
    Z = (h[:, None] / 2.0) * (1 + sz) + arch[:, None] * np.clip(-c, 0, 1) * (s < 0)
    verts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3).astype(np.float32)

    i = np.arange(nt - 1)[:, None]
    j = np.arange(nth)[None, :]
    a = i * nth + j
    b = i * nth + (j + 1) % nth
    cc = (i + 1) * nth + j
    d = (i + 1) * nth + (j + 1) % nth
    faces = np.concatenate([np.stack([a, b, cc], -1).reshape(-1, 3), np.stack([b, d, cc], -1).reshape(-1, 3)])

    ring_c = verts.reshape(nt, nth, 3).mean(axis=1, keepdims=True)       # make normals point outward
    out = verts - np.repeat(ring_c, nth, axis=1).reshape(-1, 3)
    n = np.cross(verts[faces[:, 1]] - verts[faces[:, 0]], verts[faces[:, 2]] - verts[faces[:, 0]])
    if np.sum(n * (out[faces[:, 0]] + out[faces[:, 1]] + out[faces[:, 2]])) < 0:
        faces = faces[:, ::-1].copy()

    sole_w = np.where(np.repeat((s < 0).astype(float), nt, axis=0).reshape(-1) > 0, 1.0, 0.18)
    return verts, faces, X.reshape(-1), Y.reshape(-1), sole_w


def vertex_normals(verts, faces):
    tri = verts[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    vn = np.zeros_like(verts, dtype=np.float64)
    for k in range(3):
        np.add.at(vn, faces[:, k], fn)
    return vn / (np.linalg.norm(vn, axis=1, keepdims=True) + 1e-9)


def _feet_options():
    """Translucent drawing with back faces hidden (falls back to plain 'translucent')."""
    try:
        from OpenGL.GL import GL_CULL_FACE
        from pyqtgraph.opengl.GLGraphicsItem import GLOptions
        opts = dict(GLOptions["translucent"])
        opts[GL_CULL_FACE] = True
        return opts
    except Exception:
        return "translucent"


class Feet3D(QtWidgets.QWidget):
    SKIN = np.array([0.96, 0.84, 0.72])                  # light beige
    SEE_THROUGH_ALPHA = 0.38                             # alpha when 'Feet: See-through' is chosen
    LIGHT = np.array([-0.35, -0.45, 0.82]) / np.linalg.norm([-0.35, -0.45, 0.82])
    GRAD = "qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #1a33cc,stop:0.25 #00ccf0,stop:0.5 #33cc4d,stop:0.75 #ffe626,stop:1 #e61a1a)"

    def __init__(self):
        super().__init__()
        verts, faces, x, y, _ = build_foot()
        # influence of every sensor zone on every vertex (smooth clinical-style map)
        D = np.stack([np.hypot(x - zx, y - zy) for _, (zx, zy) in C.ZONES])
        self.G_w = np.exp(-(D ** 2) / (2 * 2.3 ** 2))
        # footprint = the bottom faces of the foot, flattened onto the mat
        tri = verts[faces]
        fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        fn = fn / (np.linalg.norm(fn, axis=1, keepdims=True) + 1e-9)
        bottom = fn[:, 2] < -0.25
        self.mode = "Pressure"

        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor((24, 30, 40))
        grid = gl.GLGridItem()                           # the sensor mat
        grid.setSize(40, 40)
        grid.setSpacing(2, 2)
        grid.translate(0, 13, 0)
        try:
            grid.setColor((110, 125, 150, 150))
        except Exception:
            pass
        self.view.addItem(grid)

        self.feet_opts = _feet_options()
        self.foot, self.print_ = {}, {}
        self.shade = {}
        for side, off in (("L", -6.5), ("R", 6.5)):
            v, f = verts.copy(), faces
            if side == "L":                              # mirror -> left foot
                v[:, 0] = -v[:, 0]
                f = faces[:, ::-1].copy()
            v[:, 0] += off

            # footprint on the mat (opaque, drawn after the grid)
            vp = v.copy()
            vp[:, 2] = -0.03                                         # just under the sole and the grid: nothing covers it from below
            vp[:, 0] = off + (v[:, 0] - off) * 1.22                  # a bit larger than the foot: coloured rim
            vp[:, 1] = 13.0 + (v[:, 1] - 13.0) * 1.05
            fp = f[bottom]
            mdp = gl.MeshData(vertexes=vp, faces=fp)
            mdp.setVertexColors(self.print_colors("Pressure", [0] * len(C.ZONES)))
            mp = gl.GLMeshItem(meshdata=mdp, smooth=False, shader=None, computeNormals=False, glOptions="opaque")
            mp.setDepthValue(1)
            self.view.addItem(mp)
            self.print_[side] = (mdp, mp)

            # the foot itself (beige, see-through, lit)
            self.shade[side] = 0.60 + 0.40 * np.clip(vertex_normals(v, f) @ self.LIGHT, 0, 1)
            rgb = np.clip(self.SKIN[None, :] * self.shade[side][:, None], 0, 1)
            self.foot_rgb = getattr(self, "foot_rgb", {})
            self.foot_rgb[side] = rgb
            col = np.concatenate([rgb, np.ones((len(rgb), 1))], axis=1).astype(np.float32)        # solid
            md = gl.MeshData(vertexes=v, faces=f)
            md.setVertexColors(col)
            mf = gl.GLMeshItem(meshdata=md, smooth=True, shader=None, computeNormals=False, glOptions="opaque")
            mf.setDepthValue(2)
            self.view.addItem(mf)
            self.foot[side] = (md, mf)

        self.set_view("reset")

        top = QtWidgets.QHBoxLayout()
        self.style_combo = QtWidgets.QComboBox()
        self.style_combo.addItems(["Solid", "See-through"])
        self.style_combo.currentTextChanged.connect(self._foot_style)
        top.addWidget(QtWidgets.QLabel("Feet:"))
        top.addWidget(self.style_combo)
        top.addStretch(1)
        views = QtWidgets.QHBoxLayout()
        views.addWidget(QtWidgets.QLabel("View:"))
        for label, key in [("Reset", "reset"), ("Top", "top"), ("Side", "side"), ("Under", "sole")]:
            b = QtWidgets.QPushButton(label)
            b.setStyleSheet("padding:4px 8px;")
            b.clicked.connect(lambda _=False, k=key: self.set_view(k))
            views.addWidget(b)
        views.addStretch(1)

        self.bar = QtWidgets.QLabel()
        self.bar.setFixedHeight(C.px(12))
        self.bar.setStyleSheet(f"background:{self.GRAD};border-radius:4px;")
        self.lo, self.hi = QtWidgets.QLabel(), QtWidgets.QLabel()
        leg = QtWidgets.QHBoxLayout()
        leg.addWidget(self.lo)
        leg.addWidget(self.bar, 1)
        leg.addWidget(self.hi)
        for lbl in (self.lo, self.hi):
            lbl.setStyleSheet(f"color:#8b949e;font-size:{C.px(12)}px;")
        self._set_legend()

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(top)
        lay.addLayout(views)
        lay.addWidget(self.view, 1)
        lay.addLayout(leg)
        hint = QtWidgets.QLabel("The feet stand on the sensor mat; the colours on the mat are the sensor readings "
                                "(left foot on the left). 'Feet: See-through' or 'Under' shows the whole map. "
                                "Drag = rotate | Wheel = zoom | Shift+drag = pan")
        hint.setStyleSheet(f"color:#8b949e;font-size:{C.px(12)}px;")
        hint.setWordWrap(True)
        lay.addWidget(hint)

    def _foot_style(self, style):
        """Solid beige feet (default) or see-through feet that show the whole map under them."""
        solid = style == "Solid"
        for side in "LR":
            md, mesh = self.foot[side]
            alpha = np.ones((len(self.foot_rgb[side]), 1)) if solid else np.full((len(self.foot_rgb[side]), 1), self.SEE_THROUGH_ALPHA)
            md.setVertexColors(np.concatenate([self.foot_rgb[side], alpha], axis=1).astype(np.float32))
            mesh.setGLOptions("opaque" if solid else self.feet_opts)
            mesh.meshDataChanged()

    def _set_legend(self):
        self.lo.setText("0 kPa (low)")
        self.hi.setText(f"{C.PRESSURE_MAX:.0f}+ kPa (high)")

    def set_view(self, key):
        self.view.opts["center"] = pg.Vector(0, 13, 1)
        presets = {"reset": dict(distance=52, elevation=42, azimuth=-90),
                   "top": dict(distance=50, elevation=89, azimuth=-90),
                   "side": dict(distance=55, elevation=6, azimuth=0),
                   "sole": dict(distance=52, elevation=-50, azimuth=-90)}
        self.view.setCameraPosition(**presets[key])

    def print_colors(self, mode, values):
        """Smooth colour map for the mat under one foot."""
        w = np.asarray(values, dtype=float)
        v = np.clip((self.G_w * (w / C.PRESSURE_MAX)[:, None]).sum(0), 0, 1)
        rgb = cmap(v)
        return np.concatenate([rgb, np.ones((len(rgb), 1))], axis=1).astype(np.float32)

    def set_data(self, data):
        key = "pressure"
        for side in "LR":
            md, mesh = self.print_[side]
            md.setVertexColors(self.print_colors(self.mode, data[side][key]))
            mesh.meshDataChanged()