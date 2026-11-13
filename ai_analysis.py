"""
ai_analysis.py - general analysis of the foot from the normal (visible-light) camera.

Two layers:
  1. analyse_visual()  rule-based computer vision (works today, no training needed):
       finds each foot, then looks for areas that are redder than the rest of that foot
       (possible inflammation) and areas much darker than the rest (possible wound / tissue damage).
  2. FootAI            optionally adds a trained detector (ultralytics YOLO) if you pass a model file:
       python main.py --model my_foot_model.pt        (class names like "ulcer", "wound" are treated as serious)
       This part is UNTESTED here because there is no trained model yet.

The visible-light camera cannot measure temperature. The "redness map" it produces is a RELATIVE
colour index drawn with heat colours - it is NOT degrees.
Screening aid only: sensitive to lighting and white balance, not a diagnosis.
"""
import numpy as np
import cv2

import config as C


def _filled(mask_bool, min_area):
    cs, _ = cv2.findContours(mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros(mask_bool.shape, np.uint8)
    cv2.drawContours(out, [c for c in cs if cv2.contourArea(c) >= min_area], -1, 1, -1)
    return out > 0


def _status(red_pct, dark_pct):
    if dark_pct >= C.DARK_ALERT_PCT or red_pct >= C.RED_ALERT_PCT:
        return "ALERT", "#f85149"
    if dark_pct >= C.DARK_WATCH_PCT or red_pct >= C.RED_WATCH_PCT:
        return "WATCH", "#d29922"
    return "NORMAL", "#3fb950"


def analyse_visual(frame):
    """
    Returns dict: found, annotated (BGR), redness_img (BGR), status, color,
                  feet = [ {side 'L'/'R'/'?', cx, red_pct, dark_pct, max_red, area_pct, status, color} ]
    """
    h, w = frame.shape[:2]
    blur = cv2.GaussianBlur(frame, (5, 5), 0)

    # 1) find the feet (skin segmentation)
    ycc = cv2.cvtColor(blur, cv2.COLOR_BGR2YCrCb)
    skin = cv2.inRange(ycc, (0, 133, 77), (255, 180, 135))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, k)
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, k, iterations=2)
    cs, _ = cv2.findContours(skin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cs = [c for c in cs if cv2.contourArea(c) >= C.MIN_FOOT_FRAC * h * w]
    cs = sorted(cs, key=cv2.contourArea, reverse=True)[:2]
    if not cs:
        blank = np.full((h, w, 3), (30, 22, 20), np.uint8)
        return dict(found=False, annotated=frame, redness_img=blank, status="NO FOOT",
                    color="#8b949e", feet=[])

    def cx_of(c):
        m = cv2.moments(c)
        return m["m10"] / m["m00"] if m["m00"] else 0.0

    cs = sorted(cs, key=cx_of)                              # left to right in the image
    if len(cs) == 2:
        sides = ["L", "R"] if C.AI_LEFT_IS_IMAGE_LEFT else ["R", "L"]
    else:
        sides = ["?"]

    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, a = lab[:, :, 0], lab[:, :, 1]

    img = frame.copy()
    union = np.zeros((h, w), bool)
    masks = []
    for c in cs:
        m = np.zeros((h, w), np.uint8)
        cv2.drawContours(m, [c], -1, 255, -1)               # filled: wounds inside the foot stay inside
        masks.append(m > 0)
        union |= m > 0
    img[~union] = (img[~union] * 0.45).astype(np.uint8)

    vmap = np.zeros((h, w), np.float32)
    feet = []
    for c, m, side in zip(cs, masks, sides):
        area = float(m.sum())
        delta = a - float(np.median(a[m]))
        L_med = float(np.median(L[m]))
        min_area = C.MIN_REGION_FRAC * area
        red = _filled(m & (delta > C.RED_DELTA), min_area)
        dark = _filled(m & (L < L_med * C.DARK_RATIO), min_area)
        red_pct, dark_pct = 100.0 * red.sum() / area, 100.0 * dark.sum() / area
        status, color = _status(red_pct, dark_pct)
        feet.append(dict(side=side, cx=cx_of(c), red_pct=red_pct, dark_pct=dark_pct,
                         max_red=float(delta[m].max()), area_pct=100.0 * area / (h * w),
                         status=status, color=color))

        overlay = img.copy()
        overlay[red] = (0, 0, 255)
        overlay[dark] = (200, 60, 160)
        img = cv2.addWeighted(overlay, 0.40, img, 0.60, 0)
        cv2.drawContours(img, [c], -1, (80, 200, 80), 2)
        for mm, col in ((red, (0, 0, 255)), (dark, (200, 60, 160))):
            c2, _ = cv2.findContours(mm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, c2, -1, col, 2)
        if red.any():
            iy, ix = np.unravel_index(np.argmax(np.where(red, delta, -1e9)), delta.shape)
            cv2.drawMarker(img, (int(ix), int(iy)), (255, 255, 255), cv2.MARKER_CROSS, 14, 2)
        x, y, bw, bh = cv2.boundingRect(c)
        cv2.putText(img, side, (x + bw // 2 - 6, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        v = np.clip((delta + C.RED_DELTA) / (3 * C.RED_DELTA), 0, 1)
        vmap[m] = v[m]

    # relative redness drawn with heat colours (NOT temperature)
    vmap = cv2.GaussianBlur(vmap, (0, 0), 3)
    heat = cv2.applyColorMap((vmap * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    alpha = cv2.GaussianBlur(union.astype(np.float32), (0, 0), 2)[..., None]
    redness = (heat * alpha + np.array([30, 22, 20], np.float32) * (1 - alpha)).astype(np.uint8)

    order = {"NORMAL": 0, "WATCH": 1, "ALERT": 2}
    worst = max(feet, key=lambda f: order[f["status"]])
    return dict(found=True, annotated=img, redness_img=redness, status=worst["status"],
                color=worst["color"], feet=feet)


class FootAI:
    """analyse_visual() plus an optional trained YOLO detector."""
    def __init__(self, model_path=None):
        self.model, self.error = None, None
        if model_path:
            try:
                from ultralytics import YOLO               # pip install ultralytics
                self.model = YOLO(model_path)
            except Exception as e:                         # missing package / bad file: keep running
                self.error = str(e)

    def analyse(self, frame):
        res = analyse_visual(frame)
        res["detections"] = []
        if self.model is None:
            return res
        try:
            r = self.model(frame, verbose=False)[0]
            for b in r.boxes:
                conf = float(b.conf[0])
                label = str(r.names[int(b.cls[0])])
                x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
                cx = (x1 + x2) / 2
                side = "?"
                if res["feet"]:
                    side = min(res["feet"], key=lambda f: abs(f["cx"] - cx))["side"]
                res["detections"].append(dict(label=label, conf=conf, side=side, box=(x1, y1, x2, y2)))
                cv2.rectangle(res["annotated"], (x1, y1), (x2, y2), (0, 215, 255), 2)
                cv2.putText(res["annotated"], f"{label} {conf:.0%}", (x1, max(14, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 215, 255), 1)
        except Exception as e:
            self.error = str(e)
        return res
