"""
make_samples.py - (re)creates the sample images and video in the samples/ folder.
Run once:  python make_samples.py
The video tells a 15 second story: normal feet -> inflamed area appears -> wound appears (then it loops).
You can replace these files with your own photos / videos, or pick your own from the camera box in the window.
"""
import os

import cv2
import numpy as np

from sensors import SimScenario
from camera import DemoFootCamera

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")
os.makedirs(OUT, exist_ok=True)


def frame(scn, t):
    sc = SimScenario(scn)
    cam = DemoFootCamera(sc)
    return cam.read(t)


for name, scn in (("feet_normal", "normal"), ("feet_inflamed", "hotspot"), ("feet_wound", "ulcer")):
    cv2.imwrite(os.path.join(OUT, name + ".jpg"), frame(scn, 0.0), [cv2.IMWRITE_JPEG_QUALITY, 92])

fps, seconds = 15, 15
w = cv2.VideoWriter(os.path.join(OUT, "demo_feet.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (DemoFootCamera.W, DemoFootCamera.H))
for i in range(fps * seconds):
    t = i / fps
    scn = "normal" if t < 5 else "hotspot" if t < 10 else "ulcer"
    w.write(frame(scn, t))
w.release()
print("samples written to", OUT)