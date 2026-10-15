"""Print per-frame landmark values for a folder of frames (for tuning)."""
import sys
import time
from pathlib import Path

import cv2

from app.pipeline import Landmarks

lm = Landmarks()
print("frame  faces    cx    cy  size    yaw  pitch  blink    ms")
for path in sorted(Path(sys.argv[1]).glob("*.jpg")):
    img = cv2.imread(str(path))
    t = time.perf_counter()
    f = lm.analyze(img)
    ms = (time.perf_counter() - t) * 1000
    if not f.faces:
        print(f"{path.stem:>5}  0{'':>40}{ms:6.1f}")
        continue
    print(f"{path.stem:>5}  {f.faces:>5}  {f.cx:.2f}  {f.cy:.2f}  {f.size:.2f}  {f.yaw:5.1f}  {f.pitch:5.1f}  {f.blink:5.2f}  {ms:5.1f}")
lm.close()
