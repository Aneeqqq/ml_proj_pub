"""Overlay the BS->unit2 beam direction on camera frames to locate the tracked car.

unit1_beam (2..62) is the BS antenna beam pointing AT unit2, so it encodes unit2's
azimuth ~ horizontal pixel position. Hypothesis (calibrate by eye): low beam = right
side of frame. x = W * (1 - (beam - BMIN) / (BMAX - BMIN)).

Draw a vertical line + label at that x. View output to verify the line sits on the
tracked car; adjust BMIN/BMAX/flip if needed.

Usage: .venv/Scripts/python.exe -m scripts.beam_overlay 1 34 50 531 781 1143
"""
import sys
from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
CAM = ROOT / "scenario32" / "unit1" / "camera_data"
OUT = ROOT / "outputs" / "s32_beam"
OUT.mkdir(parents=True, exist_ok=True)

dev = pd.read_csv(ROOT / "scenario32" / "scenario32_dev.csv")

BMIN, BMAX = 2, 62
FLIP = True   # True: low beam -> right side

try:
    font = ImageFont.truetype("arialbd.ttf", 22)
except Exception:
    font = ImageFont.load_default()


def beam_to_x(beam, W):
    frac = (beam - BMIN) / (BMAX - BMIN)
    if FLIP:
        frac = 1 - frac
    return int(round(frac * W))


def overlay(num):
    p = CAM / f"image_{num}.jpg"
    im = Image.open(p).convert("RGB")
    W, H = im.size
    beam = int(dev["unit1_beam"].iloc[num - 1])
    x = beam_to_x(beam, W)
    d = ImageDraw.Draw(im)
    d.line([(x, 0), (x, H)], fill=(255, 0, 0), width=4)
    d.rectangle([x - 60, H - 34, x + 60, H], fill=(255, 0, 0))
    d.text((x - 55, H - 30), f"unit2 b={beam}", fill="white", font=font)
    out = OUT / f"ov_{num}.jpg"
    im.save(out)
    return out


if __name__ == "__main__":
    nums = [int(a) for a in sys.argv[1:]] or [1, 34, 50, 531, 781, 1143]
    for n in nums:
        print(overlay(n))
