"""Chronological contact sheets for FULL scenario-32 frame-by-frame labeling.

Walk every frame in order (image_1..image_N) at a stride, resize, tile onto sheets
with the image number on each thumb. Mirrors the GUI's +/-5 stepping so the whole
scenario can be labeled in order. Output: outputs/s32_chrono/sheet_NNN.png
"""
from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
CAM = ROOT / "scenario32" / "unit1" / "camera_data"
OUT = ROOT / "outputs" / "s32_chrono"
OUT.mkdir(parents=True, exist_ok=True)

dev = pd.read_csv(ROOT / "scenario32" / "scenario32_dev.csv")
N = len(dev)                      # image numbers are 1..N
STRIDE = 5
COLS, ROWS = 4, 6                 # 24 thumbs/sheet
THUMB = (336, 189)
PAD_X, PAD_Y, TOP = 12, 26, 16
BMIN, BMAX = 2, 62               # beam->x (low beam = right), calibrated vs imgs 1/531/781

try:
    font = ImageFont.truetype("arialbd.ttf", 16)
except Exception:
    font = ImageFont.load_default()

samples = list(range(1, N + 1, STRIDE))
per_sheet = COLS * ROWS

def load(num):
    p = CAM / f"image_{num}.jpg"
    if not p.exists():
        return Image.new("RGB", THUMB, "black")
    im = Image.open(p).convert("RGB").resize(THUMB)
    beam = int(dev["unit1_beam"].iloc[num - 1])
    frac = 1 - (beam - BMIN) / (BMAX - BMIN)     # low beam -> right
    x = int(round(frac * THUMB[0]))
    d = ImageDraw.Draw(im)
    d.line([(x, 0), (x, THUMB[1])], fill=(255, 0, 0), width=2)
    return im

n_sheets = (len(samples) + per_sheet - 1) // per_sheet
for s in range(n_sheets):
    chunk = samples[s * per_sheet:(s + 1) * per_sheet]
    W = COLS * THUMB[0] + (COLS + 1) * PAD_X
    H = ROWS * (THUMB[1] + PAD_Y) + PAD_Y
    sheet = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(sheet)
    for i, num in enumerate(chunk):
        r, c = divmod(i, COLS)
        x = PAD_X + c * (THUMB[0] + PAD_X)
        y = TOP + r * (THUMB[1] + PAD_Y)
        sheet.paste(load(num), (x, y))
        seq = int(dev["seq_index"].iloc[num - 1])
        d.text((x, y - 15), f"img_{num}  seq{seq}", fill="black", font=font)
    sheet.save(OUT / f"sheet_{s+1:03d}.png")

print(f"{N} frames, stride {STRIDE} -> {len(samples)} samples in {n_sheets} sheets -> {OUT}")
