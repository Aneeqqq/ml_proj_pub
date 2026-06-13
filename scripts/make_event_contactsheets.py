"""Assemble scenario 32 power-fade events into labeled contact sheets for visual auditing.

For each fade event, take 3 frames (just-before, peak, just-after) so motion/occlusion is
visible, downscale, label with image number + event id, and tile onto grid sheets.
Output: outputs/s32_eventsheets/sheet_NN.png
"""
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from src.data.derive_label import derive_blockage_label

ROOT = Path(__file__).resolve().parents[1]
CAM = ROOT / "scenario32" / "unit1" / "camera_data"
OUT = ROOT / "outputs" / "s32_eventsheets"
OUT.mkdir(parents=True, exist_ok=True)

dev = pd.read_csv(ROOT / "scenario32" / "scenario32_dev.csv")
pw = derive_blockage_label(dev, env_win=41, q=.90, thr_db=-3.0, min_dur=3, deep_db=-4.5).reset_index(drop=True)
b = (pw == "blocked").to_numpy()

# group contiguous fade frames into events (1-based image numbers)
events = []
i, n = 0, len(b)
while i < n:
    if b[i]:
        j = i
        while j < n and b[j]:
            j += 1
        events.append((i + 1, j))  # images i+1 .. j
        i = j
    else:
        i += 1

THUMB = (320, 180)
COLS = 3          # before / peak / after
PAD = 28
try:
    font = ImageFont.truetype("arial.ttf", 16)
except Exception:
    font = ImageFont.load_default()

def load_thumb(num):
    p = CAM / f"image_{num}.jpg"
    if not p.exists():
        im = Image.new("RGB", THUMB, "black")
    else:
        im = Image.open(p).convert("RGB").resize(THUMB)
    return im

EVENTS_PER_SHEET = 6
for s_i in range(0, len(events), EVENTS_PER_SHEET):
    chunk = events[s_i:s_i + EVENTS_PER_SHEET]
    rows = len(chunk)
    W = COLS * THUMB[0] + (COLS + 1) * PAD
    H = rows * (THUMB[1] + PAD) + PAD
    sheet = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(sheet)
    for r, (start, end) in enumerate(chunk):
        ev_id = s_i + r + 1
        peak = (start + end - 1) // 2
        frames = [max(1, start - 2), peak, min(n, end + 1)]
        tags = ["before", "PEAK", "after"]
        y = PAD + r * (THUMB[1] + PAD)
        for c, (fn, tg) in enumerate(zip(frames, tags)):
            x = PAD + c * (THUMB[0] + PAD)
            sheet.paste(load_thumb(fn), (x, y))
            d.text((x, y - 18), f"ev{ev_id} img_{fn} ({tg}) [fade {start}-{end-1}]",
                   fill="black", font=font)
    sheet_path = OUT / f"sheet_{s_i // EVENTS_PER_SHEET + 1:02d}.png"
    sheet.save(sheet_path)
    print(f"wrote {sheet_path}  (events {s_i+1}-{s_i+len(chunk)})")

print(f"\n{len(events)} events -> {OUT}")
