"""Event-verification sheets for scenario 32: before/peak/after per fade event, with
the BS->unit2 beam marker drawn, large enough to judge occlusion at the tracked car.

Works in ROW space (handles non-contiguous image numbers via dev['index']).
Output: outputs/s32_verify/vsheet_NN.png
"""
from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from src.data.derive_label import derive_blockage_label

ROOT = Path(__file__).resolve().parents[1]
CAM = ROOT / "scenario32" / "unit1" / "camera_data"
OUT = ROOT / "outputs" / "s32_verify"
OUT.mkdir(parents=True, exist_ok=True)

dev = pd.read_csv(ROOT / "scenario32" / "scenario32_dev.csv").reset_index(drop=True)
pw = derive_blockage_label(dev, env_win=41, q=.90, thr_db=-3.0, min_dur=3, deep_db=-4.5).reset_index(drop=True)
b = (pw == "blocked").to_numpy()
img = dev["index"].to_numpy()
beam_col = dev["unit1_beam"].to_numpy()
N = len(b)

# events in ROW space
events = []
i = 0
while i < N:
    if b[i]:
        j = i
        while j < N and b[j]:
            j += 1
        events.append((i, j - 1))  # inclusive row range
        i = j
    else:
        i += 1

BMIN, BMAX = 2, 62
THUMB = (440, 248)
COLS = 3                      # before / peak / after
PAD_X, PAD_Y, TOP = 14, 30, 18
EVENTS_PER_SHEET = 4
try:
    font = ImageFont.truetype("arialbd.ttf", 18)
except Exception:
    font = ImageFont.load_default()


def thumb_for_row(row):
    num = int(img[row])
    p = CAM / f"image_{num}.jpg"
    if not p.exists():
        return Image.new("RGB", THUMB, "black"), num, int(beam_col[row])
    im = Image.open(p).convert("RGB").resize(THUMB)
    beam = int(beam_col[row])
    frac = 1 - (beam - BMIN) / (BMAX - BMIN)
    x = int(round(frac * THUMB[0]))
    d = ImageDraw.Draw(im)
    d.line([(x, 0), (x, THUMB[1])], fill=(255, 0, 0), width=3)
    return im, num, beam


n_sheets = (len(events) + EVENTS_PER_SHEET - 1) // EVENTS_PER_SHEET
for s in range(n_sheets):
    chunk = events[s * EVENTS_PER_SHEET:(s + 1) * EVENTS_PER_SHEET]
    W = COLS * THUMB[0] + (COLS + 1) * PAD_X
    H = len(chunk) * (THUMB[1] + PAD_Y) + PAD_Y
    sheet = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(sheet)
    for r, (r0, r1) in enumerate(chunk):
        ev_id = s * EVENTS_PER_SHEET + r + 1
        peak = (r0 + r1) // 2
        rows = [max(0, r0 - 1), peak, min(N - 1, r1 + 1)]
        tags = ["before", "PEAK", "after"]
        y = TOP + r * (THUMB[1] + PAD_Y)
        for c, (rr, tg) in enumerate(zip(rows, tags)):
            x = PAD_X + c * (THUMB[0] + PAD_X)
            im, num, beam = thumb_for_row(rr)
            sheet.paste(im, (x, y))
            d.text((x, y - 16), f"ev{ev_id} {tg} img_{num} b={beam}", fill="black", font=font)
    sheet.save(OUT / f"vsheet_{s+1:02d}.png")
    print(f"wrote vsheet_{s+1:02d}  events {s*EVENTS_PER_SHEET+1}-{s*EVENTS_PER_SHEET+len(chunk)}")

print(f"{len(events)} events -> {OUT}")
