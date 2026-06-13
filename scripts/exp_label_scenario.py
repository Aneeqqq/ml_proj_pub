"""EXPERIMENT: power-fade=blocked label + beam-overlay verification, per scenario.

For a 30s scenario (31/32/33/34):
  1. derive the LOS-envelope power-fade label (fade=blocked) -> scenarioNN_dev_labelled_CLAUDE.csv
  2. locate the tracked car (unit2) each frame via unit1_beam -> pixel x
  3. build before/peak/after verification sheets for every fade event, beam marker drawn

The beam->x direction (low beam = right or left) can differ per scenario (camera/array
mounting), so FLIP is configurable and must be calibrated by eye per scenario.

Usage:
  python -m scripts.exp_label_scenario 33 --flip          # label + verify sheets
  python -m scripts.exp_label_scenario 33 --calib 1 500 900   # just overlay a few frames
"""
import sys
import argparse
from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from src.data.derive_label import derive_blockage_label

ROOT = Path(__file__).resolve().parents[1]
PARAMS = dict(env_win=41, q=0.90, thr_db=-3.0, min_dur=3, deep_db=-4.5)

# scenario -> folder that contains unit1/ (and where unit1_rgb is relative to)
SCN_ROOT = {
    31: ROOT / "scenario31_new" / "scenario31",
    32: ROOT / "scenario32",
    33: ROOT / "scenario33",
    34: ROOT / "scenario34_new" / "scenario34",
}
SCN_CSV = {
    31: "scenario31_dev.csv",
    32: "scenario32_dev.csv",
    33: "scenario33_dev.csv",
    34: "scenario34.csv",
}

try:
    FONT = ImageFont.truetype("arialbd.ttf", 18)
except Exception:
    FONT = ImageFont.load_default()


def load_dev(scn):
    df = pd.read_csv(SCN_ROOT[scn] / SCN_CSV[scn]).reset_index(drop=True)
    return df


def img_path(scn, rgb):
    return SCN_ROOT[scn] / str(rgb).replace("./", "", 1)


def beam_x(beam, bmin, bmax, W, flip):
    frac = (beam - bmin) / (bmax - bmin)
    if flip:
        frac = 1 - frac
    return int(round(frac * W))


def write_label(scn, df):
    pw = derive_blockage_label(df, **PARAMS).reset_index(drop=True)
    out = df.copy()
    out["label_derived"] = pw.values
    dest = SCN_ROOT[scn] / f"scenario{scn}_dev_labelled_CLAUDE.csv"
    out.to_csv(dest, index=False)
    b = (pw == "blocked").sum()
    print(f"  labels: {b}/{len(out)} blocked ({100*b/len(out):.1f}%) -> {dest.name}")
    return pw


def events_rows(pw):
    b = (pw == "blocked").to_numpy()
    evs, i, n = [], 0, len(b)
    while i < n:
        if b[i]:
            j = i
            while j < n and b[j]:
                j += 1
            evs.append((i, j - 1))
            i = j
        else:
            i += 1
    return evs


def calib(scn, frames, flip):
    df = load_dev(scn)
    bmin, bmax = int(df["unit1_beam"].min()), int(df["unit1_beam"].max())
    OUT = ROOT / "outputs" / f"s{scn}_beam"
    OUT.mkdir(parents=True, exist_ok=True)
    # frames given as image 'index' values
    for idx in frames:
        row = df.index[df["index"] == idx]
        if len(row) == 0:
            print(f"  index {idx} not found"); continue
        r = row[0]
        p = img_path(scn, df["unit1_rgb"].iloc[r])
        im = Image.open(p).convert("RGB")
        W, H = im.size
        beam = int(df["unit1_beam"].iloc[r])
        x = beam_x(beam, bmin, bmax, W, flip)
        d = ImageDraw.Draw(im)
        d.line([(x, 0), (x, H)], fill=(255, 0, 0), width=4)
        d.rectangle([x - 70, H - 34, x + 70, H], fill=(255, 0, 0))
        d.text((x - 65, H - 30), f"u2 b={beam}", fill="white", font=FONT)
        op = OUT / f"ov_{idx}.jpg"
        im.save(op)
        print(f"  {op}  (beam={beam}, x={x}/{W}, flip={flip})")


def verify_sheets(scn, df, pw, flip):
    bmin, bmax = int(df["unit1_beam"].min()), int(df["unit1_beam"].max())
    evs = events_rows(pw)
    OUT = ROOT / "outputs" / f"s{scn}_verify"
    OUT.mkdir(parents=True, exist_ok=True)
    THUMB = (440, 248)
    COLS, PER = 3, 4
    PX, PY, TOP = 14, 30, 18
    N = len(df)

    def thumb(r):
        beam = int(df["unit1_beam"].iloc[r])
        num = int(df["index"].iloc[r])
        p = img_path(scn, df["unit1_rgb"].iloc[r])
        if not p.exists():
            return Image.new("RGB", THUMB, "black"), num, beam
        im = Image.open(p).convert("RGB").resize(THUMB)
        x = beam_x(beam, bmin, bmax, THUMB[0], flip)
        ImageDraw.Draw(im).line([(x, 0), (x, THUMB[1])], fill=(255, 0, 0), width=3)
        return im, num, beam

    n_sheets = (len(evs) + PER - 1) // PER
    for s in range(n_sheets):
        chunk = evs[s * PER:(s + 1) * PER]
        W = COLS * THUMB[0] + (COLS + 1) * PX
        H = len(chunk) * (THUMB[1] + PY) + PY
        sheet = Image.new("RGB", (W, H), "white")
        d = ImageDraw.Draw(sheet)
        for r_i, (r0, r1) in enumerate(chunk):
            ev = s * PER + r_i + 1
            peak = (r0 + r1) // 2
            rows = [max(0, r0 - 1), peak, min(N - 1, r1 + 1)]
            for c, (rr, tg) in enumerate(zip(rows, ["before", "PEAK", "after"])):
                x = PX + c * (THUMB[0] + PX)
                y = TOP + r_i * (THUMB[1] + PY)
                im, num, beam = thumb(rr)
                sheet.paste(im, (x, y))
                d.text((x, y - 16), f"ev{ev} {tg} img_{num} b={beam}", fill="black", font=FONT)
        sheet.save(OUT / f"vsheet_{s+1:02d}.png")
    print(f"  {len(evs)} events -> {n_sheets} sheets in {OUT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scn", type=int)
    ap.add_argument("--flip", action="store_true", help="low beam = right (else left)")
    ap.add_argument("--calib", nargs="+", type=int, help="image index values to overlay for calibration")
    args = ap.parse_args()

    df = load_dev(args.scn)
    print(f"scenario {args.scn}: {len(df)} rows, beam {int(df['unit1_beam'].min())}-{int(df['unit1_beam'].max())}, "
          f"{df['seq_index'].nunique()} seqs")

    if args.calib:
        calib(args.scn, args.calib, args.flip)
        return

    pw = write_label(args.scn, df)
    verify_sheets(args.scn, df, pw, args.flip)


if __name__ == "__main__":
    main()
