"""GUI for auditing scenario 32 i2v blockage labels against camera frames.

View frames in chronological order (by index), mark blockages as visible/not visible.
Simple navigation: ±1 frame, ±5 frames, bulk label next N frames.

Run: .venv/Scripts/python.exe -m scripts.audit_gui
Saves to outputs/scenario32_audit.json
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parents[1]


class AuditGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Scenario 32 i2v Blockage Audit")
        self.root.geometry("1300x750")

        # Load scenario 32 frames in chronological order
        df = pd.read_csv(ROOT / "data" / "dataset_s2s5s32.csv")
        self.frames = df[df.scenario == 32].sort_values("index").reset_index(drop=True)

        self.audit_path = ROOT / "outputs" / "scenario32_audit.json"
        self.load_audit()
        self.current_idx = 0

        self._build_ui()
        self.show_frame()

    def load_audit(self):
        """Load existing audit results."""
        if self.audit_path.exists():
            with open(self.audit_path) as f:
                self.audit = json.load(f)
        else:
            self.audit = {}

    def save_audit(self):
        """Persist audit results."""
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.audit_path, "w") as f:
            json.dump(self.audit, f, indent=2)

    def _build_ui(self):
        # Top: info + jump
        top = ttk.Frame(self.root)
        top.pack(fill="x", padx=10, pady=5)

        self.info_label = ttk.Label(top, text="", font=("Courier", 10))
        self.info_label.pack(anchor="w")

        jump_frame = ttk.Frame(top)
        jump_frame.pack(anchor="e", pady=5)
        ttk.Label(jump_frame, text="Jump to:").pack(side="left")
        self.jump_entry = ttk.Entry(jump_frame, width=6)
        self.jump_entry.pack(side="left", padx=5)
        ttk.Button(jump_frame, text="Go", command=self.jump).pack(side="left")

        # Image canvas
        self.canvas = tk.Canvas(self.root, bg="gray20", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=10, pady=5)

        # Bottom: navigation
        nav = ttk.Frame(self.root)
        nav.pack(fill="x", padx=10, pady=5)

        ttk.Button(nav, text="◄◄ -5", command=self.back5).pack(side="left", padx=2)
        ttk.Button(nav, text="◄ -1", command=self.back1).pack(side="left", padx=2)
        ttk.Button(nav, text="+1 ►", command=self.fwd1).pack(side="left", padx=2)
        ttk.Button(nav, text="+5 ►►", command=self.fwd5).pack(side="left", padx=2)

        # Labeling
        label_frame = ttk.Frame(self.root)
        label_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(label_frame, text="Mark as:").pack(side="left", padx=5)
        ttk.Button(label_frame, text="Visible", command=lambda: self.mark("visible")).pack(
            side="left", padx=2
        )
        ttk.Button(label_frame, text="Not Visible", command=lambda: self.mark("not_visible")).pack(
            side="left", padx=2
        )
        ttk.Button(label_frame, text="Unclear", command=lambda: self.mark("unclear")).pack(
            side="left", padx=2
        )

        ttk.Label(label_frame, text="Next N frames:").pack(side="left", padx=15)
        self.n_entry = ttk.Entry(label_frame, width=3)
        self.n_entry.insert(0, "5")
        self.n_entry.pack(side="left", padx=2)

        ttk.Button(
            label_frame, text="Mark N as Visible", command=lambda: self.mark_n("visible")
        ).pack(side="left", padx=2)
        ttk.Button(
            label_frame, text="Mark N as Not Visible", command=lambda: self.mark_n("not_visible")
        ).pack(side="left", padx=2)

        # Status
        self.status_label = ttk.Label(self.root, text="", font=("Courier", 9))
        self.status_label.pack(anchor="e", padx=10, pady=5)

    def show_frame(self):
        """Load and display current frame."""
        if self.current_idx >= len(self.frames):
            self.current_idx = len(self.frames) - 1

        row = self.frames.iloc[self.current_idx]
        path = ROOT / row["unit1_rgb"]

        # Load image
        try:
            img = Image.open(path)
            img.thumbnail((1280, 720), Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(img)
            self.canvas.create_image(640, 360, image=self.photo)
        except Exception as e:
            self.canvas.create_text(640, 360, text=f"Error loading {path.name}: {e}", fill="red")

        # Info line
        label_key = str(row.name)
        audit_status = self.audit.get(label_key, "—")
        info = (
            f"Frame {self.current_idx + 1}/{len(self.frames)} | "
            f"Index {int(row['index'])} | Scene {row['seq_uid']} | "
            f"Derived Label: {row['label'].upper()} | Status: {audit_status}"
        )
        self.info_label.config(text=info)

        # Status: count labeled
        labeled_count = len(self.audit)
        self.status_label.config(text=f"Labeled: {labeled_count}/{len(self.frames)}")

    def mark(self, decision: str):
        """Mark current frame."""
        row = self.frames.iloc[self.current_idx]
        self.audit[str(row.name)] = decision
        self.save_audit()
        self.fwd1()

    def mark_n(self, decision: str):
        """Mark next N frames."""
        try:
            n = int(self.n_entry.get())
        except ValueError:
            return
        for i in range(n):
            if self.current_idx + i < len(self.frames):
                row = self.frames.iloc[self.current_idx + i]
                self.audit[str(row.name)] = decision
        self.save_audit()
        self.current_idx += n
        self.show_frame()

    def fwd1(self):
        self.current_idx = min(self.current_idx + 1, len(self.frames) - 1)
        self.show_frame()

    def fwd5(self):
        self.current_idx = min(self.current_idx + 5, len(self.frames) - 1)
        self.show_frame()

    def back1(self):
        self.current_idx = max(self.current_idx - 1, 0)
        self.show_frame()

    def back5(self):
        self.current_idx = max(self.current_idx - 5, 0)
        self.show_frame()

    def jump(self):
        try:
            n = int(self.jump_entry.get())
            self.current_idx = max(0, min(n - 1, len(self.frames) - 1))
            self.jump_entry.delete(0, "end")
            self.show_frame()
        except ValueError:
            pass


if __name__ == "__main__":
    root = tk.Tk()
    app = AuditGUI(root)
    root.mainloop()
