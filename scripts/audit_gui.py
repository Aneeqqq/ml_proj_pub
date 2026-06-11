"""GUI for auditing scenario 32 i2v blockage labels against camera frames.

Workflow: step through blocked/clear frames, mark whether the i2v blockage is
*visually apparent* in the camera or happening outside the field of view.

Run: .venv/Scripts/python.exe -m scripts.audit_gui
Saves audit results to outputs/scenario32_audit.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parents[1]


class AuditGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Scenario 32 i2v Blockage Audit")
        self.root.geometry("1200x800")

        # Load dataset
        self.df = pd.read_csv(ROOT / "data" / "dataset_s2s5s32.csv")
        self.s32 = self.df[self.df.scenario == 32].sort_values("index").reset_index(drop=True)
        self.blocked = self.s32[self.s32.label == "blocked"].reset_index(drop=True)
        self.clear = self.s32[self.s32.label == "not_blocked"].sample(
            min(len(self.blocked), 50), random_state=42
        ).reset_index(drop=True)

        # Interleave blocked and clear for balanced audit
        self.frames = []
        for i in range(max(len(self.blocked), len(self.clear))):
            if i < len(self.blocked):
                self.frames.append(("blocked", self.blocked.iloc[i]))
            if i < len(self.clear):
                self.frames.append(("clear", self.clear.iloc[i]))

        self.audit_path = ROOT / "outputs" / "scenario32_audit.json"
        self.load_audit()
        self.current_idx = self.last_labeled + 1 if self.last_labeled >= 0 else 0

        # UI
        self._build_ui()
        self.show_frame()

    def load_audit(self):
        """Load existing audit results or start fresh."""
        if self.audit_path.exists():
            with open(self.audit_path) as f:
                self.audit = json.load(f)
            self.last_labeled = max(
                (i for i, (_, r) in enumerate(self.frames) if r.name in self.audit), default=-1
            )
        else:
            self.audit = {}
            self.last_labeled = -1

    def save_audit(self):
        """Persist audit results."""
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.audit_path, "w") as f:
            json.dump(self.audit, f, indent=2)

    def _build_ui(self):
        # Top: progress and nav
        top = ttk.Frame(self.root)
        top.pack(fill="x", padx=10, pady=10)
        self.progress_label = ttk.Label(top, text="")
        self.progress_label.pack(side="left")

        ttk.Button(top, text="◄ Prev", command=self.prev_frame).pack(side="left", padx=5)
        ttk.Button(top, text="Next ►", command=self.next_frame).pack(side="left", padx=5)

        # Jump to frame
        ttk.Label(top, text="Frame #:").pack(side="left", padx=5)
        self.jump_entry = ttk.Entry(top, width=5)
        self.jump_entry.pack(side="left")
        ttk.Button(top, text="Jump", command=self.jump_frame).pack(side="left", padx=5)

        # Image canvas
        self.canvas = tk.Canvas(self.root, bg="gray20", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=10, pady=10)

        # Bottom: info + buttons
        bottom = ttk.Frame(self.root)
        bottom.pack(fill="x", padx=10, pady=10)

        self.info_label = ttk.Label(bottom, text="")
        self.info_label.pack(anchor="w")

        btn_frame = ttk.Frame(bottom)
        btn_frame.pack(fill="x", pady=10)
        ttk.Button(
            btn_frame, text="✓ Visible Blockage", command=lambda: self.label("visible")
        ).pack(side="left", padx=5)
        ttk.Button(
            btn_frame, text="✗ Not Visible", command=lambda: self.label("not_visible")
        ).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="? Unclear", command=lambda: self.label("unclear")).pack(
            side="left", padx=5
        )
        ttk.Button(btn_frame, text="⊘ Skip", command=self.next_frame).pack(side="left", padx=5)

        # Status
        self.status_label = ttk.Label(bottom, text="")
        self.status_label.pack(anchor="e")

    def show_frame(self):
        """Load and display current frame."""
        if self.current_idx >= len(self.frames):
            self.info_label.config(text="Audit complete!")
            return

        label_type, row = self.frames[self.current_idx]
        path = ROOT / row["unit1_rgb"]

        # Load image, scale to fit canvas
        img = Image.open(path)
        img.thumbnail((1100, 600), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.create_image(550, 300, image=self.photo)

        # Info: label type, frame index, blockage state
        audit_key = str(row.name)
        status = self.audit.get(audit_key, "—")
        info = (
            f"Frame {self.current_idx + 1}/{len(self.frames)} | "
            f"{label_type.upper()} (derived label) | "
            f"Index {int(row['index'])} | Scene {row['seq_uid']} | "
            f"Status: {status}"
        )
        self.info_label.config(text=info)

        self.progress_label.config(
            text=f"Progress: {len(self.audit)}/{len(self.frames)} labeled"
        )
        self.status_label.config(
            text=f"Last labeled: frame {self.last_labeled + 1 if self.last_labeled >= 0 else '—'}"
        )

    def label(self, decision: str):
        """Record audit decision and move to next."""
        _, row = self.frames[self.current_idx]
        self.audit[str(row.name)] = decision
        self.save_audit()
        self.last_labeled = self.current_idx
        self.next_frame()

    def next_frame(self):
        """Move to next frame."""
        self.current_idx = min(self.current_idx + 1, len(self.frames) - 1)
        self.show_frame()

    def prev_frame(self):
        """Move to previous frame."""
        self.current_idx = max(self.current_idx - 1, 0)
        self.show_frame()

    def jump_frame(self):
        """Jump to specific frame number."""
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
