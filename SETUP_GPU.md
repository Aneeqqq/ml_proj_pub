# GPU Run Setup

How to run the blockage-prediction training on an NVIDIA GPU. The code auto-detects CUDA
(`device = "cuda" if torch.cuda.is_available() else "cpu"`) and enables pinned memory + mixed
precision (AMP) automatically when a GPU is present — so the only real work is installing the
CUDA build of PyTorch.

## 1. Check your GPU / driver
```powershell
nvidia-smi
```
Note the **CUDA Version** shown top-right — that's the *max* your driver supports. Pick the largest
PyTorch CUDA wheel `<=` that:
- driver supports 12.8+  →  use **cu128** (default in `requirements-gpu.txt`)
- driver supports 12.6–12.7  →  change the index URL to **cu126**
- older  →  see https://pytorch.org/get-started/locally/ for a matching torch version

## 2. Create a fresh venv and install GPU deps
From `ML_Proj_Claude/` (PowerShell):
```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements-gpu.txt
```
> If your driver needs cu126, edit the `--index-url` line in `requirements-gpu.txt` to
> `https://download.pytorch.org/whl/cu126` before installing.

## 3. Verify CUDA is visible
```powershell
.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Expect `True` and your GPU name. If `False`, the wheel/driver mismatch — recheck step 1.

## 4. Tune the configs for GPU (optional but recommended)
Edit `configs/camera.yaml` / `configs/radar.yaml` → `train:`
- `batch_size: 32`  (or 64 if VRAM allows)
- `num_workers: 4`  (radar benefits most — its per-frame feature build is CPU-heavy)
- `amp: true`       (already on; ~1.5–2x speedup, lower VRAM)

## 5. Generate splits + run
```powershell
.venv\Scripts\python.exe -m scripts.derive_label     # power-derived label -> scenario31_dev_derived.csv
.venv\Scripts\python.exe -m scripts.make_splits      # writes splits/seq_assignment.csv
.venv\Scripts\python.exe -m scripts.smoke_test       # sanity asserts (no leakage, time guard, shapes)
.venv\Scripts\python.exe -m scripts.fit_radar_norm   # REQUIRED before radar -> outputs/radar_norm.npz
.venv\Scripts\python.exe -m scripts.train_camera     # camera -> outputs/camera_*.pt
.venv\Scripts\python.exe -m scripts.train_radar      # radar  -> outputs/radar_*.pt (uses radar_norm.npz)
.venv\Scripts\python.exe -m scripts.fuse_eval        # camera+radar late fusion -> outputs/fusion_test.json
```
Each train script writes `outputs/<modality>_best.pt` (incl. tuned thresholds), `_history.json`,
and `_test.json` reporting **F1@0.5 AND F1@tuned** per horizon. `fuse_eval` reports camera, radar,
and FUSED test metrics with softmax-over-val-F1 weights. Targets: camera ≈97% F1@t+5, radar ≈93.5%,
**camera+radar ≈97.2%** (see ML_Proj_Vault/paper/results.md).

Fusion options: `--temperature 0.5` (sharpen the near-uniform F1 weights) or `--per-horizon-weights`.

## Notes
- **AMP** is auto-disabled off-CUDA, so the same scripts still run on CPU unchanged.
- The dataset is self-contained under `scenario31_new/`; paths in `configs/data.yaml` are relative
  to `ML_Proj_Claude/`.
- Known TODOs before chasing paper-exact numbers: per-horizon **threshold tuning** (F1@0.5 is
  pessimistic) and train-split radar normalization stats — see ML_Proj_Vault/lessons-learned.md.
- Next phase after both models converge: **camera+radar late fusion** (softmax-over-val-F1 weights).
