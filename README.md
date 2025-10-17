# STN_Final_Term — YOLOv11 + Spatial Transformer Network (STN) + SupCon

A research pipeline for chest X-ray abnormality detection built on a **forked Ultralytics YOLO** with:
- an **STN** placed before the backbone (with validation-time identity),
- **Supervised Contrastive Loss (SupCon)** and optional **background pairing**,
- training/validation utilities and debugging callbacks.

---

## 1) Quick start

### Environment (choose one)

**A) Using `requirements.txt`**
```bash
python -m venv .venv
# Windows
.venv\Scripts\Activate.ps1
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

**B) Using Conda `environment.yml`**
```bash
conda env create -f environment.yml
conda activate STN_Final_Term
```

> **PyTorch & CUDA:** If collaborators have different GPUs/driver stacks, install PyTorch separately:
> - CUDA 11.8:
>   ```bash
>   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
>   ```
> - CUDA 12.x (example 12.1):
>   ```bash
>   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
>   ```
> Adjust versions to match your local setup.

---

## 2) Dataset (kept **outside** the repo)

Do **not** commit real medical images or labels. Place your dataset outside the repository, e.g.:

- Windows: `D:\datasets\VinDrCXR\...`
- Linux/Server: `/media/ssd220/xray/VinDrCXR/...`

Point `dataset.yaml` to **absolute** paths:
```yaml
train: D:/datasets/VinDrCXR/images/train
val:   D:/datasets/VinDrCXR/images/val
test:  D:/datasets/VinDrCXR/images/test
# names: { 0: Aortic_enlargement, 1: Atelectasis, ... }
```

`.gitignore` already excludes `data/`, `datasets/`, `images/`, `labels/`, and training artifacts.

---

## 3) Repository layout

```
scripts/
  ├─ main_stn.py                 # main training entry (YOLO + STN + SupCon)
  ├─ build_bgpair_map.py         # optional: build background pairing JSON
  ├─ build_normal_bank.py        # optional: extract normal/abnormal feature banks
  └─ Archive/ ...                # backups
ultralytics/
  ├─ cfg/models/11/yolo11_stn.yaml   # model with STN inserted
  ├─ nn/modules/block.py             # STN/SDTN modules & blocks
  ├─ engine/trainer.py               # customized trainer
  └─ utils/stn_utils.py, stn_pairing.py, loss.py ...
dataset.yaml
requirements.txt | environment.yml
```

---

## 4) Train

**Windows**
```bash
python scripts/main_stn.py ^
  --yaml "C:/path/to/dataset.yaml" ^
  --model "ultralytics/cfg/models/11/yolo11_stn.yaml" ^
  --epochs 61 --batch 8 --imgsz 640 ^
  --output "runs_stn" --device 0 --patience 61 ^
  --freeze_epochs 21 --debug_images
```

**Linux**
```bash
python scripts/main_stn.py   --yaml /media/ssd220/xray/dataset.yaml   --model ultralytics/cfg/models/11/yolo11_stn.yaml   --epochs 61 --batch 8 --imgsz 640   --output runs_stn --device 0 --patience 61   --freeze_epochs 21 --debug_images
```

**Key knobs (may vary by script):**
- `--freeze_epochs`: bypass/freeze STN for the first *N* epochs to keep early mAP stable.
- SupCon args (injected via code): `supcon_on`, `supcon_feat=stn`, `supcon_gain`, `supcon_loss_weight`,
  `supcon_temp`, `supcon_warmup`, `supcon_queue`, `supcon_log`, …
- `DebugImages`: saves **pre/post-STN** images; draws GT/PRED to verify bbox transforms.
- `STNControl`: identity during validation; SVD/translation clamps; progressive warmup.

---

## 5) Resume / Evaluate

Resume from the last checkpoint:
```bash
python scripts/main_stn.py --resume
```

Validate only (if supported in your script):
```bash
python scripts/main_stn.py --yaml ... --model ... --mode val --device 0
```

---

## 6) Background pairing & feature banks (optional)

Build pairing map:
```bash
python scripts/build_bgpair_map.py --yaml D:/datasets/VinDrCXR/dataset.yaml --out scripts/bgpair_map.json
```

Extract normal/abnormal banks:
```bash
python scripts/build_normal_bank.py --yaml D:/datasets/VinDrCXR/dataset.yaml --out_dir scripts/bank_out
```

> Generated `.json`, `.jpg`, `.npy` are ignored by `.gitignore`.

---

## 7) Outputs & logging

- Experiments, logs, and debug images are placed under `runs_stn/` (ignored by Git).
- Line endings are normalized via `.gitattributes` to avoid LF/CRLF warnings.

---

## 8) Branching & contributions

- Primary branch: `main` (protected).  
- Workflow:
  ```bash
  git checkout -b feat/<feature-name>
  # implement changes
  git commit -m "feat: <summary>"
  git push -u origin feat/<feature-name>
  # open a Pull Request -> review -> merge to main
  ```
- A ruleset may require PRs and linear history (unless a user/app is on the bypass list).

---

## 9) License & citation

- **License:** TODO (MIT/Apache-2.0 or keep Private during research).  
- For academic use, please cite this repository and the relevant Ultralytics release/fork.

---

## 10) Ethics & privacy

- **Never** commit patient-identifying data.  
- When sharing publicly, ensure datasets are properly anonymized and permitted by license/policy.

---

## 11) Contact

- Maintainer: **Nguyen Thanh Nhan** — `ntnhan.0705@gmail.com`  
- For bugs/feature requests, please open an Issue.
