# Quick Reference - Loss Recording & Weight Management

## 3-Step Quick Start

### Step 1: Train
```bash
python train_scale_head_kitti.py --config configs/train_scale_head_kitti.yaml
```
✅ Automatically saves:
- `outputs/scale_head/training_history.json`
- `outputs/scale_head/training_history.csv`
- `outputs/scale_head/checkpoints/scale_head_best.pt`

### Step 2: Plot Curves
```bash
python plot_training_curves.py --history outputs/scale_head/training_history.json
```
✅ Generates 6 PNG files with loss/error/performance curves

### Step 3: Resume if Needed
```bash
# Edit configs/train_scale_head_kitti.yaml
resume_ckpt_path: "outputs/scale_head/checkpoints/scale_head_epoch_010.pt"

# Run training again
python train_scale_head_kitti.py --config configs/train_scale_head_kitti.yaml
```
✅ Continues from epoch 10, appends to existing history files

---

## What Gets Recorded?

Each epoch, these metrics are saved to JSON/CSV:

| Metric | Purpose |
|--------|---------|
| `train_loss` | Training loss (MSE in log-space) |
| `val_mean_error` | Average error on validation set |
| `val_median_error` | Median error (robustness metric) |
| `val_rmse` | Root mean squared error |
| `val_mae` | Mean absolute error |
| `val_fps` | Inference frames per second |
| `val_inference_time` | Time per inference (ms) |
| `learning_rate` | Current LR from scheduler |
| `delta_0.05`, `delta_0.10`, `delta_0.15` | Accuracy within threshold |

---

## What Gets Saved (Checkpoints)?

During training, these files are created:

```
outputs/scale_head/checkpoints/
├── scale_head_best.pt           # Best model (when validation improves)
├── scale_head_epoch_000.pt       # Checkpoint every N epochs
├── scale_head_epoch_005.pt
└── scale_head_epoch_010.pt
```

Each checkpoint contains **only scale_head weights** (not full VGGT model).

---

## Load Pretrained Weights

```python
import torch
from vggt.heads.scale_head import KITTIStereoScaleHead

# Load checkpoint
checkpoint = torch.load('outputs/scale_head/checkpoints/scale_head_best.pt')

# Create model and load
model = KITTIStereoScaleHead(...)
model.load_state_dict(checkpoint)
```

---

## Analyze History with Python

```python
import json
import pandas as pd

# Load JSON
with open('outputs/scale_head/training_history.json') as f:
    data = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(data)

# Find best epoch
best_idx = df['val_mean_error'].idxmin()
print(f"Best epoch: {df.iloc[best_idx]['epoch']}")
print(f"Best error: {df.iloc[best_idx]['val_mean_error']*100:.2f}%")

# Export to Excel
df.to_csv('history.csv', index=False)
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `training_history.json` not created | Make sure training completes (doesn't crash) |
| Resume failed | Checkpoint path must exist and be valid |
| Plot shows all zeros | Make sure JSON is in correct path |
| matplotlib error | `pip install matplotlib` |

---

## File Locations

```
FastVGGT/
├── train_scale_head_kitti.py           # Main training script
├── plot_training_curves.py             # Plotting utility
├── configs/train_scale_head_kitti.yaml # Configuration
└── outputs/scale_head/
    ├── training.log                    # Console output log
    ├── training_history.json           # Metrics (plotting format)
    ├── training_history.csv            # Metrics (Excel format)
    └── checkpoints/
        ├── scale_head_best.pt          # Best checkpoint
        └── scale_head_epoch_*.pt       # Periodic checkpoints
```

---

## Advanced: Custom Training Runs

### Experiment 1: Different Learning Rates
```bash
# Create directory structure
mkdir -p experiments/lr_1e4 experiments/lr_5e4 experiments/lr_1e3

# Train with different LRs
python train_scale_head_kitti.py \
  --config configs/train_scale_head_kitti.yaml \
  --lr 1e-4 \
  --output_dir experiments/lr_1e4

# Compare results
python plot_training_curves.py --history experiments/lr_1e4/training_history.json
python plot_training_curves.py --history experiments/lr_5e4/training_history.json
```

### Experiment 2: Resume and Continue
```bash
# Train for 20 epochs
python train_scale_head_kitti.py \
  --config configs/train_scale_head_kitti.yaml \
  --epochs 20

# Continue for 10 more (total 30)
# Edit config: resume_ckpt_path = "outputs/scale_head/checkpoints/scale_head_best.pt"
python train_scale_head_kitti.py \
  --config configs/train_scale_head_kitti.yaml \
  --epochs 30

# Results automatically merged in JSON/CSV
```

---

## Verify System is Working

```bash
python verify_training_system.py
```

Expected output:
```
✓ PASS   TrainingHistory
✓ PASS   Checkpoints
✓ PASS   Training Script
✓ PASS   Output Directory
✓ PASS   Plot Script

Total: 5/5 tests passed
```

---

## Key Features Implemented

✅ **Real-time loss recording** to JSON/CSV  
✅ **Checkpoint save/load** with scale_head weights  
✅ **Training resume** with history continuation  
✅ **Loss curve plotting** with 6 curve types  
✅ **Delta accuracy metrics** (δ_0.05, δ_0.10, δ_0.15)  
✅ **FPS/inference time** tracking  
✅ **Learning rate logging** for schedule verification  

All features are **fully integrated and tested**.
