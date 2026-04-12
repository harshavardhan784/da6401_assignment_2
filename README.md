# DA6401 Assignment 2 — Visual Perception Pipeline

**WandB Report:** [DA6401_A2 Report](https://wandb.ai/da25s018-iit-madras/da6401-a2/reports/DA6401_A2--VmlldzoxNjQ5NzAxMA?accessToken=jja4sl546dq0jq2jtapbojt9ur8h7u0tf3zav0fbubddc97hex1ogz5idl150zrq)

---

## Overview

A multi-task visual perception pipeline built on the Oxford-IIIT Pet Dataset, implementing classification, object localization, and semantic segmentation using a VGG11 backbone.

---

## Tasks

| Task | Model | Loss | Metric |
|------|-------|------|--------|
| Classification | VGG11Classifier | CrossEntropyLoss | Macro F1 |
| Localization | VGG11Localizer | MSE + IoU | mAP |
| Segmentation | VGG11UNet | CE + Dice | Dice Score |

---

## Project Structure

```
.
├── checkpoints/
│   └── checkpoints.md
├── data/
│   └── pets_dataset.py
├── losses/
│   ├── __init__.py
│   ├── iou_loss.py
│   └── localization_loss.py
├── models/
│   ├── __init__.py
│   ├── classification.py
│   ├── layers.py
│   ├── localization.py
│   ├── multitask.py
│   ├── segmentation.py
│   └── vgg11.py
├── inference.py
├── train.py
└── README.md
```

---

## Setup

```bash
pip install torch torchvision numpy matplotlib scikit-learn wandb gdown tqdm
```

---

## Training

```bash
# Classification
python train.py --task classification --data_dir /path/to/oxford-pets --epochs 30

# Localization (with pretrained backbone)
python train.py --task localization --data_dir /path/to/oxford-pets --epochs 30 \
    --classifier_ckpt checkpoints/classifier.pth

# Segmentation
python train.py --task segmentation --data_dir /path/to/oxford-pets --epochs 30
```

---

## Inference

```bash
# Single task
python inference.py --task classification --image_path img.jpg --model_path checkpoints/classifier.pth
python inference.py --task localization   --image_path img.jpg --model_path checkpoints/localizer.pth
python inference.py --task segmentation  --image_path img.jpg --model_path checkpoints/unet.pth

# Multi-task (downloads checkpoints automatically)
python inference.py --task multitask --image_path img.jpg
```

---

## Key Design Choices

- **CustomDropout** — Inverted dropout implemented from scratch (no `nn.Dropout`). Scales activations by `1/(1-p)` at train time so inference requires no adjustment.
- **BatchNorm** — Placed after every conv layer for training stability and faster convergence.
- **Localization output** — `[cx, cy, w, h]` in pixel space `[0..224]` via sigmoid × 224.
- **IoU Loss** — Custom `nn.Module` supporting `mean` / `sum` reduction, range `[0, 1]`.
- **U-Net skip connections** — Real skip connections from VGG11 encoder blocks (not a separate parallel encoder).
- **MultiTask model** — Three independent sub-models sharing the same VGG11 encoder weights, loaded from individual checkpoints at init.
