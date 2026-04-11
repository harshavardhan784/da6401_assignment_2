"""Training entrypoint for DA6401 Assignment 2."""

import argparse
import os

import torch
import torch.nn as nn
import wandb
from tqdm import tqdm

from data.pets_dataset import get_dataloaders
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet, CombinedSegmentationLoss
from losses.iou_loss import IoULoss
from losses.localization_loss import LocalizationLoss

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_SIZE = 224


# ── Helpers ───────────────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, epoch, loss, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return (logits.argmax(1) == labels).float().mean().item()


def dice_score(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 3,
               smooth: float = 1e-7) -> float:
    import torch.nn.functional as F
    probs    = F.softmax(pred, dim=1)
    pred_cls = probs.argmax(dim=1)
    scores   = []
    for c in range(num_classes):
        p = (pred_cls == c).float()
        t = (target == c).float()
        inter = (p * t).sum()
        union = p.sum() + t.sum()
        scores.append(((2 * inter + smooth) / (union + smooth)).item())
    return sum(scores) / num_classes


# ── Task trainers ─────────────────────────────────────────────────────────────

def train_classifier(args):
    train_loader, val_loader, mean, std = get_dataloaders(
        args.data_dir, args.batch_size, 'classification', IMAGE_SIZE, args.num_workers
    )

    model = VGG11Classifier(num_classes=37, dropout_p=args.dropout_p).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    wandb.init(project=args.wandb_project, name='classifier', config=vars(args))

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        # ── Train ──
        model.train()
        train_loss, train_acc = 0.0, 0.0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc  += accuracy(logits, labels)

        train_loss /= len(train_loader)
        train_acc  /= len(train_loader)

        # ── Val ──
        model.eval()
        val_loss, val_acc = 0.0, 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                logits = model(imgs)
                val_loss += criterion(logits, labels).item()
                val_acc  += accuracy(logits, labels)
        val_loss /= len(val_loader)
        val_acc  /= len(val_loader)

        scheduler.step()
        print(f"Epoch {epoch:3d}  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}"
              f"  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")
        wandb.log({'epoch': epoch, 'train/loss': train_loss, 'train/acc': train_acc,
                   'val/loss': val_loss, 'val/acc': val_acc})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, val_loss, 'checkpoints/classifier.pth')

    wandb.finish()
    print(f"Best val accuracy: {best_val_acc:.4f}")


def train_localizer(args):
    train_loader, val_loader, mean, std = get_dataloaders(
        args.data_dir, args.batch_size, 'localization', IMAGE_SIZE, args.num_workers
    )

    model = VGG11Localizer(dropout_p=args.dropout_p).to(DEVICE)
    if args.classifier_ckpt:
        model.load_backbone_weights(args.classifier_ckpt)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = LocalizationLoss(mse_weight=0.5, iou_weight=0.5)

    wandb.init(project=args.wandb_project, name='localizer', config=vars(args))

    best_val_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for imgs, bboxes in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
            imgs, bboxes = imgs.to(DEVICE), bboxes.to(DEVICE)
            optimizer.zero_grad()
            preds = model(imgs)
            loss  = criterion(preds, bboxes, IMAGE_SIZE)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, bboxes in val_loader:
                imgs, bboxes = imgs.to(DEVICE), bboxes.to(DEVICE)
                val_loss += criterion(model(imgs), bboxes, IMAGE_SIZE).item()
        val_loss /= len(val_loader)

        scheduler.step()
        print(f"Epoch {epoch:3d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
        wandb.log({'epoch': epoch, 'train/loss': train_loss, 'val/loss': val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss, 'checkpoints/localizer.pth')

    wandb.finish()


def train_segmentation(args):
    train_loader, val_loader, mean, std = get_dataloaders(
        args.data_dir, args.batch_size, 'segmentation', IMAGE_SIZE, args.num_workers
    )

    model = VGG11UNet(num_classes=3).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = CombinedSegmentationLoss(ce_weight=1.0, dice_weight=1.0)

    wandb.init(project=args.wandb_project, name='unet', config=vars(args))

    best_dice = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            preds = model(imgs)
            loss  = criterion(preds, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss, val_dice = 0.0, 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                preds = model(imgs)
                val_loss += criterion(preds, masks).item()
                val_dice += dice_score(preds, masks)
        val_loss /= len(val_loader)
        val_dice /= len(val_loader)

        scheduler.step()
        print(f"Epoch {epoch:3d}  train_loss={train_loss:.4f}"
              f"  val_loss={val_loss:.4f}  val_dice={val_dice:.4f}")
        wandb.log({'epoch': epoch, 'train/loss': train_loss,
                   'val/loss': val_loss, 'val/dice': val_dice})

        if val_dice > best_dice:
            best_dice = val_dice
            save_checkpoint(model, optimizer, epoch, val_loss, 'checkpoints/unet.pth')

    wandb.finish()


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='DA6401 Assignment 2 Training')
    p.add_argument('--task', choices=['classification', 'localization', 'segmentation'],
                   required=True)
    p.add_argument('--data_dir', type=str, required=True, help='Path to Oxford-IIIT Pet root')
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch_size', type=int, default=15)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--dropout_p', type=float, default=0.5)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--wandb_project', type=str, default='da6401-assignment2')
    p.add_argument('--classifier_ckpt', type=str, default=None,
                   help='Path to classifier checkpoint for backbone init (localization task)')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.task == 'classification':
        train_classifier(args)
    elif args.task == 'localization':
        train_localizer(args)
    elif args.task == 'segmentation':
        train_segmentation(args)