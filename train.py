"""
Training script for DA6401 Assignment 2
Supports training classification, localization, and segmentation models
"""
import os
import argparse
import time
import gc
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import f1_score as sk_f1

from data.pets_dataset import get_dataloaders
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer, LocalizationLoss
from models.segmentation import VGG11UNet, CombinedSegmentationLoss
from losses.iou_loss import IoULoss


# Constants
IMAGE_SIZE = 224
NUM_CLASSES = 37
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed=42):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Trainer:
    def __init__(self, model, task, device=DEVICE,
                 lr=1e-3, weight_decay=1e-4,
                 num_epochs=30, patience=7,
                 warmup_epochs=5, use_amp=False):
        
        self.model = model.to(device)
        self.task = task
        self.device = device
        self.num_epochs = num_epochs
        self.patience = patience
        self.warmup_epochs = warmup_epochs
        self.base_lr = lr
        self.use_amp = use_amp and torch.cuda.is_available()
        
        # Loss functions based on task
        if task == 'classification':
            self.criterion = nn.CrossEntropyLoss()
        elif task == 'localization':
            self.criterion = LocalizationLoss(mse_weight=0.5, iou_weight=0.5)
        elif task == 'segmentation':
            self.criterion = CombinedSegmentationLoss(ce_weight=1.0, dice_weight=1.0, use_focal=True)
        
        # Optimizer with weight decay
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=weight_decay
        )
        
        # Cosine annealing scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_epochs - warmup_epochs, eta_min=lr/100
        )
        
        # Mixed precision scaler
        self.scaler = torch.amp.GradScaler() if self.use_amp else None
        
        self.best_val = float('inf')
        self.es_count = 0
    
    def _warmup_lr(self, epoch):
        """Linear warmup from 0.1*base_lr to base_lr"""
        if epoch <= self.warmup_epochs:
            scale = epoch / self.warmup_epochs
            for g in self.optimizer.param_groups:
                g['lr'] = self.base_lr * max(scale, 0.1)
    
    def _train_epoch(self, loader, epoch):
        self.model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        num_batches = 0
        
        pbar = tqdm(loader, desc=f'Epoch {epoch:3d} [Train]', leave=False)
        
        for batch in pbar:
            if self.task == 'classification':
                images, labels = batch
                images, labels = images.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                
                if self.use_amp:
                    with torch.amp.autocast('cuda'):
                        logits = self.model(images)
                        loss = self.criterion(logits, labels)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)
                    loss.backward()
                    self.optimizer.step()
                
                running_correct += (logits.argmax(1) == labels).sum().item()
                running_total += labels.size(0)
                
            elif self.task == 'localization':
                images, bboxes = batch
                images, bboxes = images.to(self.device), bboxes.to(self.device)
                
                self.optimizer.zero_grad()
                pred = self.model(images)
                loss = self.criterion(pred, bboxes, IMAGE_SIZE)
                loss.backward()
                self.optimizer.step()
                
            elif self.task == 'segmentation':
                images, masks = batch
                images, masks = images.to(self.device), masks.to(self.device)
                
                self.optimizer.zero_grad()
                logits = self.model(images)
                loss = self.criterion(logits, masks)
                loss.backward()
                self.optimizer.step()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            running_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix(loss=f'{loss.item():.4f}')
            
            if num_batches % 50 == 0:
                torch.cuda.empty_cache()
        
        train_loss = running_loss / max(num_batches, 1)
        train_acc = running_correct / max(running_total, 1) if self.task == 'classification' else None
        
        return train_loss, train_acc
    
    @torch.no_grad()
    def _val_epoch(self, loader, epoch):
        self.model.eval()
        running_loss = 0.0
        num_batches = 0
        
        # Metrics accumulation
        all_preds = []
        all_labels = []
        total_iou = 0.0
        total_dice = 0.0
        total_correct = 0
        total_samples = 0
        
        pbar = tqdm(loader, desc=f'Epoch {epoch:3d} [Val  ]', leave=False)
        
        for batch in pbar:
            if self.task == 'classification':
                images, labels = batch
                images, labels = images.to(self.device), labels.to(self.device)
                
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                
                preds = logits.argmax(1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
                total_correct += (logits.argmax(1) == labels).sum().item()
                total_samples += labels.size(0)
                
            elif self.task == 'localization':
                images, bboxes = batch
                images, bboxes = images.to(self.device), bboxes.to(self.device)
                
                pred = self.model(images)
                loss = self.criterion(pred, bboxes, IMAGE_SIZE)
                
                # Compute IoU for monitoring
                iou_loss = IoULoss('none')(pred, bboxes)
                total_iou += (1.0 - iou_loss).sum().item()
                total_samples += bboxes.size(0)
                
            elif self.task == 'segmentation':
                images, masks = batch
                images, masks = images.to(self.device), masks.to(self.device)
                
                logits = self.model(images)
                loss = self.criterion(logits, masks)
                
                # Compute Dice score
                pred_masks = logits.argmax(1)
                intersection = (pred_masks * masks).sum().float()
                dice = (2. * intersection / (pred_masks.sum() + masks.sum() + 1e-7))
                total_dice += dice.item()
                total_samples += 1
            
            running_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=f'{loss.item():.4f}')
        
        val_loss = running_loss / max(num_batches, 1)
        metrics = {}
        
        if self.task == 'classification':
            f1 = sk_f1(all_labels, all_preds, average='macro', zero_division=0)
            metrics['accuracy'] = total_correct / max(total_samples, 1)
            metrics['f1_macro'] = f1
        elif self.task == 'localization':
            metrics['mean_iou'] = total_iou / max(total_samples, 1)
        elif self.task == 'segmentation':
            metrics['dice'] = total_dice / max(total_samples, 1)
        
        return val_loss, metrics
    
    def fit(self, train_loader, val_loader, ckpt_name='best_model'):
        print(f"\n{'='*62}")
        print(f" Task={self.task.upper()}  Device={self.device}  "
              f"Epochs={self.num_epochs}  Patience={self.patience}")
        print(f"{'='*62}\n")
        
        for epoch in range(1, self.num_epochs + 1):
            self._warmup_lr(epoch)
            start_time = time.time()
            
            train_loss, train_acc = self._train_epoch(train_loader, epoch)
            val_loss, metrics = self._val_epoch(val_loader, epoch)
            
            if epoch > self.warmup_epochs:
                self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            epoch_time = time.time() - start_time
            
            # Print results
            acc_str = f"  Train Acc: {train_acc:.4f}" if train_acc is not None else ""
            metrics_str = "  ".join(f"Val {k}: {v:.4f}" for k, v in metrics.items())
            print(f"Epoch {epoch:3d}/{self.num_epochs}  "
                  f"Train Loss: {train_loss:.4f}{acc_str}  "
                  f"Val Loss: {val_loss:.4f}  {metrics_str}  "
                  f"LR: {current_lr:.2e}  ({epoch_time:.1f}s)")
            
            # Checkpoint and early stopping
            if val_loss < self.best_val:
                self.best_val = val_loss
                self.es_count = 0
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'val_loss': val_loss,
                    'metrics': metrics,
                }
                os.makedirs('checkpoints', exist_ok=True)
                torch.save(checkpoint, f'checkpoints/{ckpt_name}.pth')
                print(f"  ✓ Saved checkpoint: checkpoints/{ckpt_name}.pth")
            else:
                self.es_count += 1
            
            if self.es_count >= self.patience:
                print(f"\n⏹ Early stopping at epoch {epoch}")
                break
            
            torch.cuda.empty_cache()
            gc.collect()
        
        print(f"\n✓ Done. Best val loss: {self.best_val:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Train DA6401 Assignment 2 models')
    parser.add_argument('--task', type=str, required=True, 
                       choices=['classification', 'localization', 'segmentation'],
                       help='Task to train')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Path to dataset root directory')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--checkpoint_name', type=str, default=None,
                       help='Name for checkpoint file')
    args = parser.parse_args()
    
    # Set seed
    set_seed(42)
    
    # Get dataloaders
    train_loader, val_loader = get_dataloaders(
        args.data_root, 
        batch_size=args.batch_size, 
        task=args.task
    )
    
    # Create model based on task
    if args.task == 'classification':
        model = VGG11Classifier(num_classes=NUM_CLASSES, dropout_p=0.5, use_batch_norm=True)
        ckpt_name = args.checkpoint_name or 'classifier'
    elif args.task == 'localization':
        model = VGG11Localizer(use_batch_norm=True, freeze_backbone=False)
        # Try to load pretrained backbone
        if os.path.exists('checkpoints/classifier.pth'):
            model.load_backbone_weights('checkpoints/classifier.pth')
        ckpt_name = args.checkpoint_name or 'localizer'
    elif args.task == 'segmentation':
        model = VGG11UNet(num_classes=2, use_batch_norm=True, freeze_backbone=False)
        ckpt_name = args.checkpoint_name or 'unet'
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    trainer = Trainer(
        model, args.task, DEVICE,
        lr=args.lr, weight_decay=1e-4,
        num_epochs=args.epochs, patience=args.patience, 
        warmup_epochs=5, use_amp=True
    )
    trainer.fit(train_loader, val_loader, ckpt_name=ckpt_name)


if __name__ == '__main__':
    main()