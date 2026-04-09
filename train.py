"""Training entrypoint
"""

"""
Complete Training Pipeline for Oxford-IIIT Pet Dataset
Supports GPU/CPU, multi-threaded data loading, and all three tasks
"""

from models.classification import VGG11Classifier
from data.pets_dataset import get_dataloaders
from models.layers import CustomDropout
from losses.iou_loss import IoULoss
from sklearn.metrics import f1_score as sk_f1
import torch.nn as nn
import torch
import os, time, gc, math
import numpy as np
from sklearn.metrics import f1_score as sk_f1
from typing import Tuple, Optional, Callable, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF

from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from collections import Counter
import wandb
from models.vgg11 import init_weights

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight  # class weights
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Cross entropy loss (no reduction)
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        
        # Get probabilities
        pt = torch.exp(-ce_loss)  # pt = softmax prob of true class
        
        # Focal loss
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class Trainer:
    def __init__(self, model, task, device=DEVICE, class_weights = None,
                 lr=1e-3, weight_decay=1e-4,
                 num_epochs=30, patience=7,
                 warmup_epochs=5,
                 loc_mse_w=0.5):
        self.model        = model.to(device)
        # if torch.cuda.device_count() > 1:
        #     print(f"  Using {torch.cuda.device_count()} GPUs via DataParallel")
        #     self.model = nn.DataParallel(self.model)

        self.task         = task
        self.device       = device
        self.num_epochs   = num_epochs
        self.patience     = patience
        self.warmup_epochs= warmup_epochs
        self.loc_mse_w    = loc_mse_w
        self.base_lr      = lr
        self.class_weights    = class_weights

        # losses — NO label smoothing (hurts small datasets)
        # self.loss_cls = nn.CrossEntropyLoss()
        self.loss_cls = FocalLoss(gamma=2.0)
        self.loss_iou = IoULoss('mean')
        self.loss_mse = nn.MSELoss()
        self.loss_seg = nn.CrossEntropyLoss()

        self.opt = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=weight_decay
        )
        # cosine decay after warmup
        self.sched = optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=num_epochs - warmup_epochs, eta_min=lr/100
        )

        self.best_val  = float('inf')
        self.es_count  = 0

    def _warmup_lr(self, epoch):
        """Linear warmup: scale LR from base_lr/10 → base_lr over warmup_epochs."""
        if epoch <= self.warmup_epochs:
            scale = (epoch / self.warmup_epochs)
            for g in self.opt.param_groups:
                g['lr'] = self.base_lr * max(scale, 0.1)

    def _grad_norms(self):
        total = 0.0
        d     = {}
        for name, p in self.model.named_parameters():
            if p.grad is not None:
                n = p.grad.detach().norm(2).item()
                d[f'grad/{name}'] = n
                total += n**2
        d['grad/total'] = math.sqrt(total)
        return d

    # ── train one epoch ───────────────────────────────────────────────────────
    def _train_epoch(self, loader, epoch, gstep, use_wb):
        self.model.train()
        run_loss = 0.0
        run_corr = 0
        run_tot  = 0
        n        = 0

        pbar = tqdm(loader, desc=f'Ep {epoch:>3} [Train]', leave=False)
        for batch in pbar:
            if self.task == 'classification':
                imgs, labels = batch
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                self.opt.zero_grad()
                logits = self.model(imgs)
                loss   = self.loss_cls(logits, labels)
                loss.backward()
                run_corr += (logits.argmax(1)==labels).sum().item()
                run_tot  += labels.size(0)

            elif self.task == 'localization':
                imgs, bboxes = batch
                imgs, bboxes = imgs.to(self.device), bboxes.to(self.device)
                self.opt.zero_grad()
                pred     = self.model(imgs)
                iou_loss = self.loss_iou(pred, bboxes)
                mse_loss = self.loss_mse(pred, bboxes) / (IMAGE_SIZE**2)
                loss     = self.loc_mse_w*mse_loss + (1-self.loc_mse_w)*iou_loss
                loss.backward()

            elif self.task == 'segmentation':
                imgs, masks = batch
                imgs, masks = imgs.to(self.device), masks.to(self.device)
                self.opt.zero_grad()
                logits = self.model(imgs)
                loss   = self.loss_seg(logits, masks)
                loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            # log grads every 100 steps using the same gstep counter
            if gstep % 100 == 0:
                gd = self._grad_norms()
                if use_wb:
                    wandb.log(gd, step=gstep)

            self.opt.step()
            run_loss += loss.item()
            n        += 1
            gstep    += 1
            pbar.set_postfix(loss=f'{loss.item():.4f}')
            if n % 50 == 0: torch.cuda.empty_cache()

        tr_loss = run_loss / max(n,1)
        tr_acc  = run_corr / max(run_tot,1) if self.task=='classification' else None
        return tr_loss, tr_acc, gstep

    # ── validate ──────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _val_epoch(self, loader, epoch):
        self.model.eval()
        run_loss = 0.0
        corr=tot=iou_s=dice_s=0
        n=0
        all_preds, all_labels = [], []


        pbar = tqdm(loader, desc=f'Ep {epoch:>3} [Val  ]', leave=False)
        for batch in pbar:
            # In _val_epoch, replace the classification block:
            if self.task == 'classification':
                imgs, labels = batch
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                logits = self.model(imgs)
                loss   = self.loss_cls(logits, labels)
                preds_batch = logits.argmax(1).cpu().numpy()
                labels_batch = labels.cpu().numpy()
                all_preds.extend(preds_batch)
                all_labels.extend(labels_batch)
                corr  += (logits.argmax(1)==labels).sum().item()
                tot   += labels.size(0)

            elif self.task == 'localization':
                imgs, bboxes = batch
                imgs, bboxes = imgs.to(self.device), bboxes.to(self.device)
                pred     = self.model(imgs)
                iou_loss = self.loss_iou(pred, bboxes)
                mse_loss = self.loss_mse(pred, bboxes) / (IMAGE_SIZE**2)
                loss     = self.loc_mse_w*mse_loss + (1-self.loc_mse_w)*iou_loss
                iou_s   += 1.0 - self.loss_iou(pred, bboxes).item()

            elif self.task == 'segmentation':
                imgs, masks = batch
                imgs, masks = imgs.to(self.device), masks.to(self.device)
                logits = self.model(imgs)
                loss   = self.loss_seg(logits, masks)
                pm     = logits.argmax(1)
                inter  = (pm*masks).sum().float()
                dice_s+= (2.*inter/(pm.sum()+masks.sum()+1e-7)).item()

            run_loss += loss.item()
            n        += 1
            pbar.set_postfix(loss=f'{loss.item():.4f}')

        vl = run_loss/max(n,1)
        metrics = {}
        if self.task=='classification':
            f1 = sk_f1(all_labels, all_preds, average='macro', zero_division=0)
            metrics['accuracy'] = corr/max(tot,1)
            metrics['f1_macro'] = f1

        elif self.task=='localization':   metrics['mean_iou'] = iou_s/max(n,1)
        elif self.task=='segmentation':   metrics['dice']     = dice_s/max(n,1)
        return vl, metrics

    # ── full loop ─────────────────────────────────────────────────────────────
    def fit(self, tr_loader, vl_loader, ckpt_name='best_model', use_wb=True):
        print(f"\n{'='*62}")
        print(f' Task={self.task.upper()}  Device={self.device}  '
              f'Epochs={self.num_epochs}  Patience={self.patience}')
        print(f"{'='*62}")

        gstep = 0
        for ep in range(1, self.num_epochs+1):
            self._warmup_lr(ep)
            t0 = time.time()

            tr_loss, tr_acc, gstep = self._train_epoch(tr_loader, ep, gstep, use_wb)
            vl_loss, metrics       = self._val_epoch(vl_loader, ep)

            # scheduler steps after warmup
            if ep > self.warmup_epochs:
                self.sched.step()

            lr_now  = self.opt.param_groups[0]['lr']
            elapsed = time.time()-t0

            # ── console print ─────────────────────────────────────────────────
            acc_s = f'  TrAcc:{tr_acc:.4f}' if tr_acc is not None else ''
            met_s = '  '.join(f'Val {k}:{v:.4f}' for k,v in metrics.items())
            print(f'Ep {ep:>3}/{self.num_epochs}  '
                  f'TrLoss:{tr_loss:.4f}{acc_s}  '
                  f'VlLoss:{vl_loss:.4f}  {met_s}  '
                  f'LR:{lr_now:.2e}  ({elapsed:.1f}s)')

            # ── W&B — ALL metrics logged at the SAME gstep ────────────────────
            # This is the fix for the step-conflict warning:
            # grad norms were already logged at gstep values 0,100,200,...
            # Epoch-level metrics are logged at the CURRENT gstep value
            # (end of epoch), so steps are always increasing.
            if use_wb:
                ld = {
                    'epoch':       ep,
                    'train/loss':  tr_loss,
                    'val/loss':    vl_loss,
                    'lr':          lr_now,
                    'epoch_time':  elapsed,
                }
                if tr_acc is not None: ld['train/accuracy'] = tr_acc
                for k,v in metrics.items(): ld[f'val/{k}'] = v
                wandb.log(ld, step=gstep)   # <── same step counter as grad logs

            # ── checkpoint + early stop ───────────────────────────────────────
            if vl_loss < self.best_val:
                self.best_val = vl_loss
                self.es_count = 0
                path = f'/kaggle/working/{ckpt_name}.pth'
                torch.save({'epoch':ep,'model_state_dict':self.model.state_dict(),
                            'val_loss':vl_loss,'metrics':metrics}, path)
                print(f'  ✓ checkpoint → {path}')
            else:
                self.es_count += 1

            if self.es_count >= self.patience:
                print(f'\n⏹ Early stop at epoch {ep}')
                break

            torch.cuda.empty_cache(); gc.collect()

        print(f'\n✅ Done. Best val loss: {self.best_val:.4f}')


if __name__ == "__main__":
    print('\n'+'='*62)
    print(' TASK 1: VGG11 Classification')
    print('='*62)
    DATA_ROOT = "D:\IITM\DL\DL_A2\da6401_assignment_2\data"
    train_cls, val_cls = get_dataloaders(DATA_ROOT, batch_size=16, task='classification')
    num_classes = 37
    USE_WANDB = True
    
    # class_weights = compute_class_weights(train_cls, num_classes, DEVICE)
    # print(class_weights)

    cls_model = VGG11Classifier(num_classes=37, dropout_p=0.5, use_batch_norm=True)
    init_weights(cls_model)
    print(f'Params: {sum(p.numel() for p in cls_model.parameters()):,}')

    if USE_WANDB:
        wandb.init(project='da6401-oxford-pets', name='vgg11_classification',
                config=dict(task='classification', epochs=40, bs=32,
                            lr=5e-4, dropout=0.3, warmup=5, label_smooth=False))

    trainer_cls = Trainer(cls_model, 'classification', DEVICE,
                        lr=5e-4, weight_decay=1e-4,
                        num_epochs=40, patience=10, warmup_epochs=5)
    trainer_cls.fit(train_cls, val_cls, ckpt_name='classifier', use_wb=USE_WANDB)

    if USE_WANDB: wandb.finish()
    print('\n✅ classifier.pth saved')