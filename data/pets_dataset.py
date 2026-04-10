"""
Oxford-IIIT Pet Dataset implementation for DA6401 Assignment 2
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split


IMAGE_SIZE = 224  # VGG11 standard input size
NUM_CLASSES = 37


class OxfordIIITPetDataset(Dataset):
    """
    Oxford-IIIT Pet Dataset with support for:
    - Classification: breed labels (0-36)
    - Localization: bounding boxes [cx, cy, w, h] in pixel space
    - Segmentation: binary masks (0=background, 1=foreground)
    """
    def __init__(self, root_dir, split='train', task='classification',
                 image_size=IMAGE_SIZE, transform=None, test_size=0.2, random_state=42):
        self.root_dir = root_dir
        self.split = split
        self.task = task
        self.image_size = image_size
        self.transform = transform

        self.images_dir = os.path.join(root_dir, 'images')
        self.trimaps_dir = os.path.join(root_dir, 'annotations', 'trimaps')
        self.bboxes_dir = os.path.join(root_dir, 'annotations', 'bboxes')

        all_files = sorted([f for f in os.listdir(self.images_dir)
                           if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

        # Extract class names from filenames (format: breed_name_xx.jpg)
        self.class_names = sorted(set(self._cls(f) for f in all_files))
        self.class_to_idx = {n: i for i, n in enumerate(self.class_names)}
        
        # Ensure zero-centric labels: mapping from breed name to 0..36
        assert len(self.class_names) == 37, f"Expected 37 classes, got {len(self.class_names)}"

        if split == 'test':
            self.files = all_files
        else:
            labels = [self._cls(f) for f in all_files]
            tr, vl = train_test_split(all_files, test_size=test_size,
                                      random_state=random_state, stratify=labels)
            self.files = tr if split == 'train' else vl

    def _cls(self, f):
        return f.rsplit('_', 1)[0]

    def _tp(self, f):
        return os.path.join(self.trimaps_dir, os.path.splitext(f)[0] + '.png')
    
    def _bbox_path(self, f):
        return os.path.join(self.bboxes_dir, os.path.splitext(f)[0] + '.txt')

    def _bbox_from_trimap(self, arr, orig_w, orig_h):
        """Compute bounding box from trimap (1=pet, 2=background, 3=border)"""
        # Pet pixels are 1 or 3
        mask = ((arr == 1) | (arr == 3)).astype(np.uint8)
        c = np.argwhere(mask)
        if len(c) == 0:
            # Fallback: center box
            return IMAGE_SIZE // 2, IMAGE_SIZE // 2, IMAGE_SIZE, IMAGE_SIZE
        
        y0, x0 = c.min(0)
        y1, x1 = c.max(0)
        
        # Add 10% padding
        dy = max(1, int((y1 - y0) * 0.1))
        dx = max(1, int((x1 - x0) * 0.1))
        h, w = arr.shape
        x0 = max(0, x0 - dx)
        x1 = min(w, x1 + dx)
        y0 = max(0, y0 - dy)
        y1 = min(h, y1 + dy)
        
        # Scale to image_size
        sx = self.image_size / orig_w
        sy = self.image_size / orig_h
        x0s, x1s = x0 * sx, x1 * sx
        y0s, y1s = y0 * sy, y1 * sy
        
        cx = (x0s + x1s) / 2
        cy = (y0s + y1s) / 2
        bw = x1s - x0s
        bh = y1s - y0s
        
        return cx, cy, bw, bh

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img = Image.open(os.path.join(self.images_dir, fname)).convert('RGB')
        ow, oh = img.size
        img_t = self.transform(img) if self.transform else transforms.ToTensor()(img)

        if self.task == 'classification':
            label = self.class_to_idx[self._cls(fname)]
            return img_t, torch.tensor(label, dtype=torch.long)

        elif self.task == 'localization':
            tp = self._tp(fname)
            if os.path.exists(tp):
                arr = np.array(Image.open(tp))
                cx, cy, bw, bh = self._bbox_from_trimap(arr, ow, oh)
            else:
                # Fallback: center box
                cx, cy, bw, bh = IMAGE_SIZE//2, IMAGE_SIZE//2, IMAGE_SIZE, IMAGE_SIZE
            return img_t, torch.tensor([cx, cy, bw, bh], dtype=torch.float32)

        elif self.task == 'segmentation':
            tp = self._tp(fname)
            if not os.path.exists(tp):
                mask = torch.zeros(self.image_size, self.image_size, dtype=torch.long)
            else:
                arr = np.array(Image.open(tp))
                # Convert trimap to binary: 0=background, 1=foreground
                seg = np.zeros_like(arr, dtype=np.uint8)
                seg[(arr == 1) | (arr == 3)] = 1
                seg_img = Image.fromarray(seg).resize((self.image_size, self.image_size), Image.NEAREST)
                mask = torch.from_numpy(np.array(seg_img)).long()
            return img_t, mask

        raise ValueError(f'Unknown task {self.task}')


def get_dataloaders(root_dir, batch_size=16, task='classification',
                    image_size=IMAGE_SIZE, num_workers=4):
    """Create train and validation dataloaders with proper normalization."""
    # ImageNet normalization (standard practice)
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    
    # Training transforms with aggressive augmentation
    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    # Validation transforms (no augmentation)
    val_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    tr = OxfordIIITPetDataset(root_dir, 'train', task, image_size, train_tf)
    vl = OxfordIIITPetDataset(root_dir, 'val', task, image_size, val_tf)
    
    tr_loader = DataLoader(tr, batch_size, shuffle=True, num_workers=num_workers,
                           pin_memory=True, drop_last=True)
    val_loader = DataLoader(vl, batch_size, shuffle=False, num_workers=num_workers,
                            pin_memory=True)
    
    print(f'[{task}] train={len(tr):,}  val={len(vl):,}  classes={len(tr.class_names)}')
    return tr_loader, val_loader