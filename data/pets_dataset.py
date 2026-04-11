"""
Oxford-IIIT Pet Dataset – corrected implementation.

Key fixes vs. the skeleton
──────────────────────────
1. Missing annotation files are filtered out at __init__ time (not at
   __getitem__ time).  This keeps __len__ accurate, ensures DataLoader
   batches are always full, and prevents silent bad data from entering
   training.  'continue' inside __getitem__ is invalid Python anyway.
2. Bounding-box output is in pixel space (cx, cy, w, h in pixels at
   image_size × image_size), as required by the assignment spec.
3. Augmentation pipeline order is guaranteed:
       PIL augmentations  →  ToTensor()  →  Normalize()
4. Mean / std computation accumulates sum / sum-of-squares over all
   pixels (not per-image means) so std is not underestimated.
5. Segmentation masks are resized with NEAREST interpolation so class
   ids are never blended.
"""

import os
import xml.etree.ElementTree as ET

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────────────
# Mean / std helper
# ──────────────────────────────────────────────────────────────────────────────

def compute_mean_std(dataset: Dataset) -> tuple[list, list]:
    """
    Compute channel-wise mean and std over *all pixels* in `dataset`.
    Dataset must return (image_tensor [C,H,W], *anything*).
    Uses sum / sum-of-squares accumulation: Var[X] = E[X^2] - E[X]^2
    """
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    pixel_sum   = torch.zeros(3)
    pixel_sq    = torch.zeros(3)
    pixel_count = 0

    for imgs, _ in tqdm(loader, desc="Computing mean/std"):
        b, c, h, w  = imgs.shape
        imgs_flat    = imgs.view(b, c, -1)           # (B, C, H*W)
        pixel_sum   += imgs_flat.sum(dim=[0, 2])
        pixel_sq    += (imgs_flat ** 2).sum(dim=[0, 2])
        pixel_count += b * h * w

    mean = pixel_sum / pixel_count
    std  = torch.sqrt((pixel_sq / pixel_count) - mean ** 2)
    return mean.tolist(), std.tolist()


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class OxfordIIITPetDataset(Dataset):
    """
    Oxford-IIIT Pet multi-task dataset.

    Parameters
    ----------
    root_dir     : path containing images/ and annotations/
    split        : 'train' | 'val'
    task         : 'classification' | 'localization' | 'segmentation'
    image_size   : spatial size after resizing (default 224)
    transform    : callable | None
    val_size     : fraction held out for validation (stratified)
    random_state : RNG seed for reproducible splits
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        task: str = "classification",
        image_size: int = 224,
        transform=None,
        val_size: float = 0.2,
        random_state: int = 42,
    ):
        self.root_dir   = root_dir
        self.split      = split
        self.task       = task
        self.image_size = image_size
        self.transform  = transform

        self.images_dir  = os.path.join(root_dir, "images")
        self.trimaps_dir = os.path.join(root_dir, "annotations", "trimaps")
        self.xmls_dir    = os.path.join(root_dir, "annotations", "xmls")

        # ── collect all valid image files ────────────────────────────────
        all_files = sorted([
            f for f in os.listdir(self.images_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
            and not f.startswith("._")
        ])

        # ── build class index from the full file list ────────────────────
        # Must be built before filtering so class indices are consistent
        # across tasks and splits.
        self.class_names  = sorted(set(self._cls(f) for f in all_files))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}

        assert len(self.class_names) == 37, (
            f"Expected 37 classes, found {len(self.class_names)}"
        )

        # ── stratified train / val split ─────────────────────────────────
        labels = [self._cls(f) for f in all_files]
        tr, vl = train_test_split(
            all_files,
            test_size=val_size,
            stratify=labels,
            random_state=random_state,
        )
        self.files = tr if split == "train" else vl

        # ── filter out samples with missing annotation files ─────────────
        # Filtering here (not in __getitem__) is critical:
        #   • __len__ stays accurate → DataLoader batches are always full
        #   • __getitem__ stays clean → no None returns or dummy tensors
        #   • Missing files are reported once at startup, not mid-epoch
        before = len(self.files)

        if task == "localization":
            self.files = [
                f for f in self.files
                if os.path.exists(
                    os.path.join(self.xmls_dir, f.rsplit(".", 1)[0] + ".xml")
                )
            ]
        elif task == "segmentation":
            self.files = [
                f for f in self.files
                if os.path.exists(
                    os.path.join(self.trimaps_dir, f.rsplit(".", 1)[0] + ".png")
                )
            ]
        # classification: every image is valid, no annotation file needed

        after = len(self.files)
        if before != after:
            print(
                f"[{task}] {split}: dropped {before - after} samples with "
                f"missing annotations ({after} remaining)"
            )

    # ── private helpers ───────────────────────────────────────────────────

    def _cls(self, fname: str) -> str:
        """'Abyssinian_001.jpg' -> 'Abyssinian'"""
        return fname.rsplit("_", 1)[0]

    def _trimap_path(self, fname: str) -> str:
        return os.path.join(
            self.trimaps_dir, fname.rsplit(".", 1)[0] + ".png"
        )

    def _bbox_pixels(self, fname: str, orig_w: int, orig_h: int) -> torch.Tensor:
        """
        Parse Pascal VOC XML and return [cx, cy, bw, bh] in pixel coordinates
        at the resized resolution (image_size x image_size).

        Pipeline:
          raw pixel coords  ->  normalise by orig size  ->  scale to image_size
        """
        xml_path = os.path.join(
            self.xmls_dir, fname.rsplit(".", 1)[0] + ".xml"
        )
        # No existence check needed — guaranteed to exist by init-time filter

        tree = ET.parse(xml_path)
        root = tree.getroot()
        bb   = root.find(".//bndbox")

        xmin = int(bb.find("xmin").text)
        ymin = int(bb.find("ymin").text)
        xmax = int(bb.find("xmax").text)
        ymax = int(bb.find("ymax").text)

        # Clamp: some annotations slightly exceed image bounds
        xmin = max(0, min(xmin, orig_w))
        ymin = max(0, min(ymin, orig_h))
        xmax = max(0, min(xmax, orig_w))
        ymax = max(0, min(ymax, orig_h))

        # Normalise by original dims then scale to target pixel space
        s  = float(self.image_size)
        cx = ((xmin + xmax) / 2.0) / orig_w * s
        cy = ((ymin + ymax) / 2.0) / orig_h * s
        bw = (xmax - xmin) / orig_w * s
        bh = (ymax - ymin) / orig_h * s

        return torch.tensor([cx, cy, bw, bh], dtype=torch.float32)


    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        fname = self.files[idx]
        img   = Image.open(os.path.join(self.images_dir, fname)).convert("RGB")
        orig_w, orig_h = img.size   # capture before any transform

        if self.transform:
            img_t = self.transform(img)
        else:
            img_t = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
            ])(img)

        # classification
        if self.task == "classification":
            label = self.class_to_idx[self._cls(fname)]
            return img_t, torch.tensor(label, dtype=torch.long)

        # localization
        elif self.task == "localization":
            # XML guaranteed to exist — filtered at init
            bbox = self._bbox_pixels(fname, orig_w, orig_h)
            return img_t, bbox

        # segmentation
        elif self.task == "segmentation":
            # Trimap guaranteed to exist — filtered at init
            tp  = self._trimap_path(fname)
            arr = np.array(Image.open(tp).convert("L"))
            arr = np.clip(arr, 1, 3)            # clamp stray artefacts
            seg = (arr - 1).astype(np.uint8)    # remap: 1->0, 2->1, 3->2

            # NEAREST: never interpolate class ids
            seg_img = Image.fromarray(seg).resize(
                (self.image_size, self.image_size), Image.NEAREST
            )
            mask = torch.from_numpy(np.array(seg_img)).long()
            return img_t, mask

        else:
            raise ValueError(f"Unknown task '{self.task}'")


def get_dataloaders(
    root_dir: str,
    batch_size: int = 16,
    task: str = "classification",
    image_size: int = 224,
    num_workers: int = 4,
    mean: list | None = None,
    std:  list | None = None,
):
    """
    Build train / val DataLoaders for the given task.

    Augmentation order (classification train split)
    
    PIL augmentations  ->  ToTensor()  ->  Normalize()

    • PIL ops (Flip, Rotation, ColorJitter, Affine) require a PIL Image.
    • ToTensor() converts PIL uint8 H×W×C  ->  float32 C×H×W in [0, 1].
    • Normalize() requires a float tensor and must come last.

    Localization / segmentation use only the basic transform because
    geometric augmentation would also need to transform the labels.
    Use albumentations for joint image+label augmentation if desired.

    Parameters
    ----------
    mean, std : pre-computed channel statistics.
                If None they are computed from the training split.
    """

    # compute mean/std if not provided
    if mean is None or std is None:
        base_tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        temp_ds = OxfordIIITPetDataset(
            root_dir=root_dir,
            split="train",
            task="classification",   # only needs images, no annotation files
            image_size=image_size,
            transform=base_tf,
        )
        mean, std = compute_mean_std(temp_ds)
        print(f"Computed mean = {mean}")
        print(f"Computed std  = {std}")

    # base transform: val split and loc/seg train
    basic_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # augmented transform: classification train only
    if task == "classification":
        train_tf = transforms.Compose([
            # PIL-level ops — must come before ToTensor
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            # tensor ops — must come after ToTensor
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        train_tf = basic_tf

    val_tf = basic_tf

    # ── build datasets ────────────────────────────────────────────────────
    train_ds = OxfordIIITPetDataset(
        root_dir, "train", task, image_size, train_tf
    )
    val_ds = OxfordIIITPetDataset(
        root_dir, "val", task, image_size, val_tf
    )

    # ── build loaders ─────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(
        f"[{task}] train={len(train_ds)}  val={len(val_ds)}"
        f"  classes={len(train_ds.class_names)}"
    )
    return train_loader, val_loader, mean, std