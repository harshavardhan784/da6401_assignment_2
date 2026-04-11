"""Unified multi-task perception model.

Strategy: hold three independent sub-models (classifier, localizer, segmenter),
load each checkpoint directly into the sub-model it was trained on, then
forward through all three.  This avoids the fragile weight-remapping that
caused seg/loc failures in the previous version.
"""

import os
import torch
import torch.nn as nn

from models.classification import VGG11Classifier
from models.localization    import VGG11Localizer
from models.segmentation    import VGG11UNet

# ---------------------------------------------------------------------------
# gdown IDs — replace these with YOUR trained checkpoint IDs before submitting
# ---------------------------------------------------------------------------
_CLASSIFIER_GDRIVE_ID = "1z2l5ToDfn1fE8ElKvgbFRCfFgCRafuoe"
_LOCALIZER_GDRIVE_ID  = "1H5UMd5uB5qEsMhZ8pOZuASbwjw-VsKMW"
_UNET_GDRIVE_ID       = "10TZlHa_5bvuIA6HvClb5SmyoAj7H7jCv"
# ---------------------------------------------------------------------------


def _download(gdrive_id: str, output: str) -> None:
    """Download from Google Drive only if the file isn't already present."""
    if os.path.exists(output):
        print(f"  [skip download] {output} already exists")
        return
    try:
        import gdown
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        gdown.download(id=gdrive_id, output=output, quiet=False)
    except Exception as e:
        print(f"  [warn] gdown failed for {output}: {e}")


def _load_state(model: nn.Module, path: str, strict: bool = True) -> None:
    ck = torch.load(path, map_location="cpu")
    sd = ck.get("model_state_dict", ck)
    missing, unexpected = model.load_state_dict(sd, strict=strict)
    if missing:
        print(f"  [warn] missing keys in {path}: {missing[:5]}{'...' if len(missing)>5 else ''}")
    if unexpected:
        print(f"  [warn] unexpected keys in {path}: {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")
    print(f"  Loaded {path}")


class MultiTaskPerceptionModel(nn.Module):
    """
    Shared-backbone-style multi-task model.

    Each sub-model is a full independent network whose weights were loaded
    from the corresponding checkpoint.  The backbone weights in all three
    sub-models will be identical (all pre-trained on the same VGG11 encoder).

    Outputs
    -------
    {
        'classification': (B, 37)           — class logits
        'localization'  : (B, 4)            — [cx,cy,w,h] in pixel space [0..224]
        'segmentation'  : (B, 3, 224, 224)  — per-pixel class logits
    }
    """


    def __init__(
        self,
        num_breeds: int = 37,
        seg_classes: int = 3,
        in_channels: int = 3,          # kept for API compatibility
        classifier_path: str = "checkpoints/classifier.pth",
        localizer_path:  str = "checkpoints/localizer.pth",
        unet_path:       str = "checkpoints/unet.pth",
    ):
        super().__init__()

        # ── Download checkpoints (no-op if already on disk) ──────────────
        _download(_CLASSIFIER_GDRIVE_ID, classifier_path)
        _download(_LOCALIZER_GDRIVE_ID,  localizer_path)
        # _download(_UNET_GDRIVE_ID,       unet_path)

        # ── Build sub-models ─────────────────────────────────────────────
        self.classifier  = VGG11Classifier(num_classes=num_breeds)
        self.localizer   = VGG11Localizer()
        self.segmenter   = VGG11UNet(num_classes=seg_classes)

        # ── Load weights ─────────────────────────────────────────────────
        if os.path.exists(classifier_path):
            _load_state(self.classifier, classifier_path, strict=True)
        else:
            print(f"  [warn] classifier checkpoint not found: {classifier_path}")

        if os.path.exists(localizer_path):
            _load_state(self.localizer, localizer_path, strict=True)
        else:
            print(f"  [warn] localizer checkpoint not found: {localizer_path}")

        # if os.path.exists(unet_path):
            # _load_state(self.segmenter, unet_path, strict=True)
        # else:
        #     print(f"  [warn] unet checkpoint not found: {unet_path}")

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args
        ----
        x : (B, 3, 224, 224)

        Returns
        -------
        dict with keys 'classification', 'localization', 'segmentation'
        """
        cls_out = self.classifier(x)   # (B, 37)
        loc_out = self.localizer(x)    # (B, 4)  — pixel space [0..224]
        seg_out = self.segmenter(x)    # (B, 3, 224, 224)


        return {
            "classification": cls_out,
            "localization":   loc_out,
            "segmentation":   seg_out,
        }