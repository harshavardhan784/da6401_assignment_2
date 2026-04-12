"""Inference script for DA6401 Assignment 2."""

import argparse

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

IMAGE_SIZE = 224
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ImageNet stats (use the same mean/std used during training)
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


def preprocess(image_path: str) -> torch.Tensor:
    tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return tf(Image.open(image_path).convert('RGB')).unsqueeze(0)


def _load(model, path):
    ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt.get('model_state_dict', ckpt))
    model.to(DEVICE).eval()
    return model


# ── Single-task inference ──

def classify(model_path: str, image_path: str):
    from models.classification import VGG11Classifier
    model  = _load(VGG11Classifier(num_classes=37), model_path)
    img    = preprocess(image_path).to(DEVICE)
    with torch.no_grad():
        logits = model(img)
    pred  = logits.argmax(1).item()
    prob  = torch.softmax(logits, dim=1)[0, pred].item()
    print(f"Class: {pred}  Confidence: {prob:.4f}")
    return pred, prob


def localize(model_path: str, image_path: str):
    from models.localization import VGG11Localizer
    model = _load(VGG11Localizer(), model_path)
    img   = preprocess(image_path).to(DEVICE)
    with torch.no_grad():
        bbox = model(img)[0].cpu().numpy()
    print(f"BBox [cx, cy, w, h]: {bbox}")
    return bbox


def segment(model_path: str, image_path: str):
    from models.segmentation import VGG11UNet
    model = _load(VGG11UNet(num_classes=3), model_path)
    img   = preprocess(image_path).to(DEVICE)
    with torch.no_grad():
        mask = model(img).argmax(1)[0].cpu().numpy()
    print(f"Mask shape: {mask.shape}  Unique classes: {np.unique(mask)}")
    return mask


def multitask_inference(image_path: str):
    from multitask import MultiTaskPerceptionModel
    model = MultiTaskPerceptionModel().to(DEVICE).eval()
    img   = preprocess(image_path).to(DEVICE)

    with torch.no_grad():
        out = model(img)

    cls_logits = out['classification']
    bbox       = out['localization']
    seg_logits = out['segmentation']

    pred_class = cls_logits.argmax(1).item()
    pred_prob  = torch.softmax(cls_logits, dim=1)[0, pred_class].item()
    pred_bbox  = bbox[0].cpu().numpy()
    pred_mask  = seg_logits.argmax(1)[0].cpu().numpy()

    print(f"Classification : class={pred_class}  confidence={pred_prob:.4f}")
    print(f"Localization   : bbox={pred_bbox}")
    print(f"Segmentation   : foreground_pixels={(pred_mask == 1).sum()}")
    return pred_class, pred_prob, pred_bbox, pred_mask


# ── CLI ──

def main():
    p = argparse.ArgumentParser(description='Run inference')
    p.add_argument('--task', required=True,
                   choices=['classification', 'localization', 'segmentation', 'multitask'])
    p.add_argument('--image_path', required=True)
    p.add_argument('--model_path', default=None)
    args = p.parse_args()

    defaults = {
        'classification': 'checkpoints/classifier.pth',
        'localization':   'checkpoints/localizer.pth',
        'segmentation':   'checkpoints/unet.pth',
    }

    if args.task == 'classification':
        classify(args.model_path or defaults['classification'], args.image_path)
    elif args.task == 'localization':
        localize(args.model_path or defaults['localization'], args.image_path)
    elif args.task == 'segmentation':
        segment(args.model_path or defaults['segmentation'], args.image_path)
    elif args.task == 'multitask':
        multitask_inference(args.image_path)


if __name__ == '__main__':
    main()