"""
Inference script for DA6401 Assignment 2
Run predictions using trained models
FIXED VERSION - Handles both tuple and dict returns
"""
import argparse
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet
from multitask import MultiTaskPerceptionModel


IMAGE_SIZE = 224
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_and_preprocess_image(image_path):
    """Load and preprocess image for inference"""
    # ImageNet normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    
    return img_tensor


def classify(model_path, image_path):
    """Run classification inference"""
    model = VGG11Classifier(num_classes=37)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    
    img = load_and_preprocess_image(image_path)
    img = img.to(DEVICE)
    
    with torch.no_grad():
        logits = model(img)
        pred = logits.argmax(1).item()
        prob = torch.softmax(logits, dim=1)[0, pred].item()
    
    print(f"Predicted class: {pred}")
    print(f"Confidence: {prob:.4f}")
    
    return pred, prob


def localize(model_path, image_path):
    """Run localization inference"""
    model = VGG11Localizer()
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    
    img = load_and_preprocess_image(image_path)
    img = img.to(DEVICE)
    
    with torch.no_grad():
        bbox = model(img)[0].cpu().numpy()
    
    print(f"Bounding box [cx, cy, w, h]: {bbox}")
    
    return bbox


def segment(model_path, image_path):
    """Run segmentation inference"""
    model = VGG11UNet(num_classes=2)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    
    img = load_and_preprocess_image(image_path)
    img = img.to(DEVICE)
    
    with torch.no_grad():
        logits = model(img)
        mask = logits.argmax(1)[0].cpu().numpy()
    
    print(f"Segmentation mask shape: {mask.shape}")
    print(f"Foreground pixels: {(mask == 1).sum()}")
    
    return mask


def multitask_inference(image_path):
    """Run multi-task inference - FIXED to handle both tuple and dict returns"""
    model = MultiTaskPerceptionModel()
    model.to(DEVICE)
    model.eval()
    
    img = load_and_preprocess_image(image_path)
    img = img.to(DEVICE)
    
    with torch.no_grad():
        outputs = model(img)
        
        # Handle both tuple and dict returns
        if isinstance(outputs, dict):
            # Dict format: {'classification': ..., 'localization': ..., 'segmentation': ...}
            cls_logits = outputs['classification']
            bbox = outputs['localization']
            seg_logits = outputs['segmentation']
        elif isinstance(outputs, tuple) and len(outputs) == 3:
            # Tuple format: (cls_logits, bbox, seg_logits)
            cls_logits, bbox, seg_logits = outputs
        else:
            raise ValueError(f"Unexpected model output type: {type(outputs)}")
        
        pred_class = cls_logits.argmax(1).item()
        pred_prob = torch.softmax(cls_logits, dim=1)[0, pred_class].item()
        pred_bbox = bbox[0].cpu().numpy()
        pred_mask = seg_logits.argmax(1)[0].cpu().numpy()
    
    print(f"Classification: class={pred_class}, confidence={pred_prob:.4f}")
    print(f"Localization: bbox={pred_bbox}")
    print(f"Segmentation: foreground_pixels={(pred_mask == 1).sum()}")
    
    return pred_class, pred_prob, pred_bbox, pred_mask


def main():
    parser = argparse.ArgumentParser(description='Run inference on trained models')
    parser.add_argument('--task', type=str, required=True,
                       choices=['classification', 'localization', 'segmentation', 'multitask'],
                       help='Task to run')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to model checkpoint')
    parser.add_argument('--image_path', type=str, required=True,
                       help='Path to input image')
    args = parser.parse_args()
    
    if args.task == 'classification':
        model_path = args.model_path or 'checkpoints/classifier.pth'
        classify(model_path, args.image_path)
    elif args.task == 'localization':
        model_path = args.model_path or 'checkpoints/localizer.pth'
        localize(model_path, args.image_path)
    elif args.task == 'segmentation':
        model_path = args.model_path or 'checkpoints/unet.pth'
        segment(model_path, args.image_path)
    elif args.task == 'multitask':
        multitask_inference(args.image_path)


if __name__ == '__main__':
    main()