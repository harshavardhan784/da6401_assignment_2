"""Inference and evaluation for multi-task model
"""

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

from models.multitask import MultiTaskPerceptionModel


class MultiTaskInference:
    """Inference class for multi-task perception model."""
    
    def __init__(self, device: str = None, image_size: int = 224):
        """
        Initialize inference model.
        
        Args:
            device: Device to run inference on
            image_size: Input image size (default 224 for VGG11)
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.image_size = image_size
        
        # Initialize model - it will auto-download weights from Google Drive
        print("Initializing multi-task model...")
        self.model = MultiTaskPerceptionModel(
            num_breeds=37,
            seg_classes=2,  # Changed from 3 to 2 (background + pet)
            in_channels=3,
            image_size=image_size
        ).to(self.device)
        
        # Model is already in eval mode from __init__
        # No need to call self.model.eval() again
        
        print(f"Model loaded on {self.device}")
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess image."""
        image = Image.open(image_path).convert('RGB')
        original_size = image.size  # (width, height)
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        return input_tensor, original_size, image
    
    def predict(self, image_path: str):
        """
        Run inference on a single image.
        
        Returns:
            Dictionary with classification, localization, and segmentation results
        """
        input_tensor, original_size, original_image = self.preprocess_image(image_path)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
        
        # Classification (breed prediction)
        cls_logits = outputs['classification']
        cls_probs = F.softmax(cls_logits, dim=1)
        predicted_class = torch.argmax(cls_probs, dim=1).item()
        confidence = cls_probs[0, predicted_class].item()
        
        # Localization (bounding box in pixel space - already scaled by model)
        bbox_pixel = outputs['localization'][0].cpu().numpy()  # [cx, cy, w, h] in pixel space
        
        # Scale bounding box to original image size
        scale_x = original_size[0] / self.image_size
        scale_y = original_size[1] / self.image_size
        bbox_original = [
            bbox_pixel[0] * scale_x,
            bbox_pixel[1] * scale_y,
            bbox_pixel[2] * scale_x,
            bbox_pixel[3] * scale_y
        ]
        
        # Segmentation (pixel-wise mask)
        seg_logits = outputs['segmentation'][0]  # [2, H, W]
        seg_mask = torch.argmax(seg_logits, dim=0).cpu().numpy()  # [H, W]
        
        # Resize mask to original image size
        seg_mask_original = np.array(Image.fromarray(seg_mask.astype(np.uint8)).resize(
            original_size, Image.NEAREST
        ))
        
        return {
            'classification': {
                'class_id': predicted_class,
                'confidence': confidence,
                'logits': cls_logits[0].cpu().numpy()
            },
            'localization': {
                'bbox_pixel': bbox_original,  # [cx, cy, w, h]
                'bbox_corners': [
                    bbox_original[0] - bbox_original[2]/2,  # x1
                    bbox_original[1] - bbox_original[3]/2,  # y1
                    bbox_original[0] + bbox_original[2]/2,  # x2
                    bbox_original[1] + bbox_original[3]/2   # y2
                ]
            },
            'segmentation': {
                'mask': seg_mask_original,
                'mask_shape': seg_mask_original.shape
            },
            'original_image': original_image,
            'original_size': original_size
        }
    
    def predict_batch(self, images: torch.Tensor):
        """
        Run inference on a batch of preprocessed images.
        
        Args:
            images: Tensor of shape (B, 3, H, W) - already normalized
            
        Returns:
            Dictionary with batch predictions
        """
        images = images.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(images)
        
        # Classification
        cls_logits = outputs['classification']  # (B, 37)
        cls_probs = F.softmax(cls_logits, dim=1)
        predicted_classes = torch.argmax(cls_probs, dim=1)  # (B,)
        
        # Localization (already in pixel space)
        bboxes = outputs['localization']  # (B, 4)
        
        # Segmentation
        seg_logits = outputs['segmentation']  # (B, 2, H, W)
        seg_masks = torch.argmax(seg_logits, dim=1)  # (B, H, W)
        
        return {
            'classification': predicted_classes,
            'classification_probs': cls_probs,
            'localization': bboxes,
            'segmentation': seg_masks,
            'raw_outputs': outputs
        }
    
    def visualize_results(self, results, save_path=None):
        """Visualize all three task outputs."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        
        # Original image with bounding box
        axes[0, 0].imshow(results['original_image'])
        bbox = results['localization']['bbox_corners']
        rect = patches.Rectangle(
            (bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
            linewidth=2, edgecolor='red', facecolor='none'
        )
        axes[0, 0].add_patch(rect)
        axes[0, 0].set_title(f"Original + BBox\nClass: {results['classification']['class_id']}, "
                            f"Confidence: {results['classification']['confidence']:.2%}")
        axes[0, 0].axis('off')
        
        # Segmentation mask overlay
        axes[0, 1].imshow(results['original_image'])
        mask = results['segmentation']['mask']
        # Create overlay (0=background, 1=pet)
        overlay = np.zeros((*mask.shape, 4))
        overlay[mask == 1] = [0, 1, 0, 0.5]  # Green for pet
        axes[0, 1].imshow(overlay)
        axes[0, 1].set_title("Segmentation Overlay")
        axes[0, 1].axis('off')
        
        # Segmentation mask only
        axes[1, 0].imshow(mask, cmap='gray', vmin=0, vmax=1)
        axes[1, 0].set_title("Segmentation Mask\n(0=background, 1=pet)")
        axes[1, 0].axis('off')
        
        # Results summary
        axes[1, 1].axis('off')
        info_text = (
            f"Classification:\n"
            f"  Class ID: {results['classification']['class_id']}\n"
            f"  Confidence: {results['classification']['confidence']:.2%}\n\n"
            f"Localization:\n"
            f"  Center: ({bbox[0]+bbox[2])/2:.1f}, ({bbox[1]+bbox[3])/2:.1f})\n"
            f"  Size: {bbox[2]-bbox[0]:.1f} × {bbox[3]-bbox[1]:.1f}\n\n"
            f"Segmentation:\n"
            f"  Pet pixels: {np.sum(mask == 1)}\n"
            f"  Coverage: {100*np.sum(mask == 1)/mask.size:.1f}%"
        )
        axes[1, 1].text(0.1, 0.5, info_text,
                       fontsize=11, verticalalignment='center',
                       family='monospace')
        axes[1, 1].set_title("Results Summary")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        plt.show()
    
    def compute_iou(self, pred_box, gt_box):
        """
        Compute IoU between predicted and ground truth boxes.
        
        Args:
            pred_box: [cx, cy, w, h] in pixel space
            gt_box: [cx, cy, w, h] in pixel space
            
        Returns:
            IoU score (float)
        """
        # Convert [cx, cy, w, h] to [x1, y1, x2, y2]
        pred_x1 = pred_box[0] - pred_box[2]/2
        pred_y1 = pred_box[1] - pred_box[3]/2
        pred_x2 = pred_box[0] + pred_box[2]/2
        pred_y2 = pred_box[1] + pred_box[3]/2
        
        gt_x1 = gt_box[0] - gt_box[2]/2
        gt_y1 = gt_box[1] - gt_box[3]/2
        gt_x2 = gt_box[0] + gt_box[2]/2
        gt_y2 = gt_box[1] + gt_box[3]/2
        
        # Intersection
        inter_x1 = max(pred_x1, gt_x1)
        inter_y1 = max(pred_y1, gt_y1)
        inter_x2 = min(pred_x2, gt_x2)
        inter_y2 = min(pred_y2, gt_y2)
        
        if inter_x2 < inter_x1 or inter_y2 < inter_y1:
            return 0.0
        
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        
        # Union
        pred_area = pred_box[2] * pred_box[3]
        gt_area = gt_box[2] * gt_box[3]
        union_area = pred_area + gt_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def compute_dice(self, pred_mask, gt_mask):
        """
        Compute Dice coefficient for segmentation.
        
        Args:
            pred_mask: Predicted binary mask (H, W)
            gt_mask: Ground truth binary mask (H, W)
            
        Returns:
            Dice score (float)
        """
        pred_mask = (pred_mask > 0).astype(np.float32)
        gt_mask = (gt_mask > 0).astype(np.float32)
        
        intersection = np.sum(pred_mask * gt_mask)
        union = np.sum(pred_mask) + np.sum(gt_mask)
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        return 2 * intersection / union


def main():
    """Example usage."""
    # Initialize inference
    print("=" * 60)
    print("Multi-Task Perception Model - Inference Demo")
    print("=" * 60)
    
    inferencer = MultiTaskInference()
    
    # Test on an image
    test_image_path = "data/images/Abyssinian_1.jpg"
    
    if os.path.exists(test_image_path):
        print(f"\nRunning inference on: {test_image_path}")
        results = inferencer.predict(test_image_path)
        
        print("\n" + "=" * 60)
        print("RESULTS:")
        print("=" * 60)
        print(f"\nClassification:")
        print(f"  Class ID: {results['classification']['class_id']}")
        print(f"  Confidence: {results['classification']['confidence']:.2%}")
        
        print(f"\nLocalization:")
        print(f"  Bounding Box (center): ({results['localization']['bbox_pixel'][0]:.1f}, "
              f"{results['localization']['bbox_pixel'][1]:.1f})")
        print(f"  Bounding Box (size): {results['localization']['bbox_pixel'][2]:.1f} × "
              f"{results['localization']['bbox_pixel'][3]:.1f}")
        
        print(f"\nSegmentation:")
        print(f"  Mask shape: {results['segmentation']['mask_shape']}")
        print(f"  Pet pixels: {np.sum(results['segmentation']['mask'] == 1)}")
        
        # Visualize
        inferencer.visualize_results(results, save_path="inference_results.png")
        
    else:
        print(f"\nTest image not found: {test_image_path}")
        print("Please ensure the dataset is properly downloaded.")
        print("\nYou can still use the inferencer programmatically:")
        print("  inferencer = MultiTaskInference()")
        print("  results = inferencer.predict('path/to/your/image.jpg')")


if __name__ == "__main__":
    main()