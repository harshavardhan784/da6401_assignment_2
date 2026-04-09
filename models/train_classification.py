"""Training script for Task 1.1 - VGG11 Classification
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import argparse
from datetime import datetime

# Import your modules
from models.classification import VGG11Classifier
from data.pets_dataset import OxfordIIITPetDataset, get_dataloaders


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f"Train Epoch {epoch}")
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{running_loss/(batch_idx+1):.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device, epoch):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Val Epoch {epoch}")
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'Loss': f'{running_loss/(batch_idx+1):.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def main():
    parser = argparse.ArgumentParser(description='Train VGG11 on Oxford-IIIT Pets')
    parser.add_argument('--data_dir', type=str, default='data', help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--dropout_p', type=float, default=0.5, help='Dropout probability')
    parser.add_argument('--use_batch_norm', action='store_true', default=True, help='Use batch norm')
    parser.add_argument('--image_size', type=int, default=224, help='Image size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save models')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataloaders using your existing function
    print("Loading datasets...")
    train_loader, val_loader = get_dataloaders(
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        task="classification",
        image_size=args.image_size,
        num_workers=args.num_workers
    )
    
    # Get number of classes
    num_classes = len(train_loader.dataset.class_names)
    print(f"Number of classes: {num_classes}")
    
    # Create model
    model = VGG11Classifier(
        num_classes=num_classes,
        in_channels=3,
        dropout_p=args.dropout_p,
        use_batch_norm=args.use_batch_norm
    ).to(device)
    
    # Print model architecture summary
    print(f"\nModel Architecture:")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training loop
    best_val_acc = 0.0
    print("\nStarting training...")
    
    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*60}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch+1
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, epoch+1
        )
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, os.path.join(args.save_dir, 'classifier_latest.pth'))
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(checkpoint, os.path.join(args.save_dir, 'classifier_best.pth'))
            print(f"New best model saved! (Val Acc: {best_val_acc:.2f}%)")
        
        # Save periodic checkpoints
        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, os.path.join(args.save_dir, f'classifier_epoch_{epoch+1}.pth'))
    
    print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.2f}%")
    
    # Save final model as classifier.pth (required for Task 4)
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'val_acc': best_val_acc,
        'num_classes': num_classes
    }
    torch.save(final_checkpoint, os.path.join(args.save_dir, 'classifier.pth'))
    print(f"Final model saved as {os.path.join(args.save_dir, 'classifier.pth')}")


if __name__ == "__main__":
    main()