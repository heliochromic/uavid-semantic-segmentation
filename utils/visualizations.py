import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def visualize_inference(model, dataloader, device, reversed_mapping, num_samples=4):
    model.eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    all_images = []
    all_masks = []
    all_predictions = []

    with torch.no_grad():
        for batch_images, batch_masks in dataloader:
            batch_images = batch_images.to(device)
            outputs = model(batch_images)
            batch_predictions = outputs.argmax(dim=1)

            all_images.append(batch_images.cpu())
            all_masks.append(batch_masks)
            all_predictions.append(batch_predictions.cpu())

            if sum(len(img) for img in all_images) >= num_samples:
                break

    images = torch.cat(all_images, dim=0)
    masks = torch.cat(all_masks, dim=0)
    predictions = torch.cat(all_predictions, dim=0)

    num_samples = min(num_samples, len(images))
    images = images[:num_samples]
    masks = masks[:num_samples]
    predictions = predictions[:num_samples]
    fig, axes = plt.subplots(num_samples, 4, figsize=(20, num_samples * 4))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for idx in range(num_samples):
        img = images[idx] * std + mean
        img = img.permute(1, 2, 0).numpy().clip(0, 1)
        
        axes[idx, 0].imshow(img)
        axes[idx, 0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[idx, 0].axis('off')
        
        gt_mask = masks[idx].numpy().astype('uint8')
        height, width = gt_mask.shape
        colored_gt_mask = np.zeros((height, width, 3), dtype=np.uint8)
        
        for class_id, rgb in reversed_mapping.items():
            mask = gt_mask == class_id
            colored_gt_mask[mask] = rgb
        
        axes[idx, 1].imshow(colored_gt_mask)
        axes[idx, 1].set_title('Ground Truth', fontsize=12, fontweight='bold')
        axes[idx, 1].axis('off')
        
        pred_mask = predictions[idx].numpy().astype('uint8')
        colored_pred_mask = np.zeros((height, width, 3), dtype=np.uint8)
        
        for class_id, rgb in reversed_mapping.items():
            mask = pred_mask == class_id
            colored_pred_mask[mask] = rgb
        
        axes[idx, 2].imshow(colored_pred_mask)
        axes[idx, 2].set_title('Prediction', fontsize=12, fontweight='bold')
        axes[idx, 2].axis('off')
        
        overlay = img.copy()
        colored_pred_mask_float = colored_pred_mask.astype(float) / 255.0
        overlay = 0.6 * overlay + 0.4 * colored_pred_mask_float
        overlay = np.clip(overlay, 0, 1)
        
        axes[idx, 3].imshow(overlay)
        axes[idx, 3].set_title('Overlay (Image + Pred)', fontsize=12, fontweight='bold')
        axes[idx, 3].axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_best_trial_metrics(experiment_path):
    df = pd.read_csv(experiment_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].plot(df['training_iteration'], df['train_loss'], marker='o', label='Train Loss')
    axes[0].plot(df['training_iteration'], df['loss'], marker='o', label='Val Loss')
    axes[0].set_xlabel('Iteration', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(df['training_iteration'], df['train_iou'], marker='o', label='Train IoU')
    axes[1].plot(df['training_iteration'], df['iou'], marker='o', label='Val IoU')
    axes[1].set_xlabel('Iteration', fontsize=12)
    axes[1].set_ylabel('IoU', fontsize=12)
    axes[1].set_title('IoU', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()