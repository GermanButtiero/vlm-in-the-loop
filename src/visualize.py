
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.patches as patches
from PIL import Image
import cv2

def visualize_bounding_boxes(dataset, idx=0):
    """Visualize an example from the dataset"""
    image, target = dataset[idx]
    
    # Convert image from tensor to numpy
    image_np = image.permute(1, 2, 0).numpy()
    
    # Create figure
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    
    # Plot original image
    ax[0].imshow(image_np)
    ax[0].set_title('Original Image')
    ax[0].axis('off')
    
    # Create a colored mask for visualization
    h, w = image_np.shape[:2]
    colored_mask = np.zeros((h, w, 3), dtype=np.float32)
    
    # Generate random colors for each instance
    masks = target['masks'].numpy()
    n_instances = len(masks)
    
    if n_instances > 0:
        colors = np.random.rand(n_instances, 3)
        
        # For each instance
        for i in range(n_instances):
            mask = masks[i]
            color = colors[i]
            
            # Create colored instance
            for c in range(3):
                colored_mask[:, :, c] += mask * color[c]
            
            # Draw bounding box
            if 'boxes' in target:
                box = target['boxes'][i].numpy().astype(np.int32)
                rect = patches.Rectangle(
                    (box[0], box[1]), box[2] - box[0], box[3] - box[1],
                    linewidth=2, edgecolor=color, facecolor='none'
                )
                ax[1].add_patch(rect)
            
            # Add category name if available
            if 'category_ids' in target:
                cat_id = target['category_ids'][i].item()
                if hasattr(dataset, 'category_map') and cat_id in dataset.category_map:
                    cat_name = dataset.category_map[cat_id]
                else:
                    cat_name = f"Cat {cat_id}"
                
                if 'boxes' in target:
                    box = target['boxes'][i].numpy().astype(np.int32)
                    ax[1].text(
                        box[0], box[1] - 5, cat_name,
                        color='white', fontsize=10, bbox=dict(facecolor=color, alpha=0.8)
                    )
    
    # Normalize the colored mask
    if n_instances > 0:
        max_val = np.max(colored_mask)
        if max_val > 0:
            colored_mask = colored_mask / max_val
    
    # Blend with original image
    alpha = 0.5
    composite = image_np * (1 - alpha) + colored_mask * alpha
    
    # Show result
    ax[1].imshow(composite)
    ax[1].set_title(f'Segmentation ({n_instances} instances)')
    ax[1].axis('off')
    
    plt.tight_layout()
    plt.show()


def visualize_individual_masks(dataset, idx=0):
    """Visualize individual instance masks from the dataset"""
    image, target = dataset[idx]
    
    # Convert image from tensor to numpy
    image_np = image.permute(1, 2, 0).numpy()
    
    # Get masks
    masks = target['masks'].numpy()
    
    # Get category IDs if available
    if 'category_ids' in target:
        category_ids = target['category_ids'].numpy()
        has_categories = True
    else:
        has_categories = False
    
    n_instances = len(masks)
    
    if n_instances == 0:
        print("No instances found in this image")
        plt.imshow(image_np)
        plt.axis('off')
        plt.show()
        return
    
    # Calculate grid size
    n_cols = min(3, n_instances)
    n_rows = (n_instances + n_cols - 1) // n_cols
    
    # Create figure - original image + individual masks
    fig = plt.figure(figsize=(4*n_cols, 4*(n_rows+1)))
    
    # Show original image
    ax = fig.add_subplot(n_rows+1, 1, 1)
    ax.imshow(image_np)
    ax.set_title('Original Image')
    ax.axis('off')
    
    # Show each instance mask
    for i in range(n_instances):
        ax = fig.add_subplot(n_rows+1, n_cols, n_cols+i+1)
        
        # Get mask
        mask = masks[i]
        
        # Create a colored version for better visibility
        colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.float32)
        color = np.random.rand(3)
        for c in range(3):
            colored_mask[:, :, c] = mask * color[c]
        
        # Show mask overlaid on image
        overlay = image_np * 0.7 + colored_mask * 0.3
        
        # Only show mask area
        mask_area = mask > 0
        overlay = np.where(np.repeat(mask_area[:, :, np.newaxis], 3, axis=2), 
                          overlay, 
                          image_np * 0.3)  # Dim non-mask area
        
        ax.imshow(overlay)
        
        # Show category if available
        if has_categories:
            cat_id = category_ids[i]
            if hasattr(dataset, 'category_map') and cat_id in dataset.category_map:
                cat_name = dataset.category_map[cat_id]
            else:
                cat_name = f"Category {cat_id}"
            ax.set_title(f'Instance {i+1}: {cat_name}')
        else:
            ax.set_title(f'Instance {i+1}')
        
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()



def visualize_segmentation_mask(image, prediction, threshold=0.5):
    """
    Create a visualization of the segmentation masks on the image.
    
    Args:
        image: The original image tensor
        prediction: The model prediction with masks
        threshold: Score threshold for showing masks
    
    Returns:
        PIL Image with visualization
    """
    # Convert image tensor to numpy array
    img_np = image.cpu().permute(1, 2, 0).numpy()
    
    # Denormalize if needed
    if img_np.max() <= 1.0:
        img_np = (img_np * 255).astype(np.uint8)
    
    # Create a copy for visualization
    vis_img = img_np.copy()
    
    # Get masks and scores
    masks = prediction['masks'].cpu().squeeze(1).numpy()
    scores = prediction['scores'].cpu().numpy()
    
    # Apply different colors for each mask
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
              (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0)]
    
    for i, (mask, score) in enumerate(zip(masks, scores)):
        if score < threshold:
            continue
            
        # Convert mask to binary
        binary_mask = (mask > 0.5).astype(np.uint8)
        
        # Apply color to mask
        color = colors[i % len(colors)]
        colored_mask = np.zeros_like(vis_img)
        for c in range(3):
            colored_mask[:, :, c] = binary_mask * color[c]
        
        # Blend with original image
        alpha = 0.4
        vis_img = cv2.addWeighted(vis_img, 1, colored_mask, alpha, 0)
        
        # Find contours and draw them
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_img, contours, -1, color, 2)
    
    # Convert back to PIL image
    return Image.fromarray(vis_img)


def plot_learning_curve(metrics_log, use_vlm=False, save_path=None):
    """Create a plot of mean AP vs iteration"""
    plt.figure(figsize=(10, 6))
    plt.plot(metrics_log["iteration"], metrics_log["mean_ap"], 'o-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Mean Average Precision')
    method = "VLM Feedback" if use_vlm else "IoU-based Feedback"
    plt.title(f'Active Learning Performance with {method}')
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.savefig(f'active_learning_curve_{"vlm" if use_vlm else "iou"}.png')
    plt.close()