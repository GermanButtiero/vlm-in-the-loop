import json
import torch
from src.dataset import CocoSegmentationDatasetMRCNN
from src.train import train_model, get_model_instance_segmentation
from src.evaluate import calculate_ap
import argparse
from zipfile import ZipFile
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

def collate_fn(batch):
        return tuple(zip(*batch))

def calculate_mask_iou(pred_mask, gt_mask):
    """Calculate IoU between prediction and ground truth masks"""
    # Convert to binary masks if needed
    if pred_mask.dim() > 2 and pred_mask.shape[0] == 1:
        pred_mask = pred_mask.squeeze(0)
    if pred_mask.max() <= 1.0:
        pred_mask = (pred_mask > 0.5).float()
    
    intersection = (pred_mask * gt_mask).sum().float()
    union = ((pred_mask + gt_mask) > 0).float().sum()
    
    if union == 0:
        return 0.0
    return (intersection / union).item()

def simulate_human_feedback(prediction, target, iou_threshold=0.6):
    """
    Simulate human feedback by comparing prediction with ground truth masks.
    Returns True if the prediction meets the IoU threshold criteria.
    """
    pred_masks = prediction['masks'].cpu()
    gt_masks = target['masks'].cpu()
    
    # Skip images with no masks
    if len(pred_masks) == 0 or len(gt_masks) == 0:
        return False
    
    # Filter predictions with score > 0.5
    keep = prediction['scores'].cpu() > 0.5
    pred_masks = pred_masks[keep]
    
    if len(pred_masks) == 0:
        return False
    
    # Check if each ground truth has a good match
    all_matched = True
    for gt_mask in gt_masks:
        best_iou = 0
        for pred_mask in pred_masks:
            iou = calculate_mask_iou(pred_mask, gt_mask)
            best_iou = max(best_iou, iou)
        
        if best_iou < iou_threshold:
            all_matched = False
            break
    
    return all_matched

def run_active_learning(config, device, iou_threshold=0.6, iterations=10):
    """
    Run active learning loop with simulated human feedback.
    
    Args:
        config: Configuration dictionary
        device: Torch device to use
        iou_threshold: Minimum IoU threshold for human approval
        iterations: Number of active learning iterations
    """
    # Load the full training dataset
    full_dataset = CocoSegmentationDatasetMRCNN(
        config["train_image_dir"],
        config["train_annotation_file"]
    )
    
    # Create initial split: 20% train, 80% inference pool
    dataset_size = len(full_dataset)
    all_indices = list(range(dataset_size))
    np.random.shuffle(all_indices)  # Randomize
    
    initial_train_size = int(0.2 * dataset_size)
    train_indices = set(all_indices[:initial_train_size])
    inference_indices = set(all_indices[initial_train_size:])
    
    # Prepare test dataset for consistent evaluation
    test_dataset = CocoSegmentationDatasetMRCNN(
        config["val_image_dir"],
        config["val_annotation_file"]
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config["batch_size"],
        shuffle=False, collate_fn=collate_fn
    )
    
    # Initialize metrics log
    metrics_log = {
        "iteration": [],
        "train_size": [],
        "mean_ap": [],
        "ap_per_class": []
    }
    
    print(f"Starting active learning with {initial_train_size} initial training images")
    print(f"Using IoU threshold of {iou_threshold}")
    
    # Active learning iterations
    for iteration in range(iterations):
        print(f"\n===== Iteration {iteration+1}/{iterations} =====")
        print(f"Training set size: {len(train_indices)}")
        print(f"Inference pool size: {len(inference_indices)}")
        
        # Create current training dataset
        train_dataset = torch.utils.data.Subset(full_dataset, list(train_indices))
        
        # Split into train and validation
        train_size = int(0.9 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        
        dataset_train, dataset_val = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
        
        # Create data loaders
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, batch_size=config["batch_size"],
            shuffle=True, collate_fn=collate_fn
        )
        
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, batch_size=config["batch_size"],
            shuffle=False, collate_fn=collate_fn
        )
        
        # Train model
        print("Training model...")
        num_classes = config["num_classes"]
        train_model(data_loader_train, data_loader_val, num_classes, 
                  num_epochs=config["num_epochs"], device=device)
        
        # Load trained model
        model = get_model_instance_segmentation(num_classes=num_classes + 1)  # +1 for background
        model.load_state_dict(torch.load(config["model_path"]))
        model.to(device)
        model.eval()
        
        # Evaluate on test set
        print("Evaluating model...")
        mean_ap, ap_per_class = calculate_ap(model, test_loader, device, iou_threshold=0.5)
        print(f"Mean Average Precision: {mean_ap:.4f}")
        
        # Log metrics
        metrics_log["iteration"].append(iteration+1)
        metrics_log["train_size"].append(len(train_indices))
        metrics_log["mean_ap"].append(float(mean_ap))
        metrics_log["ap_per_class"].append([float(ap) for ap in ap_per_class])
        
        # Save intermediate log
        with open(f"active_learning_metrics_iter_{iteration+1}.json", "w") as f:
            json.dump(metrics_log, f, indent=4)
        
        # If last iteration, break
        if iteration == iterations - 1:
            break
        
        # Run inference on the pool and simulate human feedback
        print("Simulating human feedback...")
        inference_dataset = torch.utils.data.Subset(full_dataset, list(inference_indices))
        inference_loader = torch.utils.data.DataLoader(
            inference_dataset, batch_size=1,  # Process one at a time
            shuffle=False, collate_fn=collate_fn
        )
        
        approved_indices = []
        with torch.no_grad():
            for i, (image, target) in enumerate(tqdm(inference_loader, desc="Evaluating images")):
                original_idx = list(inference_indices)[i]
                
                # Forward pass
                image = [img.to(device) for img in image]
                outputs = model(image)
                
                # Check if prediction meets quality threshold
                if simulate_human_feedback(outputs[0], target[0], iou_threshold):
                    approved_indices.append(original_idx)
        
        print(f"Approved {len(approved_indices)} new images")
        
        # Update indices
        train_indices.update(approved_indices)
        inference_indices = inference_indices.difference(set(approved_indices))
    
    # Save final metrics log
    with open("active_learning_metrics_final.json", "w") as f:
        json.dump(metrics_log, f, indent=4)
    
    # Plot results
    plot_learning_curve(metrics_log)
    
    print("\nActive learning completed!")
    print(f"Final training set size: {len(train_indices)}")
    print(f"Final mean AP: {metrics_log['mean_ap'][-1]:.4f}")
    
    return metrics_log

def plot_learning_curve(metrics_log):
    """Create a plot of mean AP vs iteration"""
    plt.figure(figsize=(10, 6))
    plt.plot(metrics_log["iteration"], metrics_log["mean_ap"], 'o-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Mean Average Precision')
    plt.title('Active Learning Performance')
    plt.grid(True)
    plt.savefig('active_learning_curve.png')
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Mask R-CNN model on COCO dataset")
    parser.add_argument("--train", action="store_true", default=False, help="Train the model")
    parser.add_argument("--test", action="store_true", default=False, help="Test the model")
    parser.add_argument("--active", action="store_true", default=False, help="Run active learning loop")
    parser.add_argument("--iou-threshold", type=float, default=0.6, help="IoU threshold for active learning")
    parser.add_argument("--iterations", type=int, default=10, help="Number of active learning iterations")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    config = json.load(open("config.json"))

    if args.active:
        print("Running active learning loop...")
        run_active_learning(config, device, 
                          iou_threshold=args.iou_threshold,
                          iterations=args.iterations)
    elif args.train:
        # Split the original training set
        train_dataset_full = CocoSegmentationDatasetMRCNN(
            config["train_image_dir"],
            config["train_annotation_file"]
        )

        train_size = int(0.9 * len(train_dataset_full))
        val_size = len(train_dataset_full) - train_size

        dataset_train, dataset_val = torch.utils.data.random_split(
            train_dataset_full, [train_size, val_size]
        )

        data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate_fn
    )

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, batch_size=config["batch_size"],
            shuffle=False,
            collate_fn=collate_fn
        )

        print("Training the model...")
        num_classes = config["num_classes"]
        train_model(data_loader_train, data_loader_val, num_classes, num_epochs= config["num_epochs"], device=device)
        
    if args.test:
        print("Evaluating the model...")

        dataset_test = CocoSegmentationDatasetMRCNN(
            config["val_image_dir"],
            config["val_annotation_file"]
        )
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=config["batch_size"],
            shuffle=False, 
            collate_fn=collate_fn
        )
        num_classes = config["num_classes"] + 1  # +1 for background
    
        model = get_model_instance_segmentation(num_classes=num_classes)
         
        # Load the trained weights
        model.load_state_dict(torch.load(config["model_path"]))
        model.to(device)

        # Get mean average precision on test data
        mean_ap, ap_per_class = calculate_ap(model, data_loader_test, device, iou_threshold=0.5)
        print(f"Mean Average Precision: {mean_ap:.4f}")