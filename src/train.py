import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torch
from src.evaluate import evaluate
import os
from src.evaluate import calculate_ap
import numpy as np
import time
from src.yolo_model import get_yolov8_model
import yaml
import os
import cv2
from PIL import Image

def get_model_instance_segmentation(num_classes):
    # Load a pre-trained Mask R-CNN model
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    
    # Get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Get number of input features for mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    
    # Replace mask predictor with new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes)
    
    return model
def get_segmentation_model(model_type, num_classes):
    """
    Create either MaskRCNN or YOLOv8 model
    
    Args:
        model_type: "maskrcnn" or "yolov8"
        num_classes: Number of classes (including background)
    """
    if model_type.lower() == "maskrcnn":
        return get_model_instance_segmentation(num_classes)
    elif model_type.lower() == "yolov8":
        return get_yolov8_model(num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
def convert_dataset_to_yolo_format(dataset, images_dir, labels_dir):
    """
    Convert PyTorch/COCO data to YOLO format
    YOLO expects:
    - Images in images_dir
    - One .txt file per image in labels_dir with format: class_idx x_center y_center width height
    - For segmentation, additional polygon coordinates are added
    """
    # Iterate through dataset
    for i in range(len(dataset)):
        # Get image and annotations
        image, target = dataset[i]
        
        # Save image
        if isinstance(image, torch.Tensor):
            img_np = image.permute(1, 2, 0).numpy() * 255
            img_np = img_np.astype(np.uint8)
            img = Image.fromarray(img_np)
        else:
            img = image
        
        img_path = os.path.join(images_dir, f"image_{i:05d}.jpg")
        img.save(img_path)
        
        # Save labels
        h, w = image.shape[-2:]
        label_path = os.path.join(labels_dir, f"image_{i:05d}.txt")
        
        with open(label_path, 'w') as f:
            boxes = target['boxes']
            labels = target['labels']
            masks = target['masks'] if 'masks' in target else None
            
            for j, (box, label) in enumerate(zip(boxes, labels)):
                # YOLOv8 format: class x_center y_center width height (normalized)
                x1, y1, x2, y2 = box.tolist()
                x_center = (x1 + x2) / 2.0 / w
                y_center = (y1 + y2) / 2.0 / h
                width = (x2 - x1) / w
                height = (y2 - y1) / h
                class_idx = label.item() - 1  # YOLO doesn't use background class
                
                # Write box
                f.write(f"{class_idx} {x_center} {y_center} {width} {height}")
                
                # If segmentation masks are present, add polygon points
                if masks is not None and j < len(masks):
                    mask = masks[j]
                    # Extract contours from mask
                    mask_np = mask.numpy() > 0.5
                    contours, _ = cv2.findContours((mask_np * 255).astype(np.uint8),
                                                  cv2.RETR_EXTERNAL,
                                                  cv2.CHAIN_APPROX_SIMPLE)
                    
                    if contours:
                        # Take largest contour
                        contour = max(contours, key=cv2.contourArea)
                        # Simplify contour
                        epsilon = 0.005 * cv2.arcLength(contour, True)
                        approx = cv2.approxPolyDP(contour, epsilon, True)
                        
                        # Add polygon points (normalized)
                        for point in approx.reshape(-1, 2):
                            x, y = point
                            f.write(f" {x/w} {y/h}")
                
                f.write("\n")
                
def prepare_yolo_dataset(data_loader_train, data_loader_val, num_classes):
    """
    Convert PyTorch data loaders to YOLO format dataset
    Creates necessary files and returns the data.yaml path
    """
    # Extract dataset information
    dataset = data_loader_train.dataset
    
    # Get class names 
    class_names = []
    for i in range(1, num_classes + 1):  # Skip background class
        if hasattr(dataset, 'category_map') and i in dataset.category_map:
            class_names.append(dataset.category_map[i])
        else:
            class_names.append(f"class_{i}")
    
    # Create YOLO dataset structure
    os.makedirs("data/yolo_dataset", exist_ok=True)
    os.makedirs("data/yolo_dataset/images/train", exist_ok=True)
    os.makedirs("data/yolo_dataset/images/val", exist_ok=True)
    os.makedirs("data/yolo_dataset/labels/train", exist_ok=True)
    os.makedirs("data/yolo_dataset/labels/val", exist_ok=True)
    
    # Convert and save training data
    convert_dataset_to_yolo_format(data_loader_train.dataset, 
                                "data/yolo_dataset/images/train", 
                                "data/yolo_dataset/labels/train")
    
    # Convert and save validation data
    convert_dataset_to_yolo_format(data_loader_val.dataset,
                                "data/yolo_dataset/images/val",
                                "data/yolo_dataset/labels/val")
    
    # Create data.yaml file
    data_yaml = {
        'path': os.path.abspath("data/yolo_dataset"),
        'train': 'images/train',
        'val': 'images/val',
        'names': {i: name for i, name in enumerate(class_names)},
        'nc': len(class_names)
    }
    
    with open("data/yolo_dataset/data.yaml", 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    return "data/yolo_dataset/data.yaml"
def train_yolo_model(data_loader_train, data_loader_val, num_classes, num_epochs=None, device=None, img_size=640):
    """
    Train a YOLOv8 model
    """
    # Calculate adaptive epochs based on dataset size
    train_size = len(data_loader_train.dataset)
    base_epochs = 1#10
    min_epochs = 1#5
    
    if num_epochs is None:
        adaptive_epochs = max(min_epochs, int(base_epochs * (1 + train_size/1000)))
        print(f"Using adaptive epochs: {adaptive_epochs} for dataset size {train_size}")
    else:
        adaptive_epochs = num_epochs
    
    # Convert data format for YOLO training
    # YOLO uses YAML files for dataset configuration
    data_yaml = prepare_yolo_dataset(data_loader_train, data_loader_val, num_classes)
    
    # Initialize YOLO model
    model = get_yolov8_model(num_classes, pretrained=True, model_size="m")
    
    # Setup training configuration
    training_args = {
        "data": data_yaml,
        "epochs": adaptive_epochs,
        "imgsz": img_size,
        "batch": data_loader_train.batch_size,
        "device": 0 if device.type == 'cuda' else 'cpu',
        "val": True,
        "save": True,
        "project": "models/yolo_runs",
        "name": f"train_{train_size}_imgs"
    }
    
    print("Starting YOLOv8 training...")
    model.model.train(**training_args)
    
    # Get results
    metrics = {}

# Safely get metrics - handles missing keys gracefully
    try:
        # Check what keys are available
        print("Available metrics keys:", model.model.trainer.metrics.keys())
        
        # Extract metrics safely with fallbacks - handle both list and scalar values
        def safe_extract(metrics_dict, key, default=0):
            """Safely extract a metric value whether it's a list or scalar"""
            value = metrics_dict.get(key, default)
            if isinstance(value, (list, tuple)):
                return value[-1]  # Return the last element if it's a list
            return value  # Return as is if it's a scalar
        
        # Map the keys to the correct ones based on observed metrics keys
        metrics = {
            "train_loss": safe_extract(model.model.trainer.metrics, 'box_loss', 0),
            "val_loss": safe_extract(model.model.trainer.metrics, 'val/loss', 0),
            "train_map": safe_extract(model.model.trainer.metrics, 'metrics/mAP50-95(B)', 0),
            "val_map": safe_extract(model.model.trainer.metrics, 'metrics/mAP50-95(M)', 0),
            "best_epoch": getattr(model.model.trainer, "best_epoch", 0),
            "early_stopped": False
        }
    except Exception as e:
        print(f"Error accessing YOLOv8 metrics: {e}")
        # Provide fallback metrics
        metrics = {
            "train_loss": [0.0],
            "val_loss": [0.0],
            "train_map": [0.0],
            "val_map": [0.0],
            "best_epoch": 0,
            "early_stopped": False
        }
    
    # Save best model to config path location
    best_model_path = f"models/yolo_runs/train_{train_size}_imgs/weights/best.pt"
    if os.path.exists(best_model_path):
        os.makedirs("models", exist_ok=True)
        os.system(f"cp {best_model_path} models/best_yolo_model_data.pt")
    
    print(f"YOLOv8 training complete! Best mAP: {metrics['val_map']}")
    return metrics

def train_model(data_loader_train, data_loader_val, num_classes, num_epochs=None, device=None, model_type="yolov8", yolo_img_size=640):
    """
    Train a segmentation model with adaptive training schedule based on performance.
    
    Args:
        data_loader_train: Training data loader
        data_loader_val: Validation data loader
        num_classes: Number of classes (including background)
        num_epochs: Maximum number of epochs (will be adjusted based on dataset size)
        device: Device to train on
        model_type: "maskrcnn" or "yolov8"
        yolo_img_size: Image size for YOLO training
        
    Returns:
        Dictionary of training metrics
    """
    if model_type.lower() == "yolov8":
        return train_yolo_model(data_loader_train, data_loader_val, num_classes, num_epochs, device, yolo_img_size)
    else:
        # Existing MaskRCNN training code
        # Unchanged from your current implementation
        num_classes = num_classes + 1  # +1 for background
        model = get_model_instance_segmentation(num_classes)
        model.to(device)
    
    # Calculate adaptive max epochs based on dataset size
    train_size = len(data_loader_train.dataset)
    base_epochs = 1
    min_epochs = 1
    
    # Scale epochs based on dataset size (larger datasets need more epochs)
    if num_epochs is None:
        adaptive_epochs = max(min_epochs, int(base_epochs * (1 + train_size/1000)))
        print(f"Using adaptive epochs: {adaptive_epochs} for dataset size {train_size}")
    else:
        adaptive_epochs = num_epochs
    
    # Optimizer setup with weight decay for regularization
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.002, momentum=0.9, weight_decay=0.0005)

    
    # Learning rate scheduler with warm restarts
    #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #    optimizer, T_0=5, T_mult=1, eta_min=1e-6
    #)

    # Early stopping parameters
    patience = 5
    best_val_loss = float('inf')
    best_val_map = 0
    patience_counter = 0
    best_epoch = 0
    
    # Metrics tracking
    metrics_per_epoch = {
        "train_loss": [],
        "val_loss": [],
        "train_map": [],
        "val_map": [],
        "early_stopped": False,
        "stopping_reason": "",
        "best_epoch": 0
    }
    
    start_time = time.time()
    
    for epoch in range(adaptive_epochs):
        epoch_start = time.time()
        print(f"Epoch {epoch+1}/{adaptive_epochs}")
        model.train()
        total_loss = 0
        
        # Training loop
        for images, targets in data_loader_train:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            total_loss += losses.item()
        
        avg_train_loss = total_loss / len(data_loader_train)
        metrics_per_epoch["train_loss"].append(avg_train_loss)

        # Step the learning rate scheduler
        #lr_scheduler.step()
        #current_lr = optimizer.param_groups[0]['lr']
        
        # Validation phase
        model.eval()
        val_loss = evaluate(model, data_loader_val, device)
        metrics_per_epoch["val_loss"].append(val_loss)

        # Calculate mAP for both train and validation sets
        train_map, _ = calculate_ap(model, data_loader_train, device, iou_threshold=0.5)
        val_map, _ = calculate_ap(model, data_loader_val, device, iou_threshold=0.5)
        metrics_per_epoch["train_map"].append(train_map)
        metrics_per_epoch["val_map"].append(val_map)
        
        epoch_time = time.time() - epoch_start
        print(f"[Epoch {epoch+1}] train_loss: {avg_train_loss:.4f}, val_loss: {val_loss:.4f}, "
              f"train_mAP: {train_map:.4f}, val_mAP: {val_map:.4f}, "
              f"time: {epoch_time:.1f}s")#lr: {current_lr:.6f}
        
        # Check overfitting condition - if train mAP is much higher than val mAP
        overfitting_gap = 0.15  # Threshold for overfitting detection
        is_overfitting = (train_map - val_map) > overfitting_gap and epoch > min_epochs
        
        # Improved early stopping with different metrics
        val_improved = False
        
        # Save model if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            val_improved = True
        
        # Also consider mAP improvement as a valid criterion
        if val_map > best_val_map:
            best_val_map = val_map
            val_improved = True
            best_epoch = epoch
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), 'models/best_maskrcnn_model_data.pth')
            print(f"✓ Model saved (best val_mAP: {val_map:.4f})")
        
        # Early stopping logic
        if val_improved:
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter}/{patience} epochs")
        
        # Check stopping conditions
        if patience_counter >= patience and epoch >= min_epochs:
            print(f"Early stopping triggered after {epoch+1} epochs")
            metrics_per_epoch["early_stopped"] = True
            metrics_per_epoch["stopping_reason"] = "patience_exceeded"
            break
            
        if is_overfitting:
            print(f"Stopping training due to overfitting (train_mAP: {train_map:.4f}, val_mAP: {val_map:.4f})")
            metrics_per_epoch["early_stopped"] = True
            metrics_per_epoch["stopping_reason"] = "overfitting_detected"
            break
    
    # If we didn't break early due to early stopping
    if not metrics_per_epoch["early_stopped"]:
        metrics_per_epoch["stopping_reason"] = "max_epochs_reached"
    
    # Record the best epoch and total training time
    metrics_per_epoch["best_epoch"] = best_epoch
    metrics_per_epoch["total_training_time"] = time.time() - start_time
    
    # Ensure the best model is loaded back
    model.load_state_dict(torch.load('models/best_maskrcnn_model_data.pth'))
    
    print(f"Training complete! Best validation mAP: {best_val_map:.4f} at epoch {best_epoch+1}")
    return metrics_per_epoch



