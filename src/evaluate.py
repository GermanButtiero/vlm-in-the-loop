import torch
import numpy as np

def evaluate(model, data_loader, device):
    total_loss = 0

    with torch.no_grad():
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()
    
    print(f"Validation loss: {total_loss/len(data_loader):.4f}")
    return total_loss/len(data_loader)



def calculate_ap(model, data_loader, device, iou_threshold=0.5):
    """Calculate Average Precision on the test data"""
    
    all_gt_boxes = []
    all_gt_labels = []
    all_gt_masks = []
    all_pred_boxes = []
    all_pred_labels = []
    all_pred_scores = []
    all_pred_masks = []
    
    model.eval()
    with torch.no_grad():
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            
            # Save ground truth for this batch
            for target in targets:
                all_gt_boxes.append(target['boxes'].cpu())
                all_gt_labels.append(target['labels'].cpu())
                all_gt_masks.append(target['masks'].cpu())
            
            # Run inference
            outputs = model(images)
            
            # Save predictions for this batch
            for output in outputs:
                all_pred_boxes.append(output['boxes'].cpu())
                all_pred_labels.append(output['labels'].cpu())
                all_pred_scores.append(output['scores'].cpu())
                all_pred_masks.append(output['masks'].cpu() > 0.5)  # Threshold at 0.5
    
    # Calculate AP for each class
    ap_per_class = {}
    for class_id in range(1, model.num_classes):  # Skip background (0)
        true_positives = 0
        false_positives = 0
        total_gt = 0
        
        # Count total ground truth for this class
        for gt_labels in all_gt_labels:
            total_gt += (gt_labels == class_id).sum().item()
        if total_gt == 0:
            ap_per_class[class_id] = float('nan')  # No ground truth for this class
            continue
        
        # Collect all predictions for this class
        all_class_scores = []
        all_class_matched = []
        
        for i in range(len(all_pred_boxes)):
            pred_boxes = all_pred_boxes[i]
            pred_labels = all_pred_labels[i]
            pred_scores = all_pred_scores[i]
            pred_masks = all_pred_masks[i]
            
            gt_boxes = all_gt_boxes[i]
            gt_labels = all_gt_labels[i]
            gt_masks = all_gt_masks[i]
            
            # Find predictions of this class
            class_indices = (pred_labels == class_id).nonzero().flatten()
            
            # Get GT instances of this class
            gt_class_indices = (gt_labels == class_id).nonzero().flatten()
            
            # Mark GT as already matched or not
            gt_matched = [False] * len(gt_class_indices)
            
            # For each prediction of this class
            for idx in class_indices:
                pred_box = pred_boxes[idx].unsqueeze(0)
                pred_score = pred_scores[idx].item()
                pred_mask = pred_masks[idx]
                
                all_class_scores.append(pred_score)
                
                # Check if matches any GT
                if len(gt_class_indices) == 0:
                    # No GT, so it's a false positive
                    all_class_matched.append(0)
                    continue
                
                # Calculate IoU with GT masks
                max_iou = 0
                max_idx = -1
                
                for j, gt_idx in enumerate(gt_class_indices):
                    if gt_matched[j]:
                        continue
                    
                    gt_mask = gt_masks[gt_idx]
                    
                    # Calculate mask IoU
                    intersection = (pred_mask & gt_mask).sum().float()
                    union = (pred_mask | gt_mask).sum().float()
                    iou = intersection / union if union > 0 else 0
                    
                    if iou > max_iou:
                        max_iou = iou
                        max_idx = j
                
                # Check if IoU is above threshold
                if max_iou >= iou_threshold and max_idx >= 0:
                    gt_matched[max_idx] = True
                    all_class_matched.append(1)  # True positive
                else:
                    all_class_matched.append(0)  # False positive
        
        # Sort by confidence score
        if all_class_scores:
            scores_and_matched = list(zip(all_class_scores, all_class_matched))
            scores_and_matched.sort(key=lambda x: x[0], reverse=True)
            all_class_matched = [x[1] for x in scores_and_matched]
            
            # Calculate precision and recall
            precisions = []
            recalls = []
            tp_count = 0
            fp_count = 0
            
            for matched in all_class_matched:
                if matched:
                    tp_count += 1
                else:
                    fp_count += 1
                
                precision = tp_count / (tp_count + fp_count)
                recall = tp_count / total_gt
                
                precisions.append(precision)
                recalls.append(recall)
            
            # Calculate AP using the 11-point interpolation
            ap = 0
            for t in range(11):  # 11 points: 0, 0.1, 0.2, ..., 1.0
                recall_threshold = t / 10
                max_precision = 0
                
                for i in range(len(recalls)):
                    if recalls[i] >= recall_threshold:
                        max_precision = max(max_precision, precisions[i])
                
                ap += max_precision / 11
            
            ap_per_class[class_id] = ap
        else:
            ap_per_class[class_id] = 0.0
    
    # Calculate mAP
    valid_aps = [ap for ap in ap_per_class.values() if not np.isnan(ap)]
    mAP = sum(valid_aps) / len(valid_aps) if valid_aps else 0
    
    # Fix here: Convert the dictionary to match the expected format
    # Instead of returning class IDs as keys, create a dictionary with class names mapped to values
    class_names = {
        1: "book",
        2: "bird",
        3: "stop sign",
        4: "zebra"
    }
    
    # Create a proper dictionary with class names and their actual AP values
    formatted_ap = {}
    for class_id, ap_value in ap_per_class.items():
        class_name = class_names.get(class_id, f"class_{class_id}")
        formatted_ap[class_name] = float(ap_value)  # Ensure it's a native Python float
    
    return mAP, formatted_ap

