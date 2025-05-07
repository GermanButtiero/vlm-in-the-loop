from src.utils import collate_fn, calculate_mask_iou

def simulate_human_feedback(prediction, target, iou_threshold=0.6):
    pred_masks=prediction['masks'].cpu()
    pred_labels=prediction['labels'].cpu()
    gt_masks = target['masks'].cpu()
    gt_labels = target['labels'].cpu()

    # Handle empty masks
    if len(pred_masks) == 0 and len(gt_masks) == 0:
        return True  # Correct: nothing to detect, nothing detected
    if len(pred_masks) == 0 or len(gt_masks) == 0:
        return False  # One is empty, the other is not
    
    # Filter predictions with score > 0.5
    keep = prediction['scores'].cpu() > 0.5
    pred_masks = pred_masks[keep]
    pred_labels = pred_labels[keep]
    
    if len(pred_masks) == 0:
        return False
    
    #Track which predictions have been matched
    pred_matched = [False] * len(pred_masks)
    
    #Check if each GT has a good match of the same class
    used_pred_indices = set()
    for gt_mask, gt_label in zip(gt_masks, gt_labels):
        best_iou = 0
        best_idx = -1
        for idx, (pred_mask, pred_label) in enumerate(zip(pred_masks, pred_labels)):
            if pred_label != gt_label or idx in used_pred_indices:
                continue
            iou = calculate_mask_iou(pred_mask, gt_mask)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_iou >= iou_threshold:
            pred_matched[best_idx] = True
            used_pred_indices.add(best_idx)
        else:
            return False  # GT mask not matched

    # Penalize extra predictions (false positives)
    if not all(pred_matched):
        return False

    return True
    