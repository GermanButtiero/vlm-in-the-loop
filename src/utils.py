

def collate_fn(batch): # PUrpose: define how individual data samples form the dataser are combine into batches when passed to the DtataLoader. THe datasets may have varying shapes and size: iamges, annotations.
        return tuple(zip(*batch)) # if input is batch=[(a1, b1), (a2, b2), (a3, b3)], output is [(a1, a2, a3), (b1, b2, b3)]. * unpacks the tuple into two lists, one for images and one for annotations.

def calculate_mask_iou(pred_mask, gt_mask):
    """Calculate IoU between prediction and ground truth masks"""
    # Convert to binary masks if needed
    if pred_mask.dim() > 2 and pred_mask.shape[0] == 1:
        pred_mask = pred_mask.squeeze(0)
    if pred_mask.max() <= 1.0:
        pred_mask = (pred_mask > 0.5).float()
    
    intersection = (pred_mask * gt_mask).sum().float()
    union = ((pred_mask + gt_mask) > 0).float().sum()#>0 converts any non-zero value to 1
    
    if union == 0:
        return 0.0
    return (intersection / union).item()

