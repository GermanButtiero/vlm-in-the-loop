import torch
import os
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

class YOLOv8SegmentationWrapper:
    """Wrapper class for YOLOv8 to match MaskRCNN API patterns"""
    
    def __init__(self, num_classes, pretrained=True, model_size="m"):
        """
        Initialize YOLOv8 model
        """
        self.num_classes = num_classes - 1  # YOLOv8 doesn't use background class
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_size = model_size
        
        # Add this line to track training state
        self.training = False
        
        # Rest of initialization code remains the same
        if pretrained:
            self.model = YOLO(f"yolov8{model_size}-seg.pt")
        else:
            self.model = YOLO(f"yolov8{model_size}-seg.yaml")
        
        self.class_mapping = {}
    
    def to(self, device):
        """Move model to device (for API compatibility)"""
        self.device = device
        return self
    
    def train(self):
        """Set model to training mode"""
        # Update the training flag when train() is called
        self.training = True
        return self
    
    def eval(self):
        """Set model to evaluation mode"""
        # Update the training flag when eval() is called
        self.training = False
        return self
    
    def __call__(self, images, targets=None):
        """
        Forward pass - matches MaskRCNN interface
        
        Args:
            images: List of image tensors
            targets: Optional targets for training
            
        Returns:
            During training: Loss dictionary
            During inference: List of prediction dictionaries with boxes, masks, etc.
        """
        if self.training and targets is not None:
            # Training mode - convert to YOLO format and return loss dict
            # This is handled in train_model function instead
            return {"loss": torch.tensor(0.0, device=self.device)}  # Placeholder
        else:
            # Inference mode
            results = []
            
            for image in images:
                # Convert tensor to format expected by YOLO
                img_np = image.cpu().permute(1, 2, 0).numpy()
                img_np = (img_np * 255).astype(np.uint8)
                
                # Run inference
                yolo_results = self.model.predict(img_np, verbose=False)[0]
                
                # Convert YOLO results to MaskRCNN format
                pred_dict = self._convert_yolo_to_maskrcnn_format(yolo_results, image.shape[-2:])
                results.append(pred_dict)
                
            return results
    
    def _convert_yolo_to_maskrcnn_format(self, yolo_results, img_size):
        """Convert YOLO results to MaskRCNN format"""
        height, width = img_size
        
        # Extract data from YOLO results
        boxes = yolo_results.boxes.xyxy  # x1, y1, x2, y2 format
        scores = yolo_results.boxes.conf
        labels = yolo_results.boxes.cls.int() + 1  # Add 1 to match MaskRCNN's background=0 indexing
        
        # For segmentation
        masks = []
        if hasattr(yolo_results, 'masks') and yolo_results.masks is not None:
            # Get mask array and convert to tensor
            for seg_mask in yolo_results.masks.data:
                # Resize mask to original image size
                mask = torch.nn.functional.interpolate(
                    seg_mask.unsqueeze(0).unsqueeze(0), 
                    size=(height, width), 
                    mode='bilinear', 
                    align_corners=False
                )
                masks.append(mask.squeeze(0))
        
        # Create mask tensor in MaskRCNN format [N, 1, H, W]
        if masks:
            masks_tensor = torch.stack(masks)
        else:
            masks_tensor = torch.zeros((0, 1, height, width), device=self.device)
        
        # Create prediction dictionary (matching MaskRCNN output)
        return {
            'boxes': boxes,
            'labels': labels,
            'scores': scores,
            'masks': masks_tensor
        }

    def load_state_dict(self, path):
        """Load weights - API compatible with PyTorch"""
        if isinstance(path, dict):
            # It's actual state dict (from torch.load)
            # Save temporarily and load
            torch.save(path, 'temp_yolo_weights.pt')
            self.model = YOLO('temp_yolo_weights.pt')
            if os.path.exists('temp_yolo_weights.pt'):
                os.remove('temp_yolo_weights.pt')
        else:
            # It's a path string
            self.model = YOLO(path)
    
    def state_dict(self):
        """Return state dict - API compatible with PyTorch"""
        # YOLO doesn't directly expose state dict, so return model path
        return self.model.model.state_dict()

def get_yolov8_model(num_classes, pretrained=True, model_size="m"):
    """Factory function similar to MaskRCNN's get_model_instance_segmentation"""
    return YOLOv8SegmentationWrapper(num_classes, pretrained, model_size)