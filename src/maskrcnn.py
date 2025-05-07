import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torch
import torch.nn as nn
from src.evaluate import evaluate
import os
from src.evaluate import calculate_ap

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

class MaskRCNN(nn.Module):
    def __init__(self, num_classes, device):
        super(MaskRCNN, self).__init__()
        self.num_classes = num_classes + 1  # +1 for background
        self.model = get_model_instance_segmentation(self.num_classes)
        self.device = device
        self.to(device)  # This now calls nn.Module's to() method
        self.params = [p for p in self.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(self.params, lr=0.005, momentum=0.9, weight_decay=0.0005)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.1)
    
    def forward(self, *args, **kwargs):
        """Forward pass, called when you do model(input)"""
        return self.model(*args, **kwargs)
    
    def train_model(self, data_loader_train, data_loader_val, num_epochs=None, device=None):
        best_loss = float('inf')
        metrics_per_epoch = {
            "train_loss": [],
            "val_loss": [],
            "train_map": [],
            "val_map": []
        }
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            self.train()  # Inherited from nn.Module
            total_loss = 0
            for images, targets in data_loader_train:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = self(images, targets)  # Calls forward method
                losses = sum(loss for loss in loss_dict.values())
                self.optimizer.zero_grad()
                losses.backward()
                self.optimizer.step()
                total_loss += losses.item()
            avg_train_loss = total_loss / len(data_loader_train)
            print(f"training loss: {avg_train_loss:.4f}")
            metrics_per_epoch["train_loss"].append(avg_train_loss)
            
            self.lr_scheduler.step()
            
            # Validation loss
            self.eval()  # Inherited from nn.Module
            val_loss = evaluate(self, data_loader_val, device)
            metrics_per_epoch["val_loss"].append(val_loss)
            
            # Calculate mAP for train and val sets
            train_map, _ = calculate_ap(self, data_loader_train, device, iou_threshold=0.5)
            val_map, _ = calculate_ap(self, data_loader_val, device, iou_threshold=0.5)
            metrics_per_epoch["train_map"].append(train_map)
            metrics_per_epoch["val_map"].append(val_map)
            print(f"train mAP: {train_map:.4f}, val mAP: {val_map:.4f}")
            
            # Save best model
            if val_loss < best_loss:
                best_loss = val_loss
                os.makedirs('models', exist_ok=True)
                torch.save(self.state_dict(), 'models/best_maskrcnn_model_data.pth')
        print("Training complete!")
        return metrics_per_epoch