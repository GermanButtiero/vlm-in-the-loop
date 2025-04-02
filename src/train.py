import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torch
from src.evaluate import evaluate

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


def train_model(data_loader_train, data_loader_val, num_classes, num_epochs=None, device=None):
    # Define number of classes (background + your categories)
    num_classes = num_classes + 1  # +1 for background
    
    # Get the model
    model = get_model_instance_segmentation(num_classes)
    model.to(device)
    
    # Define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    
    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    # Number of epochs
    num_epochs = num_epochs
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Train
        model.train()
        total_loss = 0

        for images, targets in data_loader_train:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # Backward pass
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            total_loss += losses.item()
        
        print(f"training loss: {total_loss/len(data_loader_train):.4f}")
        
        # Update learning rate
        lr_scheduler.step()
        
        # Evaluate
        val_loss = evaluate(model, data_loader_val, device)
        
        # Save best model
        if val_loss < best_loss:
            print(f'Current best loss: {best_loss:.4f}, new best loss: {val_loss:.4f}')
            best_loss = val_loss
            torch.save(model.state_dict(), 'models/best_maskrcnn_model.pth')
            print(f"Saved best model with validation loss: {best_loss:.4f}")
        else:
            print(f'Current best loss: {best_loss:.4f}, no improvement since validation loss: {val_loss:.4f}')
    
    print("Training complete!")
    


