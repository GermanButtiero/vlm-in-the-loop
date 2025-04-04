import json
import torch
from src.dataset import CocoSegmentationDatasetMRCNN
from src.train import train_model, get_model_instance_segmentation
from src.evaluate import calculate_ap
import argparse
from zipfile import ZipFile

def collate_fn(batch):
        return tuple(zip(*batch))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Mask R-CNN model on COCO dataset")
    parser.add_argument("--train", action="store_true", default=False, help="Train the model")
    parser.add_argument("--test", action="store_true", default=False, help="Test the model")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    config = json.load(open("config.json"))

    if args.train:
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