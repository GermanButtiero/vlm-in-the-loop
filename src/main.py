import fiftyone as fo
import fiftyone.zoo as foz
import json
import numpy as np
import matplotlib.patches as patches
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from pycocotools.coco import COCO
from PIL import Image
import os
import cv2  
from pycocotools import mask as coco_mask
from matplotlib import pyplot as plt
from src.dataset import CocoSegmentationDatasetMRCNN
from src.train import train_model
from src.evaluate import evaluate, calculate_ap
import argparse


if "__name__" == "__main__":
    parser = argparse.ArgumentParser(description="Train a Mask R-CNN model on COCO dataset")
    parser.add_argument("--train", action="store_true", default=True, help="Train the model")
    parser.add_argument("--test", action="store_true", default=False, help="Test the model")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = json.load(open("config.json"))

    # Download the COCO-2017 validation split and load it into FiftyOne
    dataset = foz.load_zoo_dataset("coco-2017", split="validation")

    dataset.name = "coco-2017-validation"
    dataset.persistent = True
    print(f'Dataset {dataset.name} loaded with {len(dataset)} samples.')

    dataset_train = foz.load_zoo_dataset("coco-2017", split="train")
    dataset_train.name = "coco-2017-train"
    dataset_train.persistent = True
    print(f'Dataset {dataset_train.name} loaded with {len(dataset_train)} samples.')


    categories_to_keep = config["categories_to_keep"]

    # Split the original training set
    train_dataset_full = CocoSegmentationDatasetMRCNN(
        config["train_image_dir"],
        config["train_annotation_file"],
        categories_to_keep=categories_to_keep,
        min_area_threshold=config["min_area_threshold"]
    )

    train_size = int(0.9 * len(train_dataset_full))
    val_size = len(train_dataset_full) - train_size

    dataset_train, dataset_val = torch.utils.data.random_split(
        train_dataset_full, [train_size, val_size]
    )

    dataset_test = CocoSegmentationDatasetMRCNN(
        config["val_image_dir"],
        config["val_annotation_file"],
        categories_to_keep=categories_to_keep,
        min_area_threshold=config["min_area_threshold"]
    )

    def collate_fn(batch):
        return tuple(zip(*batch))

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

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=config["batch_size"],
        shuffle=False, 
        collate_fn=collate_fn
    )
    if args.train:
        print("Training the model...")
        num_classes = len(categories_to_keep)
        train_model(data_loader_train, data_loader_val, num_classes, num_epochs= config["num_epochs"], device=device)
        
    if args.test:
        print("Evaluating the model...")
        # Load the trained model
        model = torch.load(config["model_path"])
        model.eval()

        #Get mean average precision on test data
        mean_ap = calculate_ap(model, data_loader_test, device, iou_threshold=0.5)
        print(f"Mean Average Precision: {mean_ap:.4f}")