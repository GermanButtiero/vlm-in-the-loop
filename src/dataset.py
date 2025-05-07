
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

class CocoSegmentationDatasetMRCNN(Dataset):
    def __init__(self, image_dir, seg_annotation_file, categories_to_keep=None):
        self.image_dir = image_dir
        self.coco_seg = COCO(seg_annotation_file)
        self.categories_to_keep = categories_to_keep  # Optional filter
        
        # Get all image IDs from the dataset
        all_image_ids = list(self.coco_seg.imgs.keys())
        
        # Filter to only include images that exist in the directory
        self.image_ids = []
        for img_id in all_image_ids:
            img_info = self.coco_seg.loadImgs(img_id)[0]
            img_path = os.path.join(image_dir, img_info["file_name"])
            if os.path.exists(img_path):
                self.image_ids.append(img_id)
        
        print(f"Dataset contains {len(self.image_ids)} valid images out of {len(all_image_ids)} in annotations")
        
        # Create mappings for COCO category IDs to sequential class indices
        self.category_map = {}
        self.category_id_to_idx = {}  # Map COCO category IDs to sequential indices
        
        # Get all category IDs from the dataset or filter to only the ones we want
        if self.categories_to_keep:
            cat_ids = [cat_id for cat_id in self.categories_to_keep if self.coco_seg.loadCats(cat_id)]
        else:
            cat_ids = self.coco_seg.getCatIds()
        
        # Create mapping from category ID to sequential index (starting from 1)
        for idx, cat_id in enumerate(cat_ids, 1):  # Start from 1, as 0 is background
            self.category_id_to_idx[cat_id] = idx
            cat_info = self.coco_seg.loadCats([cat_id])[0]
            self.category_map[cat_id] = cat_info['name']
            
        print(f"Category mapping: {self.category_map}")
        print(f"Category ID to index mapping: {self.category_id_to_idx}")

    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        
        # Load image
        image_info = self.coco_seg.loadImgs(image_id)[0]
        image_path = os.path.join(self.image_dir, image_info["file_name"])
        
        # Double-check file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        image = Image.open(image_path).convert("RGB")
        
        # Convert to tensor
        image = transforms.ToTensor()(image)
        
        # Load all annotations for this image
        ann_ids = self.coco_seg.getAnnIds(imgIds=image_id, iscrowd=False)
        anns = self.coco_seg.loadAnns(ann_ids)
        
        # Initialize target dictionary
        target = {}
        boxes = []
        masks = []
        labels = []
        category_ids = []  # Keep original category IDs for reference
        
        # Process each annotation
        for ann in anns:
            # Get bounding box
            bbox = ann['bbox']  # [x, y, width, height] format
            # Convert to [x1, y1, x2, y2] format
            boxes.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
            
            # Get mask
            mask = self.coco_seg.annToMask(ann)
            masks.append(torch.as_tensor(mask, dtype=torch.uint8))
            
            # Keep original category ID for reference
            category_ids.append(ann['category_id'])
            
            # Use the actual category index for the label
            cat_id = ann['category_id']
            class_idx = self.category_id_to_idx.get(cat_id, 1)  # Default to 1 if not found
            labels.append(class_idx)
        
        # Convert to tensor format
        if boxes:
            target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
            target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
            target["masks"] = torch.stack(masks)
            target["category_ids"] = torch.as_tensor(category_ids, dtype=torch.int64)  # original IDs for reference
        else:
            # Empty annotations
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0), dtype=torch.int64)
            target["masks"] = torch.zeros((0, image.shape[1], image.shape[2]), dtype=torch.uint8)
            target["category_ids"] = torch.zeros((0), dtype=torch.int64)
        
        target["image_id"] = torch.tensor([image_id])
        
        return image, target