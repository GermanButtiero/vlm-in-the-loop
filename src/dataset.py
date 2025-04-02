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

config = json.load(open("config.json"))

class CocoSegmentationDatasetMRCNN(Dataset):
    def __init__(self, image_dir, seg_annotation_file, categories_to_keep=[1], min_area_threshold=100):
        self.image_dir = image_dir
        self.coco_seg = COCO(seg_annotation_file)
        self.min_area_threshold = min_area_threshold
        self.categories_to_keep = categories_to_keep
        
        # Filter images to keep only those containing objects from specified categories
        self.image_ids = []
        for cat_id in self.categories_to_keep:
            ann_ids = self.coco_seg.getAnnIds(catIds=[cat_id], iscrowd=False)
            anns = self.coco_seg.loadAnns(ann_ids)
            valid_anns = [ann for ann in anns if ann['area'] >= self.min_area_threshold]
            img_ids = list(set([ann['image_id'] for ann in valid_anns]))
            self.image_ids.extend(img_ids)
        
        # Remove duplicates
        self.image_ids = list(set(self.image_ids))
        print(f"Dataset contains {len(self.image_ids)} images with categories {categories_to_keep}")
        
        # For visualization, create a category mapping
        self.category_map = {}
        for cat_id in self.categories_to_keep:
            cat_info = self.coco_seg.loadCats(cat_id)[0]
            self.category_map[cat_id] = cat_info['name']

    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        
        # Load image
        image_info = self.coco_seg.loadImgs(image_id)[0]
        image_path = os.path.join(self.image_dir, image_info["file_name"])
        image = Image.open(image_path).convert("RGB")
        
        # Convert to tensor
        image = transforms.ToTensor()(image)
        
        # Load annotations
        ann_ids = self.coco_seg.getAnnIds(imgIds=image_id, catIds=self.categories_to_keep, iscrowd=False)
        anns = self.coco_seg.loadAnns(ann_ids)
        
        # Initialize target dictionary
        target = {}
        boxes = []
        masks = []
        labels = []
        category_ids = []  # Keep original category IDs for reference
        
        # Process each annotation
        for ann in anns:
            if ann['area'] < self.min_area_threshold:
                continue
                
            # Get bounding box
            bbox = ann['bbox']  # [x, y, width, height] format
            # Convert to [x1, y1, x2, y2] format
            boxes.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
            
            # Get mask
            mask = self.coco_seg.annToMask(ann)
            masks.append(torch.as_tensor(mask, dtype=torch.uint8))
            
            # Keep original category ID for reference
            category_ids.append(ann['category_id'])
            
            # For segmentation only, use class 1 for all foreground objects
            labels.append(1)  # 1 for foreground, 0 for background
        
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
    


