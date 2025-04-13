import json
import torch
from src.dataset import CocoSegmentationDatasetMRCNN
from src.train import train_model, get_model_instance_segmentation
from src.evaluate import calculate_ap
import argparse
from zipfile import ZipFile
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import cv2
from PIL import Image
from io import BytesIO
import torchvision.transforms as T

# Update imports at the top of the file
from transformers import (
    AutoModelForCausalLM,
)

def collate_fn(batch):
        return tuple(zip(*batch))

def calculate_mask_iou(pred_mask, gt_mask):
    """Calculate IoU between prediction and ground truth masks"""
    # Convert to binary masks if needed
    if pred_mask.dim() > 2 and pred_mask.shape[0] == 1:
        pred_mask = pred_mask.squeeze(0)
    if pred_mask.max() <= 1.0:
        pred_mask = (pred_mask > 0.5).float()
    
    intersection = (pred_mask * gt_mask).sum().float()
    union = ((pred_mask + gt_mask) > 0).float().sum()
    
    if union == 0:
        return 0.0
    return (intersection / union).item()

def simulate_human_feedback(prediction, target, iou_threshold=0.6):
    """
    Simulate human feedback by comparing prediction with ground truth masks.
    Returns True if the prediction meets the IoU threshold criteria.
    """
    pred_masks = prediction['masks'].cpu()
    gt_masks = target['masks'].cpu()
    
    # Skip images with no masks
    if len(pred_masks) == 0 or len(gt_masks) == 0:
        return False
    
    # Filter predictions with score > 0.5
    keep = prediction['scores'].cpu() > 0.5
    pred_masks = pred_masks[keep]
    
    if len(pred_masks) == 0:
        return False
    
    # Check if each ground truth has a good match
    all_matched = True
    for gt_mask in gt_masks:
        best_iou = 0
        for pred_mask in pred_masks:
            iou = calculate_mask_iou(pred_mask, gt_mask)
            best_iou = max(best_iou, iou)
        
        if best_iou < iou_threshold:
            all_matched = False
            break
    
    return all_matched

def visualize_segmentation_mask(image, prediction, threshold=0.5):
    """
    Create a visualization of the segmentation masks on the image.
    
    Args:
        image: The original image tensor
        prediction: The model prediction with masks
        threshold: Score threshold for showing masks
    
    Returns:
        PIL Image with visualization
    """
    # Convert image tensor to numpy array
    img_np = image.cpu().permute(1, 2, 0).numpy()
    
    # Denormalize if needed
    if img_np.max() <= 1.0:
        img_np = (img_np * 255).astype(np.uint8)
    
    # Create a copy for visualization
    vis_img = img_np.copy()
    
    # Get masks and scores
    masks = prediction['masks'].cpu().squeeze(1).numpy()
    scores = prediction['scores'].cpu().numpy()
    
    # Apply different colors for each mask
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
              (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0)]
    
    for i, (mask, score) in enumerate(zip(masks, scores)):
        if score < threshold:
            continue
            
        # Convert mask to binary
        binary_mask = (mask > 0.5).astype(np.uint8)
        
        # Apply color to mask
        color = colors[i % len(colors)]
        colored_mask = np.zeros_like(vis_img)
        for c in range(3):
            colored_mask[:, :, c] = binary_mask * color[c]
        
        # Blend with original image
        alpha = 0.4
        vis_img = cv2.addWeighted(vis_img, 1, colored_mask, alpha, 0)
        
        # Find contours and draw them
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_img, contours, -1, color, 2)
    
    # Convert back to PIL image
    return Image.fromarray(vis_img)

class LocalVLM:
    """Class to handle local or Ovis VLM inference"""
    
    def __init__(self, model_name=None, device="cuda"):
        """
        Initialize the VLM model.
        
        Args:
            model_name: Name of the model to load from HuggingFace or path for Ovis model
            device: Device to run the model on
        """
        print(f"Loading VLM model: {model_name}")
        self.model_name = model_name
        self.device = device
        
        # Check if this is an Ovis2-4B model
        is_ovis2_model = "Ovis2" in model_name
        
        if is_ovis2_model:
            try:
                # Initialize Ovis2-4B directly with transformers
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16,
                    multimodal_max_length=32768,
                    trust_remote_code=True, use_flash_attention_2=False
                ).to(device)
                
                # Get tokenizers
                self.text_tokenizer = self.model.get_text_tokenizer()
                self.visual_tokenizer = self.model.get_visual_tokenizer()
                
                self.model_type = "ovis2"
                print("Ovis2 VLM model loaded successfully")
            except Exception as e:
                raise ValueError(f"Failed to load Ovis2 model: {e}. Make sure to install required dependencies: torch==2.4.0 transformers==4.46.2 numpy==1.25.0 pillow==10.3.0 flash-attn==2.7.0.post2")
        

    def evaluate_segmentation(self, image, prompt):
        """
        Evaluate segmentation quality using the VLM.

        Args:
            image: PIL Image with segmentation visualization
            prompt: Prompt to evaluate the segmentation
            
        Returns:
            Boolean indicating approval and the model's response
        """
        try:
            if self.model_type == "ovis2":
                # Format query for Ovis2-4B
                query = f"<image>\n{prompt}"
                
                # Preprocess inputs using Ovis2-4B's specific method
                _, input_ids, pixel_values = self.model.preprocess_inputs(query, [image], max_partition=9)
                attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id)
                input_ids = input_ids.unsqueeze(0).to(device=self.model.device)
                attention_mask = attention_mask.unsqueeze(0).to(device=self.model.device)
                
                if pixel_values is not None:
                    pixel_values = pixel_values.to(
                        dtype=self.visual_tokenizer.dtype, 
                        device=self.visual_tokenizer.device
                    )
                pixel_values = [pixel_values]
                
                # Generate response
                with torch.inference_mode():
                    gen_kwargs = dict(
                        max_new_tokens=256,  # Shorter response for segmentation quality evaluation
                        do_sample=False,
                        eos_token_id=self.model.generation_config.eos_token_id,
                        pad_token_id=self.text_tokenizer.pad_token_id,
                        use_cache=True
                    )
                    output_ids = self.model.generate(
                        input_ids, 
                        pixel_values=pixel_values, 
                        attention_mask=attention_mask, 
                        **gen_kwargs
                    )[0]
                    
                    response = self.text_tokenizer.decode(output_ids, skip_special_tokens=True)
                # Extract the model's answer
                if prompt in response:
                    answer = response.split(prompt, 1)[1].strip().lower()
                else:
                    answer = response.lower()
                
                # Check if the answer indicates approval
                is_approved = "yes" in answer and not ("no" in answer and len(answer) < 20)
            
            return is_approved, answer
            
        except Exception as e:
            print(f"Error during VLM inference: {e}")
            return False, f"Error generating response: {e}"

def run_active_learning(config, device, iou_threshold=0.6, iterations=10, use_vlm=False, vlm_model_name=None):
    """
    Run active learning loop with either simulated human feedback or VLM feedback.
    
    Args:
        config: Configuration dictionary
        device: Torch device to use
        iou_threshold: Minimum IoU threshold for human approval (used when not using VLM)
        iterations: Number of active learning iterations
        use_vlm: Whether to use VLM for feedback
        vlm_model_name: Name of the VLM model to use
    """
    # Load the full training dataset
    full_dataset = CocoSegmentationDatasetMRCNN(
        config["train_image_dir"],
        config["train_annotation_file"]
    )
    
    # Create initial split: 20% train, 80% inference pool
    dataset_size = len(full_dataset)
    all_indices = list(range(dataset_size))
    np.random.shuffle(all_indices)  # Randomize
    
    initial_train_size = int(0.2 * dataset_size)
    train_indices = set(all_indices[:initial_train_size])
    inference_indices = set(all_indices[initial_train_size:])
    
    # Prepare test dataset for consistent evaluation
    test_dataset = CocoSegmentationDatasetMRCNN(
        config["val_image_dir"],
        config["val_annotation_file"]
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config["batch_size"],
        shuffle=False, collate_fn=collate_fn
    )
    
    # Initialize metrics log
    metrics_log = {
        "iteration": [],
        "train_size": [],
        "mean_ap": [],
        "ap_per_class": [],
        "feedback_method": "vlm" if use_vlm else f"iou_threshold_{iou_threshold}"
    }
    
    # Update the VLM prompt in run_active_learning
    # Update the VLM prompt in run_active_learning
    vlm_prompt_template = """
    Examine this image showing object segmentation masks.
    The colored areas represent the computer's identification of objects in the image.

    Evaluate if the segmentation is accurate. The segmentation is accurate when the colored masks precisely follow the boundaries of the actual objects in the image.

    Answer with ONLY "Yes" if the segmentation is accurate, or "No" followed by a brief explanation if the segmentation is poor (masks miss objects or don't follow object boundaries correctly).
    """
        
    # Initialize VLM if using it
    vlm = None
    if use_vlm:
        vlm = LocalVLM(model_name=vlm_model_name, device=device)
    
    print(f"Starting active learning with {initial_train_size} initial training images")
    if use_vlm:
        print(f"Using VLM feedback with model: {vlm_model_name}")
    else:
        print(f"Using simulated human feedback with IoU threshold of {iou_threshold}")
    
    # Create a folder for saving visualizations if using VLM
    if use_vlm:
        os.makedirs("vlm_decisions", exist_ok=True)
    
    # Active learning iterations
    for iteration in range(iterations):
        print(f"\n===== Iteration {iteration+1}/{iterations} =====")
        print(f"Training set size: {len(train_indices)}")
        print(f"Inference pool size: {len(inference_indices)}")
        
        # Create current training dataset
        train_dataset = torch.utils.data.Subset(full_dataset, list(train_indices))
        
        # Split into train and validation
        train_size = int(0.9 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        
        dataset_train, dataset_val = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
        
        # Create data loaders
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, batch_size=config["batch_size"],
            shuffle=True, collate_fn=collate_fn
        )
        
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, batch_size=config["batch_size"],
            shuffle=False, collate_fn=collate_fn
        )
        
        # Train model
        print("Training model...")
        num_classes = config["num_classes"]
        train_model(data_loader_train, data_loader_val, num_classes, 
                  num_epochs=config["num_epochs"], device=device)
        
        # Load trained model
        model = get_model_instance_segmentation(num_classes=num_classes + 1)  # +1 for background
        model.load_state_dict(torch.load(config["model_path"]))
        model.to(device)
        model.eval()
        
        # Evaluate on test set
        print("Evaluating model...")
        mean_ap, ap_per_class = calculate_ap(model, test_loader, device, iou_threshold=0.5)
        print(f"Mean Average Precision: {mean_ap:.4f}")
        
        # Log metrics
        metrics_log["iteration"].append(iteration+1)
        metrics_log["train_size"].append(len(train_indices))
        metrics_log["mean_ap"].append(float(mean_ap))
        metrics_log["ap_per_class"].append([float(ap) for ap in ap_per_class])
        
        # Save intermediate log
        log_filename = f"active_learning_metrics_{'vlm' if use_vlm else 'iou'}_iter_{iteration+1}.json"
        with open(log_filename, "w") as f:
            json.dump(metrics_log, f, indent=4)
        
        # If last iteration, break
        if iteration == iterations - 1:
            break
        
        # Run inference on the pool and get feedback
        feedback_method = "VLM" if use_vlm else "simulated human feedback"
        print(f"Getting {feedback_method} on segmentation quality...")
        
        inference_dataset = torch.utils.data.Subset(full_dataset, list(inference_indices))
        inference_loader = torch.utils.data.DataLoader(
            inference_dataset, batch_size=1,  # Process one at a time
            shuffle=False, collate_fn=collate_fn
        )
        
        approved_indices = []
        vlm_decisions = []  # Store VLM decisions for analysis
        
        with torch.no_grad():
            for i, (image, target) in enumerate(tqdm(inference_loader, desc="Evaluating images")):
                original_idx = list(inference_indices)[i]
                
                # Forward pass
                image = [img.to(device) for img in image]
                outputs = model(image)
                
                # Check if prediction meets quality threshold
                if use_vlm:
                    # Create visualization
                    vis_img = visualize_segmentation_mask(image[0], outputs[0])
                    
                    # Get VLM feedback
                    is_approved, answer = vlm.evaluate_segmentation(vis_img, vlm_prompt_template)
                    
                    # Save decision for analysis
                    vlm_decisions.append({
                        "image_idx": original_idx,
                        "approved": is_approved,
                        "vlm_response": answer
                    })
                    
                    # Save some examples to inspect VLM decisions
                    if i % 10 == 0:
                        decision = "approved" if is_approved else "rejected"
                        vis_img.save(f"vlm_decisions/iter{iteration+1}_img{i}_{decision}.jpg")
                else:
                    # Use simulated feedback based on IoU
                    is_approved = simulate_human_feedback(outputs[0], target[0], iou_threshold)
                
                if is_approved:
                    approved_indices.append(original_idx)
        
        # Save VLM decisions if using VLM
        if use_vlm:
            with open(f"vlm_decisions_iter_{iteration+1}.json", "w") as f:
                json.dump(vlm_decisions, f, indent=4)
        
        print(f"Approved {len(approved_indices)} new images")
        
        # Update indices
        train_indices.update(approved_indices)
        inference_indices = inference_indices.difference(set(approved_indices))
    
    # Save final metrics log
    final_log_filename = f"active_learning_metrics_{'vlm' if use_vlm else 'iou'}_final.json"
    with open(final_log_filename, "w") as f:
        json.dump(metrics_log, f, indent=4)
    
    # Plot results
    plot_learning_curve(metrics_log, use_vlm)
    
    print("\nActive learning completed!")
    print(f"Final training set size: {len(train_indices)}")
    print(f"Final mean AP: {metrics_log['mean_ap'][-1]:.4f}")
    
    return metrics_log

def plot_learning_curve(metrics_log, use_vlm=False):
    """Create a plot of mean AP vs iteration"""
    plt.figure(figsize=(10, 6))
    plt.plot(metrics_log["iteration"], metrics_log["mean_ap"], 'o-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Mean Average Precision')
    method = "VLM Feedback" if use_vlm else "IoU-based Feedback"
    plt.title(f'Active Learning Performance with {method}')
    plt.grid(True)
    plt.savefig(f'active_learning_curve_{"vlm" if use_vlm else "iou"}.png')
    plt.close()

# Fix the duplicate argument in the argparse section
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Mask R-CNN model on COCO dataset")
    parser.add_argument("--train", action="store_true", default=False, help="Train the model")
    parser.add_argument("--test", action="store_true", default=False, help="Test the model")
    parser.add_argument("--active", action="store_true", default=False, help="Run active learning loop")
    parser.add_argument("--vlm", action="store_true", default=False, help="Use VLM for feedback instead of IoU")
    parser.add_argument("--vlm-model", type=str, 
                       default="Aimv2-huge-patch14-448-Qwen2.5-3B-Instruct", 
                       help="VLM model name from HuggingFace or path to Ovis model")
    parser.add_argument("--iou-threshold", type=float, default=0.6, help="IoU threshold for active learning")
    parser.add_argument("--iterations", type=int, default=10, help="Number of active learning iterations")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    config = json.load(open("config.json"))

    if args.active:
        print("Running active learning loop...")
        run_active_learning(
            config, 
            device, 
            iou_threshold=args.iou_threshold,
            iterations=args.iterations,
            use_vlm=args.vlm,
            vlm_model_name=args.vlm_model
        )
    elif args.train:
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