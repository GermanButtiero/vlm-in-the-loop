import json
import torch
import argparse
import os
import datetime
from main import run_active_learning, plot_learning_curve # Import from main.py

def run_proportion_experiments(
    base_config,
    device,
    initial_proportions,
    iterations,
    use_vlm,
    vlm_model_name,
    iou_threshold,
    output_root="output_experiments"
):
    """
    Runs active learning experiments for different initial training proportions.

    Args:
        base_config: The base configuration dictionary.
        device: Torch device.
        initial_proportions: List of initial training proportions to test.
        iterations: Number of active learning iterations for each experiment.
        use_vlm: Whether to use VLM feedback.
        vlm_model_name: Name of the VLM model.
        iou_threshold: IoU threshold if not using VLM.
        output_root: Root directory to save all experiment results.
    """
    all_results = {}
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_group_folder = os.path.join(output_root, f"proportions_exp_{timestamp}")
    os.makedirs(experiment_group_folder, exist_ok=True)

    print(f"Starting experiments with initial proportions: {initial_proportions}")
    print(f"Results will be saved under: {experiment_group_folder}")

    for proportion in initial_proportions:
        print(f"\n===== Running Experiment: Initial Proportion = {proportion:.2f} =====")

        # Create a specific output directory for this proportion run
        run_output_folder_name = f"init_prop_{proportion:.2f}"
        run_output_path = os.path.join(experiment_group_folder, run_output_folder_name)
        # The run_active_learning function will create subdirs within this path

        # Run the active learning loop for this specific proportion
        # The run_active_learning function already handles saving its own detailed logs
        final_metrics = run_active_learning(
            config=base_config,
            device=device,
            iou_threshold=iou_threshold,
            iterations=iterations,
            init_train_proportion=proportion, # Pass the current proportion
            use_vlm=use_vlm,
            vlm_model_name=vlm_model_name,
            output_root=run_output_path # Pass the specific output path for this run
        )

        # Store the final metrics for this proportion
        all_results[f"proportion_{proportion:.2f}"] = final_metrics
        print(f"===== Finished Experiment: Initial Proportion = {proportion:.2f} =====")
        print(f"Final mAP for proportion {proportion:.2f}: {final_metrics['mean_ap'][-1]:.4f}")


    # Save the aggregated final results from all proportions
    summary_file = os.path.join(experiment_group_folder, "experiment_summary.json")
    try:
        # Attempt to save with pretty printing
        with open(summary_file, "w") as f:
            json.dump(all_results, f, indent=4)
    except TypeError as e:
        print(f"Warning: Could not save summary with full detail due to TypeError: {e}")
        # Fallback: Save only key final metrics if serialization fails
        simplified_results = {}
        for key, metrics in all_results.items():
             simplified_results[key] = {
                 "final_train_size": metrics.get("train_size", [])[-1] if metrics.get("train_size") else None,
                 "final_mean_ap": metrics.get("mean_ap", [])[-1] if metrics.get("mean_ap") else None,
                 "feedback_method": metrics.get("feedback_method")
             }
        with open(summary_file, "w") as f:
            json.dump(simplified_results, f, indent=4)
        print(f"Saved simplified summary to {summary_file}")
    else:
         print(f"Saved experiment summary to {summary_file}")

    print("\nAll experiments completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Active Learning Experiments with different initial proportions.")
    parser.add_argument("--config", type=str, default="config.json", help="Path to the base config file.")
    parser.add_argument("--proportions", type=float, nargs='+', default=[0.2, 0.4, 0.6, 0.8], help="List of initial training proportions.")
    parser.add_argument("--iterations", type=int, default=5, help="Number of active learning iterations per experiment.") # Reduced default for faster testing
    parser.add_argument("--vlm", action="store_true", default=False, help="Use VLM for feedback instead of IoU.")
    parser.add_argument("--vlm-model", type=str, default="AIDC-AI/Ovis2-1B", help="VLM model name or path.")
    parser.add_argument("--iou-threshold", type=float, default=0.6, help="IoU threshold for non-VLM active learning.")
    parser.add_argument("--output-root", type=str, default="output_experiments", help="Root directory for saving experiment results.")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_config = json.load(open(args.config))

    # Ensure the number of classes is correctly read from config for the model
    if "num_classes" not in base_config:
         # You might need to calculate this based on your categories_to_ or set it manually
         print("Warning: 'num_classes' not found in config.json. Assuming 4 based on default categories.")
         base_config["num_classes"] = 4 # Adjust if your default categories change

    run_proportion_experiments(
        base_config=base_config,
        device=device,
        initial_proportions=args.proportions,
        iterations=args.iterations,
        use_vlm=args.vlm,
        vlm_model_name=args.vlm_model,
        iou_threshold=args.iou_threshold,
        output_root=args.output_root
    )
    
    #python run_active_learning_experiments.py --vlm --iterations=2 --proportions 0.2 0.4 0.6 0.8 --output-root output_proportion_vlm_runs --vlm-model "AIDC-AI/Ovis2-1B"