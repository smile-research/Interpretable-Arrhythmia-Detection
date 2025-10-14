import json
import os
import torch  # Assuming .pt files are PyTorch tensors
import wandb
from sklearn.metrics import f1_score, accuracy_score, roc_curve, auc
import pandas as pd

def parse_args():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--json_path', default="./minimal_inference_config.json")
    parser.add_argument('--config_path', default="/net/pr2/projects/plgrid/plggpolsl5/ecg_benchmarking_lit/temps/inference/hanadi_to_guangzhou/configs/")
    parser.add_argument('--script_path', default="/net/pr2/projects/plgrid/plggpolsl5/ecg_benchmarking_lit/temps/inference/hanadi_to_guangzhou/scripts/")
    parser.add_argument('--output_dir', default="/net/pr2/projects/plgrid/plggpolsl5/ecg_benchmarking_lit/temps/inference/preds/")
    args = parser.parse_args()

    return args

def load_and_score(preds_path, targets_path):
    """
    Load prediction and target files, and then perform scoring.
    Calculates F1 score, accuracy, and ROC curve (AUC).
    """
    # Load the .pt files
    preds = torch.load(preds_path)
    targets = torch.load(targets_path)

    # Convert tensors to numpy arrays for sklearn compatibility (if necessary)
    preds = preds.squeeze().cpu().numpy() if torch.is_tensor(preds) else preds
    targets = targets.cpu().numpy() if torch.is_tensor(targets) else targets

    # For ROC curve and AUC, we need raw prediction scores before thresholding
    raw_preds = preds

    # Threshold predictions for accuracy and F1 score
    preds = (preds >= 0.5).astype(int)

    # Compute accuracy and F1 score
    accuracy = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average='macro')  # Use 'macro' to average F1 across all classes equally

    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(targets, raw_preds)
    roc_auc = auc(fpr, tpr)

    return accuracy, f1, roc_auc

def find_matching_files(directory):
    """
    Find all matching pairs of files with the suffixes '_preds.pt' and '_y.pt' in the directory.
    Returns a list of tuples, where each tuple contains (preds_file_path, targets_file_path).
    """
    preds_files = {}
    targets_files = {}

    # Iterate through the directory files
    for filename in os.listdir(directory):
        if filename.endswith("_preds.pt"):
            base_name = filename.replace("_preds.pt", "")
            preds_files[base_name] = os.path.join(directory, filename)
        elif filename.endswith("_y.pt"):
            base_name = filename.replace("_y.pt", "")
            targets_files[base_name] = os.path.join(directory, filename)

    # Match files based on the base name
    matching_files = []
    for base_name in preds_files:
        if base_name in targets_files:
            matching_files.append((base_name, preds_files[base_name], targets_files[base_name]))

    return matching_files

if __name__ == "__main__":
    args = parse_args()
    base_config = json.load(open(args.json_path))
    
    # Initialize DataFrame for storing results
    results_df = pd.DataFrame(columns=["Project", "Run", "Base_Name", "Accuracy", "F1", "AUC"])

    entity = base_config['wandb_config']['entity']
    projects = ['inference_hanadi_to_guangzhou', "inference_guangzhou_to_hanadi", "all_training_v5", "hanadi_training_v4", "guangzhou_v4"]

    api = wandb.Api()

    # Iterate over each project in the list
    for project in projects:
        runs = api.runs(f"{entity}/{project}")
        for run in runs:
            for artifact in run.logged_artifacts():
                if artifact.type == "preds":
                    path = artifact.download() + "/preds"

                    # Find matching _preds.pt and _y.pt files
                    matching_files = find_matching_files(path)

                    if matching_files:
                        for base_name, preds_path, targets_path in matching_files:
                            # Load and score each matching pair
                            accuracy, f1, roc_auc = load_and_score(preds_path, targets_path)
                            
                            # Append results to the DataFrame
                            new_row = {
                                "Project": project,
                                "Run": run.name,
                                "Base_Name": base_name.split("_")[0],
                                "Accuracy": accuracy,
                                "F1": f1,
                                "AUC": roc_auc
                            }
                            
                            # Append the new row to the DataFrame using pd.concat
                            results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
                    else:
                        print(f"No matching files found in artifact: {path}")
    
    # Output the results DataFrame
    print(results_df)
    results_df.to_csv("scores.csv", index=False)
