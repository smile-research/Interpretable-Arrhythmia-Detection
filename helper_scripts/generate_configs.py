base_config = {
    "data_module_config": {
        "data_dir": "/",
        "batch_size": 16,
        "num_workers": 8
    },
    "dataset": {
        "df": "/net/pr2/projects/plgrid/plggpolsl5/ecg_benchmarking_lit/configs/joined_splits.csv",
        "folds_train": [
            "0",
            "1",
            "2",
            "3",
            "4"
        ],
        "folds_val": [
            "valid"
        ],
        "folds_test": [
            "test"
        ],
        "folds_pred": {
            "test" : ["test"],
            "valid" : ["valid"]
        }
    },
    "model": {
        "model_name" : "efficienetS",
        "activation" : "sigmoid",
        "n_labels" : 1
    },
    "lit_model_params": {
        "learning_rate": 3e-4,
        "wd": 1e-4,
        "onecycle": False
    },
    "trainer_params": {
        "accumulate_grad_batches": 1,
        "log_every_n_steps": 1,
        "max_epochs": 35,
        "detect_anomaly": True
    },
    "wandb_config": {
        "entity": "phd-dk",
        "project": "all_training_v5"
    },
    "strategies": {
        "weigh_classes": True
    }
}

all_folds = [
    "0_hanadi",
    "1_hanadi", 
    "2_hanadi",
    "3_hanadi",
    "4_hanadi",
    "0_lip",
    "1_lip",
    "2_lip",
    "3_lip",
    "4_lip",
    "valid_hanadi",
    "valid_lip",
    "test_hanadi",
    "test_lip"
]

json_main_path = "./lip_configs/"
script_main_path = "/net/pr2/projects/plgrid/plggpolsl5/ecg_benchmarking_lit/temps/temp_runs_all/"
fake_json_path = "/net/pr2/projects/plgrid/plggpolsl5/ecg_benchmarking_lit/temps/temp_configs_all/"

json_main_path = fake_json_path


training_script = """#!/bin/sh
#SBATCH --partition plgrid-gpu-a100
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time 3:05:00
#SBATCH --job-name training-ecg
#SBATCH --account=plggrantpolsl2-gpu-a100

module load CUDA/12.0.0

singularity run --nv \
-B /net/pr2/projects/plgrid/plggpolsl5/data/guangzhou_paper_ecg_dataset/ \
-B /net/pr2/projects/plgrid/plggpolsl5/data/hanadi/ \
-B /net/pr2/projects/plgrid/plggpolsl5/runs \
-B /etc/ssl/certs/ca-bundle.crt \
-B /net/tscratch/people/plgarkacze/slurm_jobdir \
-B /net/pr2/projects/plgrid/plggpolsl5/ecg_benchmarking_lit/wandb_artifact \
-B /net/pr2/projects/plgrid/plggpolsl5/ecg_benchmarking_lit/wandb_cache \
/net/pr2/projects/plgrid/plggpolsl5/ecg_benchmarking_lit/trainer.sif python3 train_ecg_classifier.py --output_dir /net/pr2/projects/plgrid/plggpolsl5/runs --config_file {config_name}

"""

import json
import itertools

# Define the lists for folds (as strings), models, activations, and additional folds
folds = ['0', '1', '2', '3', '4', 'valid']  # Folds as string numbers and 'valid'
models = ['efficienetS', 'inception']
activations = ['sigmoid', 'gev']

# Define an additional list of folds to search for matching train and valid folds
additional_folds = all_folds

# Iterate over all combinations of the 3 lists
for comb in itertools.product(folds, models, activations):
    # Create a copy of the base JSON object
    json_data = base_config.copy()

    # If the current fold is 'valid', only include folds from additional_folds that start with 'valid'

    folds_without_test = [f for f in additional_folds if not f.startswith('test')]

    if comb[0] == 'valid':
        json_data["dataset"]["folds_train"] = [f for f in folds_without_test if not f.startswith('valid')]
        json_data["dataset"]["folds_val"] = [f for f in folds_without_test if f.startswith('valid')]
    else:
        # For non-'valid' folds, add matching folds from additional_folds
        json_data["dataset"]["folds_train"] = [f for f in folds_without_test if not f.startswith(comb[0]) and not f.startswith('valid')]
        # For train_folds, add all other matching folds from additional_folds that do not start with the current valid fold
        json_data["dataset"]["folds_val"] = [f for f in folds_without_test if f.startswith(comb[0])]

    # Update model and activation fields
    json_data["dataset"]['folds_pred']['train'] = [fold for fold in all_folds if not fold.startswith("test") and not fold.startswith("valid")]
    json_data["dataset"]['folds_pred']['test'] = [fold for fold in all_folds if fold.startswith("test")]
    json_data["dataset"]['folds_pred']['valid'] = [fold for fold in all_folds if fold.startswith("valid")]
    json_data["dataset"]['folds_pred']['test_hanadi'] = ["test_hanadi"]

    json_data["dataset"]['folds_pred']['test_lip'] = ["test_lip"]


    json_data["model"]['model_name'] = comb[1]
    json_data["model"]["activation"] = comb[2]

    # Create a filename based on the combination values
    filename = f"fold_{comb[0]}_model_{comb[1]}_activation_{comb[2]}"
    json_filename_ = json_main_path + filename + ".json"
    fake_json_path_ = fake_json_path + filename + ".json"
    formatted_script = training_script.format(config_name=fake_json_path_)
    print(fake_json_path_)
    # Save the JSON data to the file
    with open(json_filename_, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)
    with open(script_main_path + filename + ".slurm", 'w') as script_file:
        script_file.write(formatted_script)
    print(f"Saved {filename}")
