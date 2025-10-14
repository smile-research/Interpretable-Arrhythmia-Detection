import json 
import wandb

def parse_args():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    # g - guangzhou_v4 , h - hanadi_training_v4
    parser.add_argument('--project', default="hanadi_training_v4")
    parser.add_argument('--json_path', default="./configs/minimal_inference_config.json")
    parser.add_argument('--config_path', default="/net/pr2/projects/plgrid/plggpolsl5/ecg_benchmarking_lit/temps/inference/hanadi_to_guangzhou/configs/")
    parser.add_argument('--script_path', default="/net/pr2/projects/plgrid/plggpolsl5/ecg_benchmarking_lit/temps/inference/hanadi_to_guangzhou/scripts/")
    parser.add_argument('--output_dir', default="/net/pr2/projects/plgrid/plggpolsl5/ecg_benchmarking_lit/temps/inference/preds/")
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    base_config = json.load(open(args.json_path))

    entity = base_config['wandb_config']['entity']
    project = args.project

    category = "model"
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")

    model_paths = []

    for run in runs:
        for artifact in run.logged_artifacts():
            if artifact.type == "model":
                artifact_path = f"{entity}/{project}/{artifact.name}"
                model_paths.append(artifact_path)

    json_main_path = args.config_path
    fake_json_path= json_main_path
    script_main_path = args.script_path
    output_dir = args.output_dir

    training_script = """#!/bin/sh
#SBATCH --partition plgrid-gpu-a100
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time 0:35:00
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
/net/pr2/projects/plgrid/plggpolsl5/ecg_benchmarking_lit/trainer.sif python3 inference.py --output_dir {output_dir} --config_file {config_name}
    """

    import json
    import itertools

    # Define the lists for folds (as strings), models, activations, and additional folds

    # Iterate over all combinations of the 3 lists
    for model_path in model_paths:
        # Create a copy of the base JSON object
        json_data = base_config.copy()
    
        json_data['model'] = model_path
        # Create a filename based on the combination values
        filename = f'{model_path.replace("/", "_")}_pred'
        json_filename_ = json_main_path + filename + ".json"
        fake_json_path_ = fake_json_path + filename + ".json"
        formatted_script = training_script.format(config_name=fake_json_path_, output_dir=output_dir)
        print(fake_json_path_)
        # Save the JSON data to the file
        with open(json_filename_, 'w') as json_file:
            json.dump(json_data, json_file, indent=4)
        with open(script_main_path + filename + ".slurm", 'w') as script_file:
            script_file.write(formatted_script)
        print(f"Saved {filename}")
