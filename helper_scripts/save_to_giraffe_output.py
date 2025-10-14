from pathlib import Path
import torch
import os
import wandb

def parse_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--project_name')
    parser.add_argument('--output_dir')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    
    args = parse_args()
    
    api = wandb.Api()
    project = api.project(args.project_name)
    
    output_dir_path = Path(args.output_dir)
    (output_gt_path := output_dir_path / 'gt').mkdir(exist_ok=True, parents=True)


    with wandb.init(project="downloads", job_type="download", name=f"download_project_{args.project_name}") as run:
        runs = api.runs("/".join(project.path))
        for r in runs:
            print(r)
            for artifact in r.logged_artifacts():
                run.use_artifact(artifact)
                if artifact.type == 'preds':
                    path = Path(artifact.download()) / 'preds'
                    for file in path.glob("*"):
                        if file.name.endswith("y.pt"): # to label dir
                            obj = torch.load(str(file))
                            torch.save(obj, output_gt_path / file.name)
                        else:
                            dir_name = file.name.replace('_preds.pt', '')
                            (output_preds_path := output_dir_path / dir_name).mkdir(exist_ok=True)
                            obj = torch.load(str(file))
                            torch.save(obj, output_preds_path / f"{r.name}.pt")

    