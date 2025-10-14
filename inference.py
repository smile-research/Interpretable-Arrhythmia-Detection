import wandb
from pathlib import Path
import json
from src.data.physionet_challange_datamodule import PhysionetDataModule

import pandas as pd
import shutil
import os
import torch
import functools
import pytorch_lightning as pl

from src.data_model.fold_config import FoldConfig

from typing import NamedTuple
import tempfile

from src.lit_models.guangzhou_models import GuangzhouLitModel
from src.data.from_df_image_dataset import FromDfImageDataset
from src.utils.format import get_formatted_datetime
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

def parse_args():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--output_dir",
        help="Path to directory that will contain predictions",
        default="./outputs_inference",
    )
    parser.add_argument('--config_file')
    args = parser.parse_args()

    return args

def handle_wandb(wandb_config):
    if wandb_config is not None:
        if "entity" in wandb_config and "project" in wandb_config:
            run = wandb.init(
                entity=wandb_config["entity"],
                project=wandb_config["project"],
                job_type=wandb_config.get("job_type", None),
            )
            return run
        else:
            raise ValueError(f"Must provide both entity and project in wandb config, provided {wandb_config}")


def load_model(model_config, WANDB_RUN):
    if not isinstance(model_config, dict):  # assume link to wandb artifact
        if WANDB_RUN is None:
            raise Exception("Weights and Biases config has to be provided to use model from registry")
        artifact = WANDB_RUN.use_artifact(model_config, type="model")
        artifact_dir = Path(artifact.download())
        model = GuangzhouLitModel.load_from_checkpoint(artifact_dir / "model.ckpt")
    else:
        raise Exception("Supports running only wandb based models")
    return model


def prepare_folds_config(df):
    class FoldsConfig(NamedTuple):
        df: pd.DataFrame
        folds_train: list
        folds_val: list
        folds_test: list

    config = FoldsConfig(df=df, folds_test=["test"], folds_train=[], folds_val=[])

    return config


if __name__ == "__main__":
    args = parse_args()
    json_dict = json.load(open(args.config_file))

    key = get_formatted_datetime()

    folds_pred = json_dict["dataset"]["folds_pred"]
    folds_pred_unique = list(set(functools.reduce(lambda a, b: a + b, [folds for folds in folds_pred.values()]))) if folds_pred else []

    df = pd.read_csv(json_dict["dataset"]["df"])
    df = df.loc[df["fold"].isin(folds_pred_unique)]

    data_module_config = json_dict["data_module_config"]
    model_config = json_dict['model']

    WANDB_RUN = handle_wandb(json_dict['wandb_config'])

    fold_config = FoldConfig(
        df=df,
        folds_train=None,
        folds_val=None,
        folds_test=None,
        folds_pred=folds_pred,
        label_df=None,
    )
    
    data_module = PhysionetDataModule(fold_config, **data_module_config)
    
    trainer = pl.Trainer(logger=WandbLogger())
    lit_model = load_model(model_config, WANDB_RUN)
  
    os.makedirs(f"{args.output_dir}/models/{key}/preds")
    for pred_group, folds in fold_config.folds_pred.items():

        df_ = df[df['fold'].isin(folds)]

        print(pred_group, folds)
  
        dataset = FromDfImageDataset(df_, data_module.data_dir, transform=data_module.val_test_transform)

        dataloader = DataLoader(dataset, batch_size=data_module.batch_size, num_workers=data_module.num_workers)
        preds_tensors = trainer.predict(lit_model, dataloader)

        preds = torch.cat([x[0] for x in preds_tensors])
        labels = torch.cat([x[1] for x in preds_tensors])

        torch.save(preds, f"{args.output_dir}/models/{key}/preds/{pred_group}_preds.pt")
        torch.save(labels, f"{args.output_dir}/models/{key}/preds/{pred_group}_y.pt")

    artifact = wandb.Artifact("preds", type="preds")
    artifact.add_dir(f"{args.output_dir}/models/{key}")
    # artifact.add_file(local_path = f"{args.output_dir}/models/{key}/model.ckpt", name = "model")
    WANDB_RUN.log_artifact(artifact)
    artifact.wait()