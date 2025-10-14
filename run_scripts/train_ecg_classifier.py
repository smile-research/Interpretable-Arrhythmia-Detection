import json

import pandas as pd
import numpy as np
import wandb

from src.data_model.fold_config import FoldConfig
from src.lit_models.guangzhou_models import GuangzhouLitModel
from src.data.physionet_challange_datamodule import PhysionetDataModule

from pathlib import Path
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
import pytorch_lightning as pl
import functools
import os
import shutil
from src.data.from_df_image_dataset import FromDfImageDataset
from torch.utils.data import DataLoader
import torch

from src.utils.format import get_formatted_datetime
from argparse import ArgumentParser


os.environ['WANDB_CACHE_DIR'] = "/net/pr2/projects/plgrid/plggpolsl5/ecg_benchmarking_lit/wandb_cache"
os.environ['WANDB_ARTIFACT_DIR'] = "/net/pr2/projects/plgrid/plggpolsl5/ecg_benchmarking_lit/wandb_artifact"



WANDB_RUN = None

from collections import Counter

def calculate_class_weights(df, label_column):
    # Count the occurrences of each class
    class_counts = Counter(df[label_column])
    
    # Total number of samples
    total_samples = len(df)
    
    # Calculate the weight for each class as: total_samples / (number_of_classes * class_count)
    class_weights = {str(int(cls)): total_samples / (len(class_counts) * count) for cls, count in class_counts.items()}
    print(f"{class_weights=}")
    return list(class_weights.values())

def load_configs(json_dict):
    folds_train = json_dict["dataset"]["folds_train"]
    folds_val = json_dict["dataset"]["folds_val"]
    folds_test = json_dict["dataset"]["folds_test"]
    folds_pred = json_dict["dataset"].get("folds_pred", {}) 

    folds_pred_unique = list(set(functools.reduce(lambda a, b: a + b, [folds for folds in folds_pred.values()]))) if folds_pred else []

    df = pd.read_csv(json_dict["dataset"]["df"])
    df = df.loc[df["fold"].isin(list(set(folds_train + folds_val + folds_test + folds_pred_unique)))]

    label_df_path = json_dict["dataset"].get("label_df", None)
    if label_df_path is not None:
        label_df = pd.read_csv(label_df_path)
    else:
        label_df = None

    if label_df is None:
        df["label"] = df["label"].astype(np.float32)

    fold_config = FoldConfig(
        df=df,
        folds_train=folds_train,
        folds_val=folds_val,
        folds_test=folds_test,
        folds_pred=folds_pred,
        label_df=label_df,
    )

    print(f"Wandb cache dir: {os.environ['WANDB_CACHE_DIR']}")

    lit_model_params = json_dict["lit_model_params"]
    trainer_config = json_dict["trainer_params"]
    model_config = json_dict["model"]
    print(f"{model_config=}")
    data_module_config = json_dict["data_module_config"]

    wandb_config = json_dict.get("wandb_config", None)
    strategies = json_dict.get("strategies", None)

    return fold_config, lit_model_params, model_config, data_module_config, trainer_config, wandb_config, strategies


def load_model(model_config, lit_model_params):
    if not isinstance(model_config, dict):  # assume link to wandb artifact
        if WANDB_RUN is None:
            raise Exception("Weights and Biases config has to be provided to use model from registry")
        artifact = WANDB_RUN.use_artifact(model_config, type="model")
        artifact_dir = Path(artifact.download())
        model = GuangzhouLitModel.load_from_checkpoint(artifact_dir / "model.ckpt", **lit_model_params)
    else:
        model = GuangzhouLitModel(
            model_name=model_config["model_name"],
            activation=model_config["activation"],
            n_labels=model_config["n_labels"],
            **lit_model_params,
        )

    return model


def handle_wandb(wandb_config):
    global WANDB_RUN
    if wandb_config is not None:
        if "entity" in wandb_config and "project" in wandb_config:
            run = wandb.init(
                entity=wandb_config["entity"],
                project=wandb_config["project"],
                job_type=wandb_config.get("job_type", None),
            )
            WANDB_RUN = run
        else:
            raise ValueError(f"Must provide both entity and project in wandb config, provided {wandb_config}")


def prepare_callbacks(key, output_dir):
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{output_dir}/models/{key}",
        filename="model",
        monitor="val_auroc_epoch",
        save_top_k=1,
        mode="max",
    )

    early_stop_callback = EarlyStopping(
        monitor="val_auroc_epoch", min_delta=0.00, patience=10, verbose=True, mode="max"
    )
    learning_rate_monitor = LearningRateMonitor(logging_interval="step", log_momentum=True)

    callbacks = [checkpoint_callback, early_stop_callback, learning_rate_monitor]

    return callbacks


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--config_file", type=str, required=True)
    args = parser.parse_args()
    return args


def main(args):
    key = get_formatted_datetime()

    config_json = json.load(open(args.config_file))
    (
        fold_config,
        lit_model_params,
        model_config,
        data_module_config,
        trainer_config,
        wandb_config,
        strategies,
    ) = load_configs(config_json)
    handle_wandb(wandb_config)


    if strategies['weigh_classes']:
        class_weights = calculate_class_weights(fold_config.df, 'label')
    else: class_weights = None

    lit_model_params["class_weights"] = class_weights

    if WANDB_RUN is None:
        # raise Exception("Not using wandb is not yet supported, provide wandb config in json")
        print("Running in no wandb mode")

    if WANDB_RUN is not None:
        WANDB_RUN.config["CONFIG"] = config_json

    os.makedirs(f"{args.output_dir}/models/{key}")
    with open(f"{args.output_dir}/models/{key}/config.json", "w") as json_file:
        json.dump(config_json, json_file)

    lit_model = load_model(model_config, lit_model_params)

    callbacks = prepare_callbacks(key, args.output_dir)

    logger = None
    if WANDB_RUN is not None:
        wandb_logger = WandbLogger(log_model=True)
        wandb_logger.watch(lit_model)
        logger = wandb_logger

    trainer = pl.Trainer(**trainer_config, callbacks=callbacks, logger=logger)

    data_module = PhysionetDataModule(folds_config=fold_config, **data_module_config)

    trainer.fit(lit_model, datamodule=data_module)

    lit_model = GuangzhouLitModel.load_from_checkpoint(f"{args.output_dir}/models/{key}/model.ckpt")
    trainer.test(lit_model, data_module)

    if fold_config.folds_pred:  # if anything to make predictions on
        os.makedirs(f"{args.output_dir}/models/{key}/preds")
        for pred_group, folds in fold_config.folds_pred.items():
            dataset = FromDfImageDataset(
                fold_config.df[fold_config.df["fold"].isin(folds)],
                root=data_module.data_dir,
                transform=data_module.val_test_transform,
                label_df=fold_config.label_df
            )
            dataloader = DataLoader(dataset, batch_size=data_module.batch_size, num_workers=data_module.num_workers)
            preds_tensors = trainer.predict(lit_model, dataloader)

            preds = torch.cat([x[0] for x in preds_tensors])
            labels = torch.cat([x[1] for x in preds_tensors])

            torch.save(preds, f"{args.output_dir}/models/{key}/preds/{pred_group}_preds.pt")
            torch.save(labels, f"{args.output_dir}/models/{key}/preds/{pred_group}_y.pt")
        if WANDB_RUN is not None:
            artifact = wandb.Artifact("preds" + WANDB_RUN.name, type="preds")
            artifact.add_dir(f"{args.output_dir}/models/{key}")
            # artifact.add_file(local_path = f"{args.output_dir}/models/{key}/model.ckpt", name = "model")
            WANDB_RUN.log_artifact(artifact)
            artifact.wait()


if __name__ == "__main__":
    args = parse_args()
    main(args)
