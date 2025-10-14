import torch

import pandas as pd
import numpy as np

import pytorch_lightning as pl

from src.utils.preprocessing import StandardScalerTorch
from .timeseries_dataset_crops import TimeseriesDatasetCrops


from pathlib import Path
from torch.utils.data import DataLoader

from typing import NamedTuple


class FoldsConfig(NamedTuple):
    df_metadata: pd.DataFrame
    df_labels: pd.DataFrame
    raw_data: np.array
    folds_train: list
    folds_val: list
    folds_test: list


class PTB_XL_Datamodule_Folds(pl.LightningDataModule):
    def __init__(
        self,
        folds_config,
        batch_size=32,
        filter_for_singlelabel=False,
        filter_no_labels=True,
        raw_data_path=Path("./data/ptbxl"),
    ):
        super().__init__()

        self.scaler = StandardScalerTorch()
        self.batch_size = batch_size
        self.filter_for_singlelabel = filter_for_singlelabel
        self.filter_no_labels = filter_no_labels
        self.folds_config = folds_config
        self.raw_data_path = raw_data_path

    def prepare_data(self):
        self._load_data()
        if self.filter_for_singlelabel:
            self._filter_for_singlelabel()
        elif self.filter_no_labels:
            self._filter_no_labels()
        self._scale()

    def setup(self, stage=None):
        self.train_dataset = TimeseriesDatasetCrops(
            self.X_train,
            self.y_train,
            output_size=250,
            train=True,
            stride=None,
            transforms=[],
            time_dim=-1,
            batch_dim=0,
        )
        self.val_dataset = TimeseriesDatasetCrops(
            self.X_val, self.y_val, output_size=250, train=False, stride=125, transforms=[], time_dim=-1, batch_dim=0
        )
        self.test_dataset = TimeseriesDatasetCrops(
            self.X_test, self.y_test, output_size=250, train=False, stride=125, transforms=[], time_dim=-1, batch_dim=0
        )

    def train_dataloader(self, shuffle=True):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=shuffle)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def _load_data(self):
        df = self.folds_config.df_metadata
        folds_train = self.folds_config.folds_train
        folds_val = self.folds_config.folds_val
        folds_test = self.folds_config.folds_test

        if all([df is not None, folds_train, folds_val, folds_test]):
            df_train = df[df["new_fold"].isin(folds_train)]
            df_val = df[df["new_fold"].isin(folds_val)]
            df_test = df[df["new_fold"].isin(folds_test)]

            self.X_train = torch.tensor(self.folds_config.raw_data[df_train.index], dtype=torch.float32).permute(
                0, 2, 1
            )
            self.y_train = torch.tensor(
                self.folds_config.df_labels.iloc[df_train.index, 2:].values, dtype=torch.float32
            )
            self.X_val = torch.tensor(self.folds_config.raw_data[df_val.index], dtype=torch.float32).permute(0, 2, 1)
            self.y_val = torch.tensor(self.folds_config.df_labels.iloc[df_val.index, 2:].values, dtype=torch.float32)
            self.X_test = torch.tensor(self.folds_config.raw_data[df_test.index], dtype=torch.float32).permute(0, 2, 1)
            self.y_test = torch.tensor(self.folds_config.df_labels.iloc[df_test.index, 2:].values, dtype=torch.float32)

    def _filter_no_labels(self):
        mask_y_val = self.y_val.sum(axis=1) > 0
        mask_y_test = self.y_test.sum(axis=1) > 0
        mask_y_train = self.y_train.sum(axis=1) > 0

        self.X_train = self.X_train[mask_y_train]
        self.y_train = self.y_train[mask_y_train]
        self.X_val = self.X_val[mask_y_val]
        self.y_val = self.y_val[mask_y_val]
        self.X_test = self.X_test[mask_y_test]
        self.y_test = self.y_test[mask_y_test]

    def _filter_for_singlelabel(self):
        mask_y_val = self.y_val.sum(axis=1) == 1
        mask_y_test = self.y_test.sum(axis=1) == 1
        mask_y_train = self.y_train.sum(axis=1) == 1

        self.X_train = self.X_train[mask_y_train]
        self.y_train = self.y_train[mask_y_train]
        self.X_val = self.X_val[mask_y_val]
        self.y_val = self.y_val[mask_y_val]
        self.X_test = self.X_test[mask_y_test]
        self.y_test = self.y_test[mask_y_test]

    def _scale(self):
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_val = self.scaler.transform(self.X_val)
        self.X_test = self.scaler.transform(self.X_test)
