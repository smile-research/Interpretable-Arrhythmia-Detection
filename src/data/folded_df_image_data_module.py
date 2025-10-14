from src.data_model.fold_config import FoldConfig

from src.data.from_df_image_dataset import FromDfImageDataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import random
import pandas as pd


class FoldedDfImageDataModule(pl.LightningDataModule):
    def __init__(self, folds_config: FoldConfig, data_dir="./data", batch_size=2, num_workers=1):
        super().__init__()
        random.seed(42)

        self.data_dir = data_dir
        self.batch_size = batch_size

        self.train_transform = self.train_transforms()
        self.val_test_transform = self.val_test_transforms()

        self.folds_config = folds_config

        # if not self.folds_config.label_df:
        #     self.build_labels_df()

        self.num_workers = num_workers

    def setup(self, stage="fit"):
        df = self.folds_config.df
        label_df = self.folds_config.label_df
        folds_train = self.folds_config.folds_train
        folds_val = self.folds_config.folds_val
        folds_test = self.folds_config.folds_test

        assert folds_train is not None if stage in ["fit"] else True
        assert folds_val is not None if stage in ["fit"] else True
        assert folds_test is not None if stage in ["fit"] else True

        if folds_train is not None:
            df_train = df[df["fold"].isin(folds_train)]
            label_df_train = None if label_df is None else label_df.loc[df_train.index]
            self.train_dataset = FromDfImageDataset(
                df_train, root=self.data_dir, transform=self.train_transform, label_df=label_df_train
            )

        if folds_val is not None:
            df_val = df[df["fold"].isin(folds_val)]
            label_df_val = None if label_df is None else label_df.loc[df_val.index]
            self.val_dataset = FromDfImageDataset(
                df_val, root=self.data_dir, transform=self.val_test_transform, label_df=label_df_val
            )

        if folds_test is not None:
            df_test = df[df["fold"].isin(folds_test)]
            label_df_test = None if label_df is None else label_df.loc[df_test.index]
            self.test_dataset = FromDfImageDataset(
                df_test, root=self.data_dir, transform=self.val_test_transform, label_df=label_df_test
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def train_transforms(self):
        raise NotImplementedError()

    def val_test_transforms(self):
        raise NotImplementedError()

    def build_labels_df(self):
        df = self.folds_config.df
        df = pd.get_dummies(df,prefix=['label'], columns = ['label'])
        labels_df = df.iloc[:,-2:]
        self.folds_config.label_df = labels_df

