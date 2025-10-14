from torchvision import transforms
import random
import logging

from src.data.folded_df_image_data_module import FoldedDfImageDataModule
from src.data_model.fold_config import FoldConfig


class PhysionetDataModule(FoldedDfImageDataModule):
    def __init__(self, folds_config: FoldConfig, normalization_values, data_dir="./data", batch_size=2, num_workers=1):
        self.normalization_values = normalization_values
        logging.error(f"{normalization_values=}")
        super().__init__(folds_config, data_dir, batch_size, num_workers)
        random.seed(42)

    def train_transforms(self):
        train_transforms = [
            transforms.Resize((512, 1024)),
            transforms.RandomRotation(5),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x[:3, ...]),
            transforms.Normalize(*self.normalization_values)
        ]
        if self.normalization_values is not None:
            train_transforms.append(
                transforms.Normalize(*self.normalization_values)
            )
        return transforms.Compose(train_transforms)

    def val_test_transforms(self):
        val_transforms = [
            transforms.Resize((512, 1024)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x[:3, ...]),
            transforms.Normalize(*self.normalization_values)
        ]
        if self.normalization_values is not None:
            val_transforms.append(
                transforms.Normalize(*self.normalization_values)
            )
        return transforms.Compose(val_transforms)
