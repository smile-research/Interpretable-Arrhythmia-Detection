from torchvision import transforms
import torchvision.transforms.functional as TF
import random


from src.data_model.fold_config import FoldConfig
from src.data.folded_df_image_data_module import FoldedDfImageDataModule


class GuangzhouDataModule(FoldedDfImageDataModule):
    def __init__(self, folds_config: FoldConfig, data_dir="./data", batch_size=32, num_workers=1):
        super().__init__(folds_config, data_dir, batch_size, num_workers)

    def train_transforms(self):
        return transforms.Compose([
            transforms.Resize((1024, 512)),
            transforms.RandomRotation(5),
            transforms.Lambda(lambda img: TF.adjust_brightness(img, random.uniform(0.9, 1.1))),
            transforms.ToTensor(),
        ])

    def val_test_transforms(self):
        return transforms.Compose([
            transforms.Resize((1024, 512)),
            transforms.ToTensor(),
        ])
