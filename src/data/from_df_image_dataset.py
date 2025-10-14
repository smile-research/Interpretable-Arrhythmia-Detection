from PIL import Image
from torchvision.datasets import VisionDataset
import torch

from pathlib import Path


class FromDfImageDataset(VisionDataset):
    def __init__(self, df, root, transforms=None, transform=None, target_transform=None, label_df=None):
        super().__init__(root, transforms, transform, target_transform)
        self.df = df
        self.label_df = label_df
        self.root_path = Path(root)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        path = self.root_path / self.df.iloc[index]["path"]

        if self.label_df is not None:  # useful for multilabel case
            label = torch.tensor(self.label_df.iloc[index].values).float().squeeze()
        else:
            label = self.df.iloc[index]["label"]

        image = Image.open(path)
        if self.transform:
            try:
                image = self.transform(image)
            except Exception as ex:
                print(f"Something failed for image on this path : {path}")
                raise ex

        return image, label, self.df.iloc[index]["path"]
