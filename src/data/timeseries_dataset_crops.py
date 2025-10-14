import torch
import random


def seed_everything(seed: int):
    import random
    import os
    import torch
    import numpy as np

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(42)


class TimeseriesDatasetCrops(torch.utils.data.Dataset):
    def __init__(self, X, y, output_size, train=True, stride=None, transforms=[], time_dim=-1, batch_dim=0):
        self.X = X
        self.y = y

        self.output_size = output_size
        self.train = train
        self.stride = stride
        self.time_dim = time_dim
        self.batch_dim = batch_dim
        self.transforms = transforms

        assert self.X.shape[self.time_dim] % self.output_size == 0

        if not self.train:
            assert self.X.shape[self.time_dim] % self.stride == 0

    def __len__(self):
        return self.X.shape[self.batch_dim]

    def __getitem__(self, idx):
        if self.train:
            start_idx = random.randint(0, self.X.shape[self.time_dim] - self.output_size)
            end_idx = start_idx + self.output_size
            crop = self.X[idx, :, start_idx:end_idx], self.y[idx]
            for t in self.transforms:
                crop = t(crop)
            return crop
        else:
            start_idx = 0
            end_idx = self.X.shape[self.time_dim] - self.output_size
            crops = []
            for i in range(start_idx, end_idx, self.stride):
                crop = self.X[idx, :, i : i + self.output_size]
                if crop.shape[0] != 0:
                    for t in self.transforms:
                        crop = t(crop)
                    crops.append(crop)
            return torch.stack(crops), self.y[idx]
