import wandb
import pandas as pd

from pathlib import Path


class ImageErrorTable:
    LOSS_COLUMN = "loss"
    LABEL_COLUMN = "label"
    EPOCH_COLUMN = "epoch"
    PATH_COLUMN = "path"
    COLUMNS = [LOSS_COLUMN, LABEL_COLUMN, EPOCH_COLUMN, PATH_COLUMN]

    def __init__(self, top_n=5, base_path="."):
        self.wandb_table = None
        self.top_n = top_n
        self.accumulator = []
        self.base_path = Path(base_path)

    def accumulate(self, losses, paths, labels, epoch):
        self.accumulator.extend([
            {self.LOSS_COLUMN: loss, self.LABEL_COLUMN: label, self.EPOCH_COLUMN: epoch, self.PATH_COLUMN: path}
            for loss, label, path in zip(losses, labels, paths)
        ])

    def update(self):
        if not self.accumulator:
            return

        df = pd.DataFrame(self.accumulator)
        self.accumulator = []

        if self.wandb_table is None:
            self.wandb_table = wandb.Table(
                columns=[self.LOSS_COLUMN, "image", self.LABEL_COLUMN, "mode", self.EPOCH_COLUMN, self.PATH_COLUMN]
            )

        for mode, ascending in [("max", False), ("min", True)]:
            self._add_to_table(df, mode, ascending)

    def _add_to_table(self, df, mode, ascending):
        sorted_df = df.sort_values(by=[self.LOSS_COLUMN], ascending=ascending).head(self.top_n)

        for _, row in sorted_df.iterrows():
            self.wandb_table.add_data(
                row[self.LOSS_COLUMN],
                wandb.Image(str(self.base_path / row[self.PATH_COLUMN])),
                row[self.LABEL_COLUMN],
                mode,
                row[self.EPOCH_COLUMN],
                row[self.PATH_COLUMN],
            )
