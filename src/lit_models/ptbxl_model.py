import pytorch_lightning as pl
import torch
import torch.nn.functional as F


from src.utils.metrics import insert_metrics
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from torch import nn
from src.models.guangzhou_models import MODEL_MAPPING


class ModelWithActivation(nn.Module):
    def __init__(self, model, activation):
        super(ModelWithActivation, self).__init__()
        self.model = model
        self.activation = activation

    def forward(self, x):
        x = self.model(x)
        x = self.activation(x)
        return x


class ECGClassifier(pl.LightningModule):
    def __init__(
        self,
        model_name,
        activation,
        n_labels,
        learning_rate,
        wd,
        class_weights=None,
        onecycle=True,
        reduce_on_plateau=False,
        **kwargs,
    ):
        super(ECGClassifier, self).__init__()

        self.kwargs = kwargs
        self.model = MODEL_MAPPING[model_name](activation=activation, n_labels=n_labels)

        self.activation = activation
        self.n_labels = n_labels

        self.class_weights = torch.tensor(class_weights).unsqueeze(0).cuda() if class_weights is not None else None

        # Learning rate
        self.learning_rate = learning_rate
        self.wd = wd

        self.train_metrics, self.val_metrics, self.test_metrics = self._configure_metrics()

        self.onecycle = onecycle
        self.reduce_on_plateau = reduce_on_plateau
        self.loss_fn = F.binary_cross_entropy  # TODO: make this configurable

        # self.error_table = error_table # we must find a way to do it differently that providing it as arg

        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = self.loss_fn(y_pred, y)
        self.log(
            "train_loss", loss, on_epoch=True, prog_bar=True, logger=True, on_step=True
        )  # Log to progress bar and logger
        self._calculate_metrics(self.train_metrics, y_pred, y)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_pred = self.model(x).squeeze()

        loss = self.loss_fn(y_pred, y, reduction="none")

        if self.class_weights is not None:
            weight = self.class_weights[:, y.long()].squeeze()
            loss = loss * weight
        loss = loss.mean()

        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True, on_step=True)
        self._calculate_metrics(self.train_metrics, y_pred, y)

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_pred = self._do_pred_over_crops(x)
        return y_pred, y

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self._do_pred_over_crops(x)
        self._calculate_metrics(self.test_metrics, y_pred, y)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.wd)

        if self.onecycle:
            scheduler = OneCycleLR(opt, max_lr=self.learning_rate, total_steps=self.trainer.estimated_stepping_batches)
            lr_scheduler = {"scheduler": scheduler, "interval": "step"}
            out = {"optimizer": opt, "lr_scheduler": lr_scheduler}
        elif self.reduce_on_plateau:
            scheduler = ReduceLROnPlateau(opt, "max", factor=0.67, patience=3, eps=0.0001)
            return [opt], [{"scheduler": scheduler, "interval": "epoch", "monitor": "val_averageprecision_epoch"}]
        else:
            out = opt

        return out

    def _configure_metrics(
        self,
    ):
        metrics_dict_train = insert_metrics(self, self.num_classes, prefix="train", task=self.task)
        metrics_dict_val = insert_metrics(self, self.num_classes, prefix="val", task=self.task)
        metrics_dict_test = insert_metrics(self, self.num_classes, prefix="test", task=self.task)

        return metrics_dict_train, metrics_dict_val, metrics_dict_test

    def _calculate_metrics(self, metrics_dict, y_pred, y_true):
        for metric_name, metric_object in metrics_dict.items():
            # print(metric_name, self.task)
            if self.task == "multiclass":
                y_true = y_true.argmax(dim=1)

            metric_object(y_pred, y_true.int())
            self.log(metric_name, metric_object, on_step=True, on_epoch=True)

    def _do_pred_over_crops(self, x):
        batch_size, n_crops, n_channels, time = x.shape
        x = x.view(-1, n_channels, time)
        y_pred = self.forward(x)
        y_pred = y_pred.view(batch_size, n_crops, -1).max(dim=1).values  # take the max over the crops
        return y_pred
