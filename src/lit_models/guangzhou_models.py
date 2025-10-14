import pytorch_lightning as pl
import torch
import torch.nn.functional as F


from src.utils.metrics import insert_metrics
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau


import numpy as np
from captum.attr import Saliency
from captum.attr import visualization as viz


# from src.utils.wandb_error_table import ImageErrorTable
from src.models.guangzhou_models import MODEL_MAPPING


class GuangzhouLitModel(pl.LightningModule):
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
        super(GuangzhouLitModel, self).__init__()

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

        # self.error_table = error_table # we must find a way to do it differently that providing it as arg

        self.save_hyperparameters()

    def forward(self, x):
        # nans before forward pass?
        if x.isnan().any().item():
            print("NaN input")

        out_model = self.model(x)

        # nans after forward pass?
        if out_model.isnan().any().item():
            print("NaN output")

        return out_model

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_pred = self.model(x)

        
        loss = F.binary_cross_entropy(y_pred, y.view(*y_pred.shape), reduction="none")

        if self.class_weights is not None:
            weight = self.class_weights[:, y.long()]
            weight = weight.view(*y_pred.shape) # check if it works after other changes
            loss = loss * weight
        loss = loss.mean()

        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True, on_step=True)
        self._calculate_metrics(self.train_metrics, y_pred, y.view(*y_pred.shape))

        return loss

    def validation_step(self, batch, sbatch_idx):
        x, y, path = batch  # that should call some kind of loader specific unpacker, self.trainer.blabla
        y_pred = self.model(x)


        loss = F.binary_cross_entropy(y_pred, y.view(*y_pred.shape), reduction="none")

        if self.class_weights is not None:
            weight = self.class_weights[:, y.long()]
            weight = weight.view(*y_pred.shape) # check if it works after other changes
            loss = loss * weight
        loss = loss.mean()

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True, on_step=True)
        self._calculate_metrics(self.val_metrics, y_pred, y.view(*y_pred.shape))

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y, _ = batch
        return self(x), y

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        y_pred = self.model(x)
        self._calculate_metrics(self.test_metrics, y_pred, y.view(*y_pred.shape))

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

    def on_validation_epoch_end(self):
        # self.error_table.update()
        pass

    def on_train_end(self):
        # table = self.error_table.wandb_table
        # wandb.log({f"loss_table_{wandb.run.name}": table})
        pass

    def _configure_metrics(
        self,
    ):
        task = 'binary' if self.n_labels == 1 else 'multilabel' # for now does not support multiclass
        metrics_dict_train = insert_metrics(self, prefix="train", task=task, num_classes=self.n_labels)
        metrics_dict_val = insert_metrics(self, prefix="val", task=task, num_classes=self.n_labels)
        metrics_dict_test = insert_metrics(self, prefix="test", task=task, num_classes=self.n_labels)

        return metrics_dict_train, metrics_dict_val, metrics_dict_test

    def _calculate_metrics(self, metrics_dict, y_pred, y_true):
        for metric_name, metric_object in metrics_dict.items():
            metric_object(y_pred, y_true.int())
            self.log(metric_name, metric_object, on_step=True, on_epoch=True)

    def explain_image(self, x, y):
        saliency = Saliency(self.model)
        grads = saliency.attribute([x], target=[y])
        grads = np.transpose(grads.squeeze().cpu().detach().numpy(), (1, 2, 0))
        original_image = np.transpose((x.cpu().detach().numpy() / 2) + 0.5, (1, 2, 0))
        test = viz.visualize_image_attr(
            grads,
            original_image,
            method="blended_heat_map",
            sign="absolute_value",
            show_colorbar=True,
            title="Overlayed Gradient Magnitudes",
        )
        return test[0]
