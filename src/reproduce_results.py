from pathlib import Path

import torch
import pytorch_lightning as pl
import wandb

from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor

from src.lit_models.ptbxl_model import ECGClassifier
from src.models.resnet1d import resnet1d_wang
from pytorch_lightning.loggers import WandbLogger

from src.data.ptb_xl_multiclass_datamodule import PTB_XL_Datamodule

import os
from datetime import datetime


def create_directory_with_timestamp(path, prefix):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"{prefix}_{timestamp}"
    full_path = os.path.join(path, dir_name)
    os.makedirs(full_path, exist_ok=True)

    return full_path


BATCH_SIZE = 128
EPOCHS = 50
ACCUMULATE_GRADIENT_STEPS = 1
FILTER_FOR_SINGLELABEL = False

loss = torch.nn.BCEWithLogitsLoss() if FILTER_FOR_SINGLELABEL else torch.nn.CrossEntropyLoss()


run = wandb.init(project="ecg_benchmarking_lit", name="test_run", entity="phd-dk")
artifact = run.use_artifact(f"{'ptbxl_split'}:latest")

datadir = artifact.download()

data_module = PTB_XL_Datamodule(Path(datadir), filter_for_singlelabel=FILTER_FOR_SINGLELABEL, batch_size=BATCH_SIZE)

data_module.prepare_data()
data_module.setup()

print(len(data_module.val_dataset))

total_optimizer_steps = int(len(data_module.train_dataset) * EPOCHS / ACCUMULATE_GRADIENT_STEPS)

# Initialize W&B

model = resnet1d_wang(
    num_classes=5,
    input_channels=12,
    kernel_size=5,
    ps_head=0.5,
    lin_ftrs_head=[128],
)

model_lit = ECGClassifier(
    model, 5, torch.nn.BCEWithLogitsLoss(), 0.01, wd=0.01, total_optimizer_steps=total_optimizer_steps
)
wandb_logger = WandbLogger(log_model="all")
wandb_logger.watch(model_lit, log="all")

dir_model = create_directory_with_timestamp("./models", "resnet1d_wang")

early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=30, verbose=False, mode="min")
learning_rate_monitor = LearningRateMonitor(logging_interval="step", log_momentum=True)

# Create the Learner
trainer = pl.Trainer(
    accumulate_grad_batches=8,
    log_every_n_steps=1,
    max_epochs=50,
    logger=wandb_logger,
    callbacks=[early_stop_callback, learning_rate_monitor],
)

trainer.fit(model_lit, datamodule=data_module)

res = trainer.predict(dataloaders=data_module.test_dataloader())

y_hat, y = torch.concatenate([x[0] for x in res]), torch.concatenate([x[1] for x in res])


print(y_hat.shape)
print(y.shape)

print(len(data_module.test_dataloader()) * 128)


torch.save({"y_pred": y_hat, "y": y}, "prediction_data_pytorch_lightning_50.pt")
