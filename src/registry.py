from src.models.resnet1d import resnet1d_wang
from src.models.conv_transformer import conv_transformer
from src.models.inception1d import inception1d
from src.models.rnn1d import RNN1d
from src.models.xresnet1d import xresnet1d18
from src.modules.gev import GEV

import torch


def get_model_registry():
    return {
        "resnet1d_wang": resnet1d_wang,
        "conv_transformer": conv_transformer,
        "inception1d": inception1d,
        "RNN1d": RNN1d,
        "xresnet1d18": xresnet1d18,
    }


def get_activation_functions():
    return {"gev": GEV, "sigmoid": torch.nn.Sigmoid}
