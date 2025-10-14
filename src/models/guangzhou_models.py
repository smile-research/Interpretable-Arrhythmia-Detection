from torchvision.models import (
    resnet18,
    resnet34,
    googlenet,
    inception_v3,
    efficientnet_v2_s,
    mobilenet_v3_small,
    efficientnet_v2_m,
    efficientnet_v2_l,
)

from torch import nn
import torchvision
from src.modules.gev import GEV
import torch
from functools import partial


def freeze_layer(layer):
    for param in layer.parameters():
        param.requires_grad = False


# TODO: Doesn't work with multiclass; multilabel is ok


class InceptionBased(nn.Module):
    def __init__(self, activation="gev", n_labels=1):
        super(InceptionBased, self).__init__()
        self.inception = inception_v3(pretrained=True)
        self.inception.fc = nn.Identity()
        self.fc = nn.Linear(2048, n_labels)

        if activation == "gev":
            self.activation = GEV()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x):
        x = self.inception(x)
        if not isinstance(x, torch.Tensor):
            x = x.logits

        if x.isnan().any().item():
            print("NaN output")

        x = self.fc(x)

        if x.isnan().any().item():
            print("NaN output")

        x = self.activation(x)
        return x


class GooglenetBased(nn.Module):
    def __init__(self, activation="gev", n_labels=1):
        super(GooglenetBased, self).__init__()
        self.googlenet = googlenet(pretrained=True)
        self.googlenet.fc = nn.Identity()

        self.fc = nn.Linear(1024, n_labels)

        if activation == "gev":
            self.activation = GEV()
        else:
            self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.googlenet(x)
        x = self.fc(x)
        x = self.activation(x)


class MobileNetV3SBased(nn.Module):
    def __init__(self, activation="gev", n_labels=1):
        super(MobileNetV3SBased, self).__init__()
        self.mobile_net = mobilenet_v3_small(pretrained=True)
        self.mobile_net.fc = nn.Identity()

        self.fc = nn.Linear(1024, n_labels)

        if activation == "gev":
            self.activation = GEV()
        else:
            self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.mobile_net(x)
        x = self.fc(x)
        x = self.activation(x)


class EfficientNetBased(nn.Module):
    def __init__(self, activation="gev", size="s", n_labels=1):
        super(EfficientNetBased, self).__init__()

        size_mapping = {"s": efficientnet_v2_s, "m": efficientnet_v2_m, "l": efficientnet_v2_l}

        self.efficient_net = size_mapping[size](pretrained=True)

        self.dropout = self.efficient_net.classifier[0]
        self.fc = nn.Linear(1280, n_labels)

        self.efficient_net.classifier = nn.Identity()

        if activation == "gev":
            self.activation = GEV()
        else:
            self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.efficient_net(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.activation(x)
        return x


class Resnet18Based(nn.Module):
    def __init__(self, activation="gev", n_labels=1):
        super(Resnet18Based, self).__init__()
        self.resnet = resnet18(pretrained=True)
        self.resnet.fc = nn.Identity()

        self.fc = nn.Linear(512, n_labels)

        if activation == "gev":
            self.activation = GEV()
        else:
            self.activation = nn.Sigmoid()

    def forward(self, x):
        first = self.resnet(x)
        x = self.fc(first)
        x = self.activation(x)
        return x


class Resnet34Based(nn.Module):
    def __init__(self, activation="gev", n_labels=1):
        super(Resnet34Based, self).__init__()
        self.resnet = resnet34(pretrained=True)
        self.resnet.fc = nn.Identity()

        self.fc = nn.Linear(512, n_labels)

        if activation == "gev":
            self.activation = GEV()
        else:
            self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.resnet(x)
        if x.isnan().any().item():
            print("NaN output")
        x = self.fc(x)
        x = self.activation(x)
        return x


MODEL_MAPPING = {
    "inception": InceptionBased,
    "resnet18": Resnet18Based,
    "efficienetS": partial(EfficientNetBased, size="s"),
    "efficienetM": partial(EfficientNetBased, size="m"),
    "efficienetL": partial(EfficientNetBased, size="l"),
}
