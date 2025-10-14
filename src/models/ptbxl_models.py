from torch import nn


class ModelWithActivation(nn.Module):
    def __init__(self, model, activation):
        super(ModelWithActivation, self).__init__()
        self.model = model
        self.activation = activation

    def forward(self, x):
        x = self.model(x)
        x = self.activation(x)
        return x
