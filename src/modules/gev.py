import torch
import torch.nn as nn


class GEV(nn.Module):
    def __init__(self):
        super(GEV, self).__init__()
        self.mu = nn.Parameter(torch.tensor(0.0))  # Trainable parameter
        self.sigma = nn.Parameter(torch.tensor(1.0))  # Trainable parameter
        self.xi = nn.Parameter(torch.tensor(0.0))  # Trainable parameter

    def forward(self, x):
        # Ensure sigma is positive and not too close to zero to avoid large values in division
        sigma = torch.clamp(self.sigma, min=torch.finfo(self.sigma.dtype).eps, max=1e10)

        # Function for xi = 0 (Gumbel)
        def t1(x):
            z = -(x - self.mu) / sigma
            # Clipping to prevent overflow in exponential
            z = torch.clamp(z, min=-100, max=100)
            return torch.exp(-torch.exp(z))

        # Function for xi != 0 (Frechet for xi > 0 or Reversed Weibull for xi < 0)
        def t23(x):
            y = (x - self.mu) / sigma
            # Adding a small constant to xi to prevent division by zero
            xi_adj = self.xi + torch.where(self.xi == 0, 1e-6, 0)
            z = xi_adj * y
            # Clipping to prevent negative values inside the power operation
            z = torch.clamp(z, min=-0.9999)
            t = torch.pow(1.0 + z, -1.0 / xi_adj)
            # Clipping to prevent overflow in exponential
            t = torch.clamp(t, max=100)
            return torch.exp(-t)

        # Use t1 when xi is very close to zero, otherwise use t23
        out = torch.where(torch.abs(self.xi) < 1e-6, t1(x), t23(x))

        return out


class LogGev(GEV):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return super().forward(x).log()


class mGEV(nn.Module):
    def __init__(self, num_classes):
        super(mGEV, self).__init__()
        self.mu = nn.Parameter(torch.zeros(num_classes))  # trainable parameter, initialized to 0
        self.sigma = nn.Parameter(torch.ones(num_classes))  # trainable parameter, initialized to 1
        self.xi = nn.Parameter(torch.tensor(0.1))  # trainable parameter, initialized to 0.1

    def forward(self, x):
        mu = self.mu
        sigma = torch.clamp(self.sigma, min=torch.finfo(self.sigma.dtype).eps)  # ensure sigma is positive
        xi = self.xi

        x = torch.clamp(x, -20, 20)  # clipping the inputs

        # Type 1: For xi = 0 (Gumbel)
        def t1():
            return torch.exp(-torch.exp(-(x - mu) / sigma))

        # Type 2: For xi > 0 (Frechet) or xi < 0 (Reversed Weibull)
        def t23():
            y = (x - mu) / sigma
            y = xi * y
            y = torch.maximum(y, torch.tensor(-1.0))
            y = torch.exp(-torch.pow(torch.tensor(1.0) + y, -1.0 / xi))
            return y

        mGEV = torch.where(xi == 0, t1(), t23())
        mGEV = mGEV / torch.sum(mGEV, dim=1, keepdim=True)  # Normalizing to make the sum 1
        return mGEV


class LogMGev(mGEV):
    def __init__(self, num_classes):
        super().__init__(num_classes)

    def forward(self, x):
        return super().forward(x).log()
