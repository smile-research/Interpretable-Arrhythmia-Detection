class StandardScalerTorch:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X, axis=(0, 1)):
        self.mean = X.mean(axis=axis)
        self.std = X.std(axis=axis)

    def transform(self, X):
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        return X * self.std + self.mean
