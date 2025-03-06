import numpy as np

class Loss:
    def forward(self, y_pred, y_true):
        raise NotImplementedError
    def backward(self):
        raise NotImplementedError

class CrossEntropyLoss(Loss):
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return -np.sum(y_true * np.log(y_pred + 1e-8)) / len(y_true)

    def backward(self):
        return self.y_pred - self.y_true