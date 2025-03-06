import numpy as np

class Activation:
    def forward(self, input_data):
        raise NotImplementedError

    def backward(self, output_grad):
        raise NotImplementedError

class ReLU(Activation):
    def forward(self, input_data):
        self.input = input_data
        return np.maximum(0, input_data)

    def backward(self, output_grad, optimizer = None):
        return output_grad * (self.input > 0)

class Sigmoid(Activation):
    def forward(self, input_data):
        self.output = 1 / (1 + np.exp(-input_data))
        return self.output

    def backward(self, output_grad, optimizer = None):
        return output_grad * self.output * (1 - self.output)

class Tanh(Activation):
    def forward(self, input_data):
        self.output = np.tanh(input_data)
        return self.output

    def backward(self, output_grad, optimizer = None):
        return output_grad * (1 - self.output**2)

class Softmax(Activation):
    def forward(self, input_data):
        exp_values = np.exp(input_data - np.max(input_data, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.output

    def backward(self, output_grad, optimizer = None):
        return output_grad