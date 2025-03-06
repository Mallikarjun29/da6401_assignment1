import numpy as np

class Layer:
    def __init__(self, input_size, output_size, weight_init='random'):
        self.input_size = input_size
        self.output_size = output_size
        if weight_init == 'random':
            self.weights = np.random.randn(input_size, output_size) * 0.01
        elif weight_init == 'xavier':
            self.weights = np.random.randn(input_size, output_size) * np.sqrt(2 / (input_size + output_size))
        self.biases = np.zeros((1, output_size))
        self.m_w = None
        self.v_w = None
        self.m_b = None
        self.v_b = None
        self.t = 0

    def forward(self, input_data):
        self.input = input_data.reshape(-1, self.input_size)
        return np.dot(self.input, self.weights) + self.biases

    def backward(self, output_grad, optimizer):
        input_grad = np.dot(output_grad, self.weights.T)
        weights_grad = np.dot(self.input.T, output_grad)
        biases_grad = np.sum(output_grad, axis=0, keepdims=True)
        
        # Update parameters using the optimizer
        optimizer.update(self, weights_grad, biases_grad)
        
        return input_grad