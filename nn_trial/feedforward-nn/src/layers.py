class Dense:
    def __init__(self, input_size, output_size, activation=None):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros((1, output_size))
        self.input = None
        self.output = None

    def forward(self, input_data):
        self.input = input_data
        self.output = np.dot(input_data, self.weights) + self.biases
        if self.activation:
            self.output = self.activation.forward(self.output)
        return self.output

    def backward(self, output_gradient, learning_rate):
        if self.activation:
            output_gradient = self.activation.backward(output_gradient)
        weights_gradient = np.dot(self.input.T, output_gradient)
        input_gradient = np.dot(output_gradient, self.weights.T)

        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * np.sum(output_gradient, axis=0, keepdims=True)

        return input_gradient


class Activation:
    def forward(self, input_data):
        raise NotImplementedError

    def backward(self, output_gradient):
        raise NotImplementedError


class ReLU(Activation):
    def forward(self, input_data):
        self.input = input_data
        return np.maximum(0, input_data)

    def backward(self, output_gradient):
        return output_gradient * (self.input > 0)


class Sigmoid(Activation):
    def forward(self, input_data):
        self.output = 1 / (1 + np.exp(-input_data))
        return self.output

    def backward(self, output_gradient):
        return output_gradient * self.output * (1 - self.output)