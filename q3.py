import numpy as np
from keras.datasets import fashion_mnist
from q2 import FFNN, FFActivation, FFReLU, FFSigmoid

class Layer(FFNN):
    def __init__(self, input_size, output_size, optimizer_class, **optimizer_params):
        super().__init__(input_size, output_size)
        self.optimizer = optimizer_class(**optimizer_params)

    def forward(self, input_data):
        self.input = input_data.reshape(1, -1)  # Ensure input is 2D
        self.output = np.dot(self.input, self.weights) + self.biases
        return self.output

    def backward(self, output_grad):
        input_grad = np.dot(output_grad, self.weights.T)
        weights_grad = np.dot(self.input.T, output_grad)
        biases_grad = np.sum(output_grad, axis=0, keepdims=True)
        
        # Update parameters using the optimizer
        self.optimizer.update(self.weights, self.biases, weights_grad, biases_grad)
        
        return input_grad

class Activation(FFActivation):
    def backward(self, output_grad):
        raise NotImplementedError

class ReLU(FFReLU):
    def backward(self, output_grad):
        return output_grad * (self.input > 0)

class Sigmoid(FFSigmoid):
    def backward(self, output_grad):
        return output_grad * self.output * (1 - self.output)

class Optimizer:
    def update(self, weights, biases, weights_grad, biases_grad):
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, weights, biases, weights_grad, biases_grad):
        weights -= self.learning_rate * weights_grad
        biases -= self.learning_rate * biases_grad

class Momentum(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity_w = None
        self.velocity_b = None

    def update(self, weights, biases, weights_grad, biases_grad):
        if self.velocity_w is None:
            self.velocity_w = np.zeros_like(weights)
            self.velocity_b = np.zeros_like(biases)
        self.velocity_w = self.momentum * self.velocity_w - self.learning_rate * weights_grad
        self.velocity_b = self.momentum * self.velocity_b - self.learning_rate * biases_grad
        weights += self.velocity_w
        biases += self.velocity_b

class Nesterov(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity_w = None
        self.velocity_b = None

    def update(self, weights, biases, weights_grad, biases_grad):
        if self.velocity_w is None:
            self.velocity_w = np.zeros_like(weights)
            self.velocity_b = np.zeros_like(biases)
        prev_velocity_w = self.velocity_w
        prev_velocity_b = self.velocity_b
        self.velocity_w = self.momentum * self.velocity_w - self.learning_rate * weights_grad
        self.velocity_b = self.momentum * self.velocity_b - self.learning_rate * biases_grad
        weights += -self.momentum * prev_velocity_w + (1 + self.momentum) * self.velocity_w
        biases += -self.momentum * prev_velocity_b + (1 + self.momentum) * self.velocity_b

class RMSprop(Optimizer):
    def __init__(self, learning_rate=0.001, beta=0.9, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.s_w = None
        self.s_b = None

    def update(self, weights, biases, weights_grad, biases_grad):
        if self.s_w is None:
            self.s_w = np.zeros_like(weights)
            self.s_b = np.zeros_like(biases)
        self.s_w = self.beta * self.s_w + (1 - self.beta) * weights_grad**2
        self.s_b = self.beta * self.s_b + (1 - self.beta) * biases_grad**2
        weights -= self.learning_rate * weights_grad / (np.sqrt(self.s_w) + self.epsilon)
        biases -= self.learning_rate * biases_grad / (np.sqrt(self.s_b) + self.epsilon)

class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_w = None
        self.v_w = None
        self.m_b = None
        self.v_b = None
        self.t = 0

    def update(self, weights, biases, weights_grad, biases_grad):
        if self.m_w is None:
            self.m_w = np.zeros_like(weights)
            self.v_w = np.zeros_like(weights)
            self.m_b = np.zeros_like(biases)
            self.v_b = np.zeros_like(biases)
        self.t += 1
        self.m_w = self.beta1 * self.m_w + (1 - self.beta1) * weights_grad
        self.v_w = self.beta2 * self.v_w + (1 - self.beta2) * weights_grad**2
        self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * biases_grad
        self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * biases_grad**2
        m_hat_w = self.m_w / (1 - self.beta1**self.t)
        v_hat_w = self.v_w / (1 - self.beta2**self.t)
        m_hat_b = self.m_b / (1 - self.beta1**self.t)
        v_hat_b = self.v_b / (1 - self.beta2**self.t)
        weights -= self.learning_rate * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)
        biases -= self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)

class Nadam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_w = None
        self.v_w = None
        self.m_b = None
        self.v_b = None
        self.t = 0

    def update(self, weights, biases, weights_grad, biases_grad):
        if self.m_w is None:
            self.m_w = np.zeros_like(weights)
            self.v_w = np.zeros_like(weights)
            self.m_b = np.zeros_like(biases)
            self.v_b = np.zeros_like(biases)
        self.t += 1
        self.m_w = self.beta1 * self.m_w + (1 - self.beta1) * weights_grad
        self.v_w = self.beta2 * self.v_w + (1 - self.beta2) * weights_grad**2
        self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * biases_grad
        self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * biases_grad**2
        m_hat_w = self.m_w / (1 - self.beta1**self.t)
        v_hat_w = self.v_w / (1 - self.beta2**self.t)
        m_hat_b = self.m_b / (1 - self.beta1**self.t)
        v_hat_b = self.v_b / (1 - self.beta2**self.t)
        weights -= self.learning_rate * (self.beta1 * m_hat_w + (1 - self.beta1) * weights_grad / (1 - self.beta1**self.t)) / (np.sqrt(v_hat_w) + self.epsilon)
        biases -= self.learning_rate * (self.beta1 * m_hat_b + (1 - self.beta1) * biases_grad / (1 - self.beta1**self.t)) / (np.sqrt(v_hat_b) + self.epsilon)

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, output_grad):
        for layer in reversed(self.layers):
            output_grad = layer.backward(output_grad)

# Initialize the network layers with the optimizer
layer1 = Layer(784, 128, Adam, learning_rate=0.001)
activation1 = ReLU()
layer2 = Layer(128, 10, Adam, learning_rate=0.001)
activation2 = Sigmoid()

# Create the neural network
network = NeuralNetwork([layer1, activation1, layer2, activation2])

# Load the dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Preprocess the data
x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
x_test = x_test.reshape(x_test.shape[0], -1) / 255.0

# Forward pass
output = network.forward(x_train[0])

# Backward pass
output_grad = output - y_train[0]  # Assuming y_train is one-hot encoded
network.backward(output_grad)

print("output", output)
print("output_grad", output_grad)
print("weights layer1", layer1.weights)
print("biases layer1", layer1.biases)
print("weights layer2", layer2.weights)
print("biases layer2", layer2.biases)