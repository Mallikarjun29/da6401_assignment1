import numpy as np
from keras.datasets import fashion_mnist
from q2 import FFNN, FFActivation, FFReLU, FFSigmoid

class Layer(FFNN):
    def __init__(self, input_size, output_size):
        super().__init__(input_size, output_size)
        self.m_w = None
        self.v_w = None
        self.m_b = None
        self.v_b = None
        self.t = 0

    def backward(self, output_grad, optimizer):
        input_grad = np.dot(output_grad, self.weights.T)
        weights_grad = np.dot(self.input.T, output_grad)
        biases_grad = np.sum(output_grad, axis=0, keepdims=True)
        
        # Update parameters using the optimizer
        optimizer.update(self, weights_grad, biases_grad)
        
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
    def update(self, layer, weights_grad, biases_grad):
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, layer, weights_grad, biases_grad):
        layer.weights -= self.learning_rate * weights_grad
        layer.biases -= self.learning_rate * biases_grad

class Momentum(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity_w = None
        self.velocity_b = None

    def update(self, layer, weights_grad, biases_grad):
        if self.velocity_w is None:
            self.velocity_w = np.zeros_like(layer.weights)
            self.velocity_b = np.zeros_like(layer.biases)
        self.velocity_w = self.momentum * self.velocity_w - self.learning_rate * weights_grad
        self.velocity_b = self.momentum * self.velocity_b - self.learning_rate * biases_grad
        layer.weights += self.velocity_w
        layer.biases += self.velocity_b

class Nesterov(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity_w = None
        self.velocity_b = None

    def update(self, layer, weights_grad, biases_grad):
        if self.velocity_w is None:
            self.velocity_w = np.zeros_like(layer.weights)
            self.velocity_b = np.zeros_like(layer.biases)
        prev_velocity_w = self.velocity_w
        prev_velocity_b = self.velocity_b
        self.velocity_w = self.momentum * self.velocity_w - self.learning_rate * weights_grad
        self.velocity_b = self.momentum * self.velocity_b - self.learning_rate * biases_grad
        layer.weights += -self.momentum * prev_velocity_w + (1 + self.momentum) * self.velocity_w
        layer.biases += -self.momentum * prev_velocity_b + (1 + self.momentum) * self.velocity_b

class RMSprop(Optimizer):
    def __init__(self, learning_rate=0.001, beta=0.9, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.s_w = None
        self.s_b = None

    def update(self, layer, weights_grad, biases_grad):
        if self.s_w is None:
            self.s_w = np.zeros_like(layer.weights)
            self.s_b = np.zeros_like(layer.biases)
        self.s_w = self.beta * self.s_w + (1 - self.beta) * weights_grad**2
        self.s_b = self.beta * self.s_b + (1 - self.beta) * biases_grad**2
        layer.weights -= self.learning_rate * weights_grad / (np.sqrt(self.s_w) + self.epsilon)
        layer.biases -= self.learning_rate * biases_grad / (np.sqrt(self.s_b) + self.epsilon)

class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def update(self, layer, weights_grad, biases_grad):
        if layer.m_w is None:
            layer.m_w = np.zeros_like(layer.weights)
            layer.v_w = np.zeros_like(layer.weights)
            layer.m_b = np.zeros_like(layer.biases)
            layer.v_b = np.zeros_like(layer.biases)
        layer.t += 1
        layer.m_w = self.beta1 * layer.m_w + (1 - self.beta1) * weights_grad
        layer.v_w = self.beta2 * layer.v_w + (1 - self.beta2) * weights_grad**2
        layer.m_b = self.beta1 * layer.m_b + (1 - self.beta1) * biases_grad
        layer.v_b = self.beta2 * layer.v_b + (1 - self.beta2) * biases_grad**2
        m_hat_w = layer.m_w / (1 - self.beta1**layer.t)
        v_hat_w = layer.v_w / (1 - self.beta2**layer.t)
        m_hat_b = layer.m_b / (1 - self.beta1**layer.t)
        v_hat_b = layer.v_b / (1 - self.beta2**layer.t)
        layer.weights -= self.learning_rate * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)
        layer.biases -= self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)

class Nadam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def update(self, layer, weights_grad, biases_grad):
        if layer.m_w is None:
            layer.m_w = np.zeros_like(layer.weights)
            layer.v_w = np.zeros_like(layer.weights)
            layer.m_b = np.zeros_like(layer.biases)
            layer.v_b = np.zeros_like(layer.biases)
        layer.t += 1
        layer.m_w = self.beta1 * layer.m_w + (1 - self.beta1) * weights_grad
        layer.v_w = self.beta2 * layer.v_w + (1 - self.beta2) * weights_grad**2
        layer.m_b = self.beta1 * layer.m_b + (1 - self.beta1) * biases_grad
        layer.v_b = self.beta2 * layer.v_b + (1 - self.beta2) * biases_grad**2
        m_hat_w = layer.m_w / (1 - self.beta1**layer.t)
        v_hat_w = layer.v_w / (1 - self.beta2**layer.t)
        m_hat_b = layer.m_b / (1 - self.beta1**layer.t)
        v_hat_b = layer.v_b / (1 - self.beta2**layer.t)
        layer.weights -= self.learning_rate * (self.beta1 * m_hat_w + (1 - self.beta1) * weights_grad / (1 - self.beta1**layer.t)) / (np.sqrt(v_hat_w) + self.epsilon)
        layer.biases -= self.learning_rate * (self.beta1 * m_hat_b + (1 - self.beta1) * biases_grad / (1 - self.beta1**layer.t)) / (np.sqrt(v_hat_b) + self.epsilon)

class NeuralNetwork:
    def __init__(self, layers, optimizer):
        self.layers = layers
        self.optimizer = optimizer

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, output_grad):
        for layer in reversed(self.layers):
            if isinstance(layer, Layer):
                output_grad = layer.backward(output_grad, self.optimizer)
            else:
                output_grad = layer.backward(output_grad)

# Initialize the optimizer
optimizer = Adam(learning_rate=0.001)

# Initialize the network layers
layer1 = Layer(784, 128)
activation1 = ReLU()
layer2 = Layer(128, 10)
activation2 = Sigmoid()

# Create the neural network with the optimizer
network = NeuralNetwork([layer1, activation1, layer2, activation2], optimizer)

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