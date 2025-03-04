import numpy as np
from keras.datasets import fashion_mnist

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

class Activation:
    def forward(self, input_data):
        raise NotImplementedError

    def backward(self, output_grad):
        raise NotImplementedError

class ReLU(Activation):
    def forward(self, input_data):
        self.input = input_data
        return np.maximum(0, input_data)

    def backward(self, output_grad):
        return output_grad * (self.input > 0)

class Sigmoid(Activation):
    def forward(self, input_data):
        self.output = 1 / (1 + np.exp(-input_data))
        return self.output

    def backward(self, output_grad):
        return output_grad * self.output * (1 - self.output)

class Tanh(Activation):
    def forward(self, input_data):
        self.output = np.tanh(input_data)
        return self.output

    def backward(self, output_grad):
        return output_grad * (1 - self.output**2)

class Softmax(Activation):
    def forward(self, input_data):
        exp_values = np.exp(input_data - np.max(input_data, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.output

    def backward(self, output_grad):
        return output_grad

class CrossEntropyLoss:
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return -np.sum(y_true * np.log(y_pred + 1e-8)) / len(y_true)

    def backward(self):
        return self.y_pred - self.y_true

class Optimizer:
    def update(self, layer, weights_grad, biases_grad):
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, learning_rate=0.01, weight_decay=0):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def update(self, layer, weights_grad, biases_grad):
        layer.weights -= self.learning_rate * (weights_grad + self.weight_decay * layer.weights)
        layer.biases -= self.learning_rate * biases_grad

class Momentum(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.9, weight_decay=0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity_w = None
        self.velocity_b = None

    def update(self, layer, weights_grad, biases_grad):
        if self.velocity_w is None:
            self.velocity_w = np.zeros_like(layer.weights)
            self.velocity_b = np.zeros_like(layer.biases)
        self.velocity_w = self.momentum * self.velocity_w - self.learning_rate * (weights_grad + self.weight_decay * layer.weights)
        self.velocity_b = self.momentum * self.velocity_b - self.learning_rate * biases_grad
        layer.weights += self.velocity_w
        layer.biases += self.velocity_b

class Nesterov(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.9, weight_decay=0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity_w = None
        self.velocity_b = None

    def update(self, layer, weights_grad, biases_grad):
        if self.velocity_w is None:
            self.velocity_w = np.zeros_like(layer.weights)
            self.velocity_b = np.zeros_like(layer.biases)
        prev_velocity_w = self.velocity_w
        prev_velocity_b = self.velocity_b
        self.velocity_w = self.momentum * self.velocity_w - self.learning_rate * (weights_grad + self.weight_decay * layer.weights)
        self.velocity_b = self.momentum * self.velocity_b - self.learning_rate * biases_grad
        layer.weights += -self.momentum * prev_velocity_w + (1 + self.momentum) * self.velocity_w
        layer.biases += -self.momentum * prev_velocity_b + (1 + self.momentum) * self.velocity_b

class RMSprop(Optimizer):
    def __init__(self, learning_rate=0.001, beta=0.9, epsilon=1e-8, weight_decay=0):
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.s_w = None
        self.s_b = None

    def update(self, layer, weights_grad, biases_grad):
        if self.s_w is None:
            self.s_w = np.zeros_like(layer.weights)
            self.s_b = np.zeros_like(layer.biases)
        self.s_w = self.beta * self.s_w + (1 - self.beta) * weights_grad**2
        self.s_b = self.beta * self.s_b + (1 - self.beta) * biases_grad**2
        layer.weights -= self.learning_rate * (weights_grad / (np.sqrt(self.s_w) + self.epsilon) + self.weight_decay * layer.weights)
        layer.biases -= self.learning_rate * biases_grad / (np.sqrt(self.s_b) + self.epsilon)

class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay

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
        layer.weights -= self.learning_rate * (m_hat_w / (np.sqrt(v_hat_w) + self.epsilon) + self.weight_decay * layer.weights)
        layer.biases -= self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)

class Nadam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay

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
        layer.weights -= self.learning_rate * (self.beta1 * m_hat_w + (1 - self.beta1) * weights_grad / (1 - self.beta1**layer.t)) / (np.sqrt(v_hat_w) + self.epsilon) + self.weight_decay * layer.weights
        layer.biases -= self.learning_rate * (self.beta1 * m_hat_b + (1 - self.beta1) * biases_grad / (1 - self.beta1**layer.t)) / (np.sqrt(v_hat_b) + self.epsilon)

class NeuralNetwork:
    def __init__(self, layers, optimizer_class, optimizer_params):
        self.layers = layers
        self.optimizers = [optimizer_class(**optimizer_params) for _ in layers if isinstance(_, Layer)]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, output_grad):
        for layer, optimizer in zip(reversed(self.layers), reversed(self.optimizers)):
            if isinstance(layer, Layer):
                output_grad = layer.backward(output_grad, optimizer)
            else:
                output_grad = layer.backward(output_grad)

def main():
    # Initialize the optimizer parameters
    optimizer_params = {'learning_rate': 0.001}

    # Initialize the network layers
    layer1 = Layer(784, 128)
    activation1 = ReLU()
    layer2 = Layer(128, 10)
    activation2 = Softmax()

    # Create the neural network with the optimizer class and parameters
    network = NeuralNetwork([layer1, activation1, layer2, activation2], Adam, optimizer_params)

    loss = CrossEntropyLoss()

    # Load the dataset
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    # Preprocess the data
    x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
    x_test = x_test.reshape(x_test.shape[0], -1) / 255.0

    # Forward pass
    output = network.forward(x_train[0].reshape(1, -1))
    
    # Backward pass
    y_train_one_hot = np.eye(10)[y_train[0]].reshape(1, -1)  # One-hot encode the label
    loss_val = loss.forward(output, y_train_one_hot)
    output_grad = loss.backward()
    network.backward(output_grad)

    print("output", output)
    print("output_grad", output_grad)
    print("weights layer1", layer1.weights)
    print("biases layer1", layer1.biases)
    print("weights layer2", layer2.weights)
    print("biases layer2", layer2.biases)

if __name__ == "__main__":
    main()