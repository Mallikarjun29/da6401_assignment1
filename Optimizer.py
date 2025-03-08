import numpy as np

class Optimizer:
    def update(self, layer, weights_grad, biases_grad):
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, learning_rate=0.01, weight_decay=0):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def update(self, layer, weights_grad, biases_grad):
        layer.weights -= self.learning_rate * (weights_grad + self.weight_decay * layer.weights) # w = w - lr * (dw + wd * w)
        layer.biases -= self.learning_rate * biases_grad # b = b - lr * db

import numpy as np

class Momentum(Optimizer):
    def __init__(self, learning_rate=0.01, beta=0.9, weight_decay=0):
        self.learning_rate = learning_rate
        self.momentum = beta
        self.weight_decay = weight_decay
        self.velocity_w = None
        self.velocity_b = None

    def update(self, layer, weights_grad, biases_grad):
        if self.velocity_w is None:  # Initialize velocities once
            self.velocity_w = np.zeros_like(layer.weights)
            self.velocity_b = np.zeros_like(layer.biases)

        # Update velocity
        self.velocity_w = self.momentum * self.velocity_w - self.learning_rate * (weights_grad + self.weight_decay * layer.weights) # v = b * v - lr * (dw + wd * w)
        self.velocity_b = self.momentum * self.velocity_b - self.learning_rate * biases_grad # v = b * v - lr * db

        # Apply the update
        layer.weights += self.velocity_w
        layer.biases += self.velocity_b


import numpy as np

class Nesterov(Optimizer):
    def __init__(self, learning_rate=0.01, beta=0.9, weight_decay=0):
        self.learning_rate = learning_rate
        self.momentum = beta
        self.weight_decay = weight_decay
        self.velocity_w = None
        self.velocity_b = None

    def update(self, layer, weights_grad, biases_grad):
        if self.velocity_w is None:
            self.velocity_w = np.zeros_like(layer.weights)
            self.velocity_b = np.zeros_like(layer.biases)

        # Compute look-ahead weights (w_ahead) without modifying actual weights
        w_ahead = layer.weights + self.momentum * self.velocity_w # w_ahead = w + b * v
        b_ahead = layer.biases + self.momentum * self.velocity_b # b_ahead = b + b * v

        # Compute gradients at the look-ahead position
        self.velocity_w = self.momentum * self.velocity_w - self.learning_rate * (weights_grad + self.weight_decay * w_ahead) # v = b * v - lr * (dw + wd * w_ahead)
        self.velocity_b = self.momentum * self.velocity_b - self.learning_rate * biases_grad # v = b * v - lr * db

        # Apply the final update
        layer.weights += self.velocity_w
        layer.biases += self.velocity_b


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
        self.s_w = self.beta * self.s_w + (1 - self.beta) * weights_grad**2 # s = b * s + (1 - b) * (dw^2)
        self.s_b = self.beta * self.s_b + (1 - self.beta) * biases_grad**2 # s = b * s + (1 - b) * (db^2)
        layer.weights -= self.learning_rate * (weights_grad / (np.sqrt(self.s_w) + self.epsilon) + self.weight_decay * layer.weights) # w = w - lr * (dw / sqrt(s) + wd * w)
        layer.biases -= self.learning_rate * biases_grad / (np.sqrt(self.s_b) + self.epsilon) # b = b - lr * (db / sqrt(s))

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
            layer.t = 0  # Initialize timestep
        layer.t += 1
        layer.m_w = self.beta1 * layer.m_w + (1 - self.beta1) * weights_grad # m = b * m + (1 - b) * dw
        layer.v_w = self.beta2 * layer.v_w + (1 - self.beta2) * weights_grad**2 # v = b * v + (1 - b) * (dw^2)
        layer.m_b = self.beta1 * layer.m_b + (1 - self.beta1) * biases_grad
        layer.v_b = self.beta2 * layer.v_b + (1 - self.beta2) * biases_grad**2
        m_hat_w = layer.m_w / (1 - self.beta1**layer.t) # Bias correction
        v_hat_w = layer.v_w / (1 - self.beta2**layer.t)
        m_hat_b = layer.m_b / (1 - self.beta1**layer.t)
        v_hat_b = layer.v_b / (1 - self.beta2**layer.t)
        layer.weights -= self.learning_rate * (m_hat_w / (np.sqrt(v_hat_w) + self.epsilon) + self.weight_decay * layer.weights) # w = w - lr * (m_hat / sqrt(v_hat) + wd * w)
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
            layer.t = 0  # Initialize timestep

        layer.t += 1

        # Compute biased first and second moment estimates
        layer.m_w = self.beta1 * layer.m_w + (1 - self.beta1) * weights_grad # m = b * m + (1 - b) * dw
        layer.v_w = self.beta2 * layer.v_w + (1 - self.beta2) * weights_grad**2
        layer.m_b = self.beta1 * layer.m_b + (1 - self.beta1) * biases_grad
        layer.v_b = self.beta2 * layer.v_b + (1 - self.beta2) * biases_grad**2

        # Bias correction
        m_hat_w = layer.m_w / (1 - self.beta1**layer.t) 
        v_hat_w = layer.v_w / (1 - self.beta2**layer.t)
        m_hat_b = layer.m_b / (1 - self.beta1**layer.t)
        v_hat_b = layer.v_b / (1 - self.beta2**layer.t)

        # Nadam-specific Nesterov momentum correction
        m_nadam_w = self.beta1 * m_hat_w + (1 - self.beta1) * weights_grad / (1 - self.beta1**layer.t) # m_nadam = b * m_nadam + (1 - b) * dw / (1 - b^t)
        m_nadam_b = self.beta1 * m_hat_b + (1 - self.beta1) * biases_grad / (1 - self.beta1**layer.t)

        # Update parameters
        layer.weights -= self.learning_rate * (m_nadam_w / (np.sqrt(v_hat_w) + self.epsilon) + self.weight_decay * layer.weights) # w = w - lr * (m_nadam / sqrt(v_hat) + wd * w)
        layer.biases -= self.learning_rate * m_nadam_b / (np.sqrt(v_hat_b) + self.epsilon)