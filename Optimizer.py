import numpy as np

class Optimizer:
    """
    Base class for optimization algorithms.
    
    All optimizers should implement the update method which computes
    parameter updates based on gradients.
    """
    
    def update(self, layer, weights_grad, biases_grad):
        """Updates layer parameters using computed gradients"""
        raise NotImplementedError

class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer with weight decay.
    
    Updates parameters using the gradient descent rule:
    w = w - lr * (dw + wd * w)
    where:
    - lr is learning rate
    - dw is parameter gradient
    - wd is weight decay factor
    """
    
    def __init__(self, learning_rate=0.01, weight_decay=0):
        """
        Args:
            learning_rate (float): Step size for parameter updates
            weight_decay (float): L2 regularization factor
        """
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def update(self, layer, weights_grad, biases_grad):
        """
        Update parameters using SGD.
        
        Args:
            layer: Neural network layer
            weights_grad: Gradient of loss with respect to weights
            biases_grad: Gradient of loss with respect to biases
        """
        # Update weights: w = w - lr * (dw + wd * w)
        layer.weights -= self.learning_rate * (weights_grad + self.weight_decay * layer.weights)
        # Update biases: b = b - lr * db
        layer.biases -= self.learning_rate * biases_grad

class Momentum(Optimizer):
    """
    Momentum optimizer with weight decay.
    
    Implements momentum-based updates:
    v = β * v - lr * (dw + wd * w)
    w = w + v
    
    where:
    - v is velocity
    - β is momentum coefficient
    - lr is learning rate
    - dw is parameter gradient
    - wd is weight decay factor
    """
    
    def __init__(self, learning_rate=0.01, beta=0.9, weight_decay=0):
        """
        Args:
            learning_rate (float): Step size for parameter updates
            beta (float): Momentum coefficient
            weight_decay (float): L2 regularization factor
        """
        self.learning_rate = learning_rate
        self.momentum = beta
        self.weight_decay = weight_decay
        self.velocity_w = None  # Weight velocity
        self.velocity_b = None  # Bias velocity

    def update(self, layer, weights_grad, biases_grad):
        """
        Update parameters using momentum.
        
        Args:
            layer: Neural network layer
            weights_grad: Gradient of loss with respect to weights
            biases_grad: Gradient of loss with respect to biases
        """
        # Initialize velocities if first update
        if self.velocity_w is None:
            self.velocity_w = np.zeros_like(layer.weights)
            self.velocity_b = np.zeros_like(layer.biases)

        # Update velocities
        # v = β * v - lr * (dw + wd * w)
        self.velocity_w = self.momentum * self.velocity_w - \
                         self.learning_rate * (weights_grad + self.weight_decay * layer.weights)
        # v = β * v - lr * db
        self.velocity_b = self.momentum * self.velocity_b - \
                         self.learning_rate * biases_grad

        # Apply updates: w = w + v
        layer.weights += self.velocity_w
        layer.biases += self.velocity_b

class Nesterov(Optimizer):
    """
    Nesterov Accelerated Gradient optimizer.
    
    Implements NAG updates:
    w_ahead = w + β * v
    v = β * v - lr * ∇L(w_ahead)
    w = w + v
    
    where:
    - v is velocity
    - β is momentum coefficient
    - lr is learning rate
    - ∇L(w_ahead) is gradient at look-ahead position
    """
    
    def __init__(self, learning_rate=0.01, beta=0.9, weight_decay=0):
        """
        Args:
            learning_rate (float): Step size for parameter updates
            beta (float): Momentum coefficient
            weight_decay (float): L2 regularization factor
        """
        self.learning_rate = learning_rate
        self.momentum = beta
        self.weight_decay = weight_decay
        self.velocity_w = None
        self.velocity_b = None

    def update(self, layer, weights_grad, biases_grad):
        """
        Update parameters using Nesterov momentum.
        
        Args:
            layer: Neural network layer
            weights_grad: Gradient of loss with respect to weights
            biases_grad: Gradient of loss with respect to biases
        """
        if self.velocity_w is None:
            self.velocity_w = np.zeros_like(layer.weights)
            self.velocity_b = np.zeros_like(layer.biases)

        # Compute look-ahead parameters
        # w_ahead = w + β * v
        w_ahead = layer.weights + self.momentum * self.velocity_w
        b_ahead = layer.biases + self.momentum * self.velocity_b

        # Update velocity using look-ahead gradients
        # v = β * v - lr * ∇L(w_ahead)
        self.velocity_w = self.momentum * self.velocity_w - \
                         self.learning_rate * (weights_grad + self.weight_decay * w_ahead)
        self.velocity_b = self.momentum * self.velocity_b - \
                         self.learning_rate * biases_grad

        # Apply updates
        layer.weights += self.velocity_w
        layer.biases += self.velocity_b

class RMSprop(Optimizer):
    """
    RMSprop optimizer with weight decay.
    
    Maintains moving average of squared gradients:
    s = β * s + (1-β) * (dw)²
    w = w - lr * dw / (√s + ε)
    
    where:
    - s is second moment estimate
    - β is decay rate
    - ε is small constant for numerical stability
    """
    
    def __init__(self, learning_rate=0.001, beta=0.9, epsilon=1e-8, weight_decay=0):
        """
        Args:
            learning_rate (float): Step size for parameter updates
            beta (float): Decay rate for second moment estimate
            epsilon (float): Small constant for numerical stability
            weight_decay (float): L2 regularization factor
        """
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.s_w = None  # Second moment for weights
        self.s_b = None  # Second moment for biases

    def update(self, layer, weights_grad, biases_grad):
        """
        Update parameters using RMSprop.
        
        Args:
            layer: Neural network layer
            weights_grad: Gradient of loss with respect to weights
            biases_grad: Gradient of loss with respect to biases
        """
        if self.s_w is None:
            self.s_w = np.zeros_like(layer.weights)
            self.s_b = np.zeros_like(layer.biases)

        # Update second moments
        # s = β * s + (1-β) * (dw)²
        self.s_w = self.beta * self.s_w + (1 - self.beta) * weights_grad**2
        self.s_b = self.beta * self.s_b + (1 - self.beta) * biases_grad**2

        # Update parameters
        # w = w - lr * (dw / √s + wd * w)
        layer.weights -= self.learning_rate * (weights_grad / (np.sqrt(self.s_w) + self.epsilon) + 
                                             self.weight_decay * layer.weights)
        # b = b - lr * db / √s
        layer.biases -= self.learning_rate * biases_grad / (np.sqrt(self.s_b) + self.epsilon)

class Adam(Optimizer):
    """
    Adam optimizer with weight decay.
    
    Maintains moving averages of first and second moments of gradients:
    m = β1 * m + (1-β1) * dw
    v = β2 * v + (1-β2) * (dw)²
    w = w - lr * m / (√v + ε)
    
    where:
    - m is first moment estimate
    - v is second moment estimate
    - β1 is decay rate for first moment estimate
    - β2 is decay rate for second moment estimate
    - ε is small constant for numerical stability
    """
    
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0):
        """
        Args:
            learning_rate (float): Step size for parameter updates
            beta1 (float): Decay rate for first moment estimate
            beta2 (float): Decay rate for second moment estimate
            epsilon (float): Small constant for numerical stability
            weight_decay (float): L2 regularization factor
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay

    def update(self, layer, weights_grad, biases_grad):
        """
        Update parameters using Adam.
        
        Args:
            layer: Neural network layer
            weights_grad: Gradient of loss with respect to weights
            biases_grad: Gradient of loss with respect to biases
        """
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
    """
    Nadam optimizer with weight decay.
    
    Combines Adam and Nesterov Accelerated Gradient:
    m = β1 * m + (1-β1) * dw
    v = β2 * v + (1-β2) * (dw)²
    m_nadam = β1 * m + (1-β1) * dw / (1-β1^t)
    w = w - lr * m_nadam / (√v + ε)
    
    where:
    - m is first moment estimate
    - v is second moment estimate
    - m_nadam is Nesterov-corrected first moment estimate
    - β1 is decay rate for first moment estimate
    - β2 is decay rate for second moment estimate
    - ε is small constant for numerical stability
    """
    
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0):
        """
        Args:
            learning_rate (float): Step size for parameter updates
            beta1 (float): Decay rate for first moment estimate
            beta2 (float): Decay rate for second moment estimate
            epsilon (float): Small constant for numerical stability
            weight_decay (float): L2 regularization factor
        """
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay

    def update(self, layer, weights_grad, biases_grad):
        """
        Update parameters using Nadam.
        
        Args:
            layer: Neural network layer
            weights_grad: Gradient of loss with respect to weights
            biases_grad: Gradient of loss with respect to biases
        """
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