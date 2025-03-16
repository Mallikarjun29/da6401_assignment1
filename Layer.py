import numpy as np

class Layer:
    """
    A fully connected neural network layer implementation.
    
    This layer performs the following operations:
    - Forward pass: y = Wx + b
    - Backward pass: Computes gradients using backpropagation
    
    The backpropagation algorithm computes three gradients:
    1. dL/dx: Gradient of loss with respect to input (for previous layer)
    2. dL/dW: Gradient of loss with respect to weights
    3. dL/db: Gradient of loss with respect to biases
    
    Attributes:
        input_size (int): Number of input features
        output_size (int): Number of output features/neurons
        weights (np.ndarray): Weight matrix of shape (input_size, output_size)
        biases (np.ndarray): Bias vector of shape (1, output_size)
        weights_grad (np.ndarray): Accumulated weight gradients
        biases_grad (np.ndarray): Accumulated bias gradients
        input (np.ndarray): Stored input for backpropagation
        m_w, v_w (np.ndarray): First and second moment estimates for weights (used by Adam/Nadam)
        m_b, v_b (np.ndarray): First and second moment estimates for biases (used by Adam/Nadam)
        t (int): Time step counter for Adam/Nadam optimizers
    """
    
    def __init__(self, input_size, output_size, weight_init='random'):
        """
        Initialize the layer with given dimensions and weight initialization.
        
        Args:
            input_size (int): Number of input features
            output_size (int): Number of output features/neurons
            weight_init (str): Weight initialization method
                - 'random': Small random values scaled by 0.01
                - 'xavier': Xavier/Glorot initialization scaled by sqrt(2/(n_in + n_out))
        """
        self.input_size = input_size
        self.output_size = output_size
        
        # Initialize weights based on the chosen method
        if weight_init == 'random':
            # Small random initialization to prevent vanishing/exploding gradients
            self.weights = np.random.randn(input_size, output_size) * 0.01
        elif weight_init == 'xavier':
            # Xavier initialization helps maintain variance across layers
            # Formula: W ~ N(0, sqrt(2/(n_in + n_out)))
            self.weights = np.random.randn(input_size, output_size) * np.sqrt(2 / (input_size + output_size))
        
        # Initialize biases to zero
        self.biases = np.zeros((1, output_size))
        
        # Initialize optimizer-specific variables
        self.m_w = None  # First moment estimate for weights
        self.v_w = None  # Second moment estimate for weights
        self.m_b = None  # First moment estimate for biases
        self.v_b = None  # Second moment estimate for biases
        self.t = 0      # Time step for Adam/Nadam
        
        # Initialize gradient storage
        self.weights_grad = None
        self.biases_grad = None

    def forward(self, input_data):
        """
        Perform forward pass computation.
        
        Computes: y = Wx + b
        
        Args:
            input_data (np.ndarray): Input data of shape (batch_size, input_size)
        
        Returns:
            np.ndarray: Layer output of shape (batch_size, output_size)
        """
        # Reshape input to matrix form and store for backprop
        self.input = input_data.reshape(-1, self.input_size)
        
        # Compute linear transformation: y = Wx + b
        return np.dot(self.input, self.weights) + self.biases

    def backward(self, output_grad, optimizer):
        """
        Perform backward pass computation using chain rule.
        
        Given dL/dy (output_grad), computes:
        1. dL/dx = dL/dy · W^T     (for previous layer)
        2. dL/dW = x^T · dL/dy     (weight gradients)
        3. dL/db = sum(dL/dy)      (bias gradients)
        
        Args:
            output_grad (np.ndarray): Gradient of loss with respect to layer output (dL/dy)
            optimizer: Optimizer instance for parameter updates
        
        Returns:
            np.ndarray: Gradient of loss with respect to layer input (dL/dx)
        """
        # Compute gradient with respect to input (for previous layer)
        # dL/dx = dL/dy · W^T
        input_grad = np.dot(output_grad, self.weights.T)
        
        # Compute gradient with respect to weights
        # dL/dW = x^T · dL/dy
        self.weights_grad = np.dot(self.input.T, output_grad)
        
        # Compute gradient with respect to biases
        # dL/db = sum(dL/dy) across batch dimension
        self.biases_grad = np.sum(output_grad, axis=0, keepdims=True)
        
        # Update parameters using the optimizer
        optimizer.update(self, self.weights_grad, self.biases_grad)
        
        return input_grad