import numpy as np

class Activation:
    """
    Base class for activation functions in neural networks.
    
    All activation functions should implement:
    1. forward: Computes f(x) where f is the activation function
    2. backward: Computes f'(x) * output_grad for backpropagation
    """
    
    def forward(self, input_data):
        """Forward pass computation"""
        raise NotImplementedError

    def backward(self, output_grad):
        """Backward pass computation"""
        raise NotImplementedError

class ReLU(Activation):
    """
    Rectified Linear Unit activation function.
    
    Forward: f(x) = max(0, x)
    Backward: f'(x) = 1 if x > 0 else 0
    """
    
    def forward(self, input_data):
        """
        Compute ReLU activation: max(0, x)
        
        Args:
            input_data (np.ndarray): Input values
            
        Returns:
            np.ndarray: ReLU output
        """
        self.input = input_data
        return np.maximum(0, input_data)

    def backward(self, output_grad, optimizer=None):
        """
        Compute ReLU gradient.
        
        dL/dx = dL/dy * dy/dx
        dy/dx = 1 if x > 0 else 0
        
        Args:
            output_grad (np.ndarray): Gradient from next layer (dL/dy)
            
        Returns:
            np.ndarray: Gradient with respect to input
        """
        return output_grad * (self.input > 0)

class Sigmoid(Activation):
    """
    Sigmoid activation function.
    
    Forward: f(x) = 1 / (1 + e^(-x))
    Backward: f'(x) = f(x) * (1 - f(x))
    """
    
    def forward(self, input_data):
        """
        Compute sigmoid activation: 1 / (1 + e^(-x))
        
        Args:
            input_data (np.ndarray): Input values
            
        Returns:
            np.ndarray: Sigmoid output
        """
        # Clip input to prevent overflow
        input_data = np.clip(input_data, -500, 500)
        self.output = 1 / (1 + np.exp(-input_data))
        return self.output

    def backward(self, output_grad, optimizer=None):
        """
        Compute sigmoid gradient.
        
        dL/dx = dL/dy * dy/dx
        dy/dx = y * (1 - y) where y = sigmoid(x)
        
        Args:
            output_grad (np.ndarray): Gradient from next layer (dL/dy)
            
        Returns:
            np.ndarray: Gradient with respect to input
        """
        return output_grad * self.output * (1 - self.output)

class Tanh(Activation):
    """
    Hyperbolic tangent activation function.
    
    Forward: f(x) = tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    Backward: f'(x) = 1 - tanh^2(x)
    """
    
    def forward(self, input_data):
        """
        Compute tanh activation
        
        Args:
            input_data (np.ndarray): Input values
            
        Returns:
            np.ndarray: Tanh output
        """
        self.output = np.tanh(input_data)
        return self.output

    def backward(self, output_grad, optimizer=None):
        """
        Compute tanh gradient.
        
        dL/dx = dL/dy * dy/dx
        dy/dx = 1 - tanh^2(x)
        
        Args:
            output_grad (np.ndarray): Gradient from next layer (dL/dy)
            
        Returns:
            np.ndarray: Gradient with respect to input
        """
        return output_grad * (1 - self.output**2)

class Softmax(Activation):
    """
    Softmax activation function.
    
    Forward: f(x_i) = e^(x_i) / Σ(e^(x_j))
    Backward: Simplified to pass through gradient when used with cross-entropy loss
    """
    
    def forward(self, input_data):
        """
        Compute softmax activation with numerical stability.
        
        Subtracts max value before exponential to prevent overflow.
        
        Args:
            input_data (np.ndarray): Input values
            
        Returns:
            np.ndarray: Softmax probabilities
        """
        # Subtract max for numerical stability
        exp_values = np.exp(input_data - np.max(input_data, axis=1, keepdims=True))
        # Normalize to get probabilities
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.output

    def backward(self, output_grad, optimizer=None):
        """
        Compute softmax gradient.
        
        Note: This implementation assumes softmax is used with cross-entropy loss,
        which simplifies the gradient to the difference between predicted and true probabilities.
        
        Args:
            output_grad (np.ndarray): Gradient from next layer (dL/dy)
            
        Returns:
            np.ndarray: Gradient with respect to input
        """
        return output_grad