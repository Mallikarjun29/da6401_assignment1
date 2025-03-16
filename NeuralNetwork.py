import numpy as np

class NeuralNetwork:
    """
    A feed-forward neural network implementation supporting various layer types and optimizers.
    
    The network implements forward propagation and backpropagation for training.
    Backpropagation uses the chain rule to compute gradients:
    dL/dw = dL/dy * dy/dw where L is loss, y is output, w is weights
    
    Attributes:
        layers (list): List of layer objects (dense layers and activation functions)
        optimizers (list): List of optimizer instances for each layer
    """
    
    def __init__(self, layers, optimizer_class, optimizer_params):
        """
        Initialize the neural network with layers and optimizers.
        
        Args:
            layers (list): List of layer objects defining network architecture
            optimizer_class (class): Class of the optimizer to use (e.g., SGD, Adam)
            optimizer_params (dict): Parameters for the optimizer (learning_rate, etc.)
        """
        self.layers = layers
        # Create an optimizer instance for each layer with same parameters
        self.optimizers = [optimizer_class(**optimizer_params) for _ in layers]

    def forward(self, x):
        """
        Perform forward propagation through the network.
        
        For each layer l:
            z[l] = W[l]·a[l-1] + b[l]  (for dense layers)
            a[l] = g(z[l])             (where g is activation function)
        
        Args:
            x (np.ndarray): Input data of shape (batch_size, input_features)
        
        Returns:
            np.ndarray: Network output after forward pass
        """
        for layer in self.layers:  # Forward pass through each layer
            x = layer.forward(x)
        return x

    def backward(self, output_grad):
        """
        Perform backpropagation through the network.
        
        Implements the chain rule for each layer l from last to first:
        dL/dz[l] = dL/da[l] * da[l]/dz[l]    (activation gradient)
        dL/dW[l] = dL/dz[l] * da[l-1]        (weight gradient)
        dL/db[l] = dL/dz[l]                   (bias gradient)
        dL/da[l-1] = dL/dz[l] * W[l]         (propagate to previous layer)
        
        Args:
            output_grad (np.ndarray): Gradient of loss with respect to network output
                                    Shape: (batch_size, output_features)
        """
        # Backward pass through each layer in reverse order
        # Using zip to pair each layer with its optimizer
        for layer, optimizer in zip(reversed(self.layers), reversed(self.optimizers)):
            # Compute gradients and update weights for the current layer
            # output_grad contains dL/da[l] for the current layer
            # layer.backward returns dL/da[l-1] for the next layer
            output_grad = layer.backward(output_grad, optimizer)