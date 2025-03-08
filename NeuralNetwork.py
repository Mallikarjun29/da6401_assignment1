import numpy as np

class NeuralNetwork:
    def __init__(self, layers, optimizer_class, optimizer_params):
        self.layers = layers
        self.optimizers = [optimizer_class(**optimizer_params) for _ in layers] # Created an optimizer for each layer

    def forward(self, x):
        for layer in self.layers: # Forward pass through each layer
            x = layer.forward(x)
        return x

    def backward(self, output_grad):
        for layer, optimizer in zip(reversed(self.layers), reversed(self.optimizers)): # Backward pass through each layer
            output_grad = layer.backward(output_grad, optimizer)