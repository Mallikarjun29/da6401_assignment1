class FeedforwardNN:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, output_gradient):
        for layer in reversed(self.layers):
            output_gradient = layer.backward(output_gradient)
        return output_gradient

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            output = self.forward(X)
            loss = self.calculate_loss(output, y)
            output_gradient = self.calculate_loss_gradient(output, y)
            self.backward(output_gradient)

            for layer in self.layers:
                layer.update_parameters(learning_rate)

            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')

    def calculate_loss(self, output, y):
        # Implement loss calculation (e.g., mean squared error)
        pass

    def calculate_loss_gradient(self, output, y):
        # Implement loss gradient calculation
        pass