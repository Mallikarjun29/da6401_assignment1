import numpy as np
from keras.datasets import fashion_mnist

class FFNN:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros((1, output_size))
        self.input = None
        self.output = None

    def forward(self, input_data):
        self.input = input_data
        self.output = np.dot(input_data, self.weights) + self.biases
        return self.output

class FFActivation:
    def forward(self, input_data):
        raise NotImplementedError


class FFReLU(FFActivation):
    def forward(self, input_data):
        self.input = input_data
        return np.maximum(0, input_data)


class FFSigmoid(FFActivation):
    def forward(self, input_data):
        self.output = 1 / (1 + np.exp(-input_data))
        return self.output
    
# Load the dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

Layer = FFNN(784, 10) #784 because 28*28=784 and 10 because there are 10 classes. And I'll be flatten the input image.
ReLU = FFReLU()
Sigmoid = FFSigmoid()
output = Layer.forward(x_train[0].flatten())
output = ReLU.forward(output)
output = Sigmoid.forward(output)
print(output)