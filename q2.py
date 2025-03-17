import numpy as np
from keras.datasets import fashion_mnist
from Layer import Layer
from Activation import ReLU, Sigmoid, Softmax
from NeuralNetwork import NeuralNetwork
from Loss import CrossEntropyLoss



def main():
    # Load the dataset
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    # Preprocess the data
    x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
    x_test = x_test.reshape(x_test.shape[0], -1) / 255.0

    # Initialize the network
    layer = Layer(784, 10)  # 784 because 28*28=784 and 10 because there are 10 classes
    relu = ReLU()
    softmax = Softmax()

    # Forward pass
    output = layer.forward(x_train[0:32]) # Forward pass for the first 32 samples
    output = relu.forward(output)
    output = softmax.forward(output)

    print("feedforward output", output)

if __name__ == "__main__":
    main()