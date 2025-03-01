import numpy as np
from model import FeedforwardNN
from data.dataset import load_data

def main():
    # Load the dataset
    X_train, y_train, X_test, y_test = load_data()

    # Initialize the neural network
    model = FeedforwardNN(input_size=X_train.shape[1], hidden_size=64, output_size=10)

    # Train the model
    model.train(X_train, y_train, epochs=100, learning_rate=0.01)

    # Evaluate the model
    accuracy = model.evaluate(X_test, y_test)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')

if __name__ == "__main__":
    main()