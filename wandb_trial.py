import wandb
import numpy as np
from keras.datasets import fashion_mnist
from NeuralNetwork import NeuralNetwork
from Layer import Layer
from Activation import ReLU, Softmax
from Optimizer import SGD
from Loss import CrossEntropyLoss

def one_hot_encode(y, num_classes):
    return np.eye(num_classes)[y]

# Initialize wandb
# wandb.login()

# Load the Fashion MNIST dataset
(X_train, y_train), (X_val, y_val) = fashion_mnist.load_data()

# Preprocess the data
X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
X_val = X_val.reshape(X_val.shape[0], -1) / 255.0

# Define the loss
loss = CrossEntropyLoss()

# Define the optimizer
optimizer = SGD
optimizer_params = {
    "learning_rate": 0.01
}
# Define the model
model = NeuralNetwork([
    Layer(784, 128),
    ReLU(),
    Layer(128, 64),
    ReLU(),
    Layer(64, 10),
    Softmax()
], optimizer, optimizer_params)

# Initialize the wandb run
# wandb.init(project="da6401_assignment1", name="wandb_trial2")

def train(model, loss, optimizer, X_train, y_train, X_val, y_val, num_epochs=10, batch_size=32):
    num_classes = 10
    for epoch in range(num_epochs):
        for i in range(0, X_train.shape[0], batch_size):
            # Get the next batch of data
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            # One-hot encode the labels
            y_batch_one_hot = one_hot_encode(y_batch, num_classes)

            # Forward pass
            y_pred = model.forward(X_batch)
            loss_val = loss.forward(y_pred, y_batch_one_hot)

            # Backward pass
            loss_grad = loss.backward()
            model.backward(loss_grad)

        # Compute the validation loss
        y_pred_val = model.forward(X_val)
        y_val_one_hot = one_hot_encode(y_val, num_classes)
        val_loss = loss.forward(y_pred_val, y_val_one_hot)
        print(f"Epoch {epoch+1}/{num_epochs}, loss={loss_val}, val_loss={val_loss}")
        # Log the loss
        # wandb.log({"train_loss": loss_val, "val_loss": val_loss})

# Train the model
train(model, loss, optimizer, X_train, y_train, X_val, y_val, num_epochs=10, batch_size=16)

# Log the model
# wandb.sklearn.plot_model(model, "model.pkl")
