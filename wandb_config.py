import wandb
import numpy as np
from keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
from q3 import Layer, NeuralNetwork, Activation, ReLU, Tanh, Sigmoid, Softmax, CrossEntropyLoss, SGD, Momentum, RMSprop, Adam, Nadam, Nesterov
def one_hot_encode(y, num_classes):
    return np.eye(num_classes)[y]

# Initialize wandb
wandb.login()

# Load the Fashion MNIST dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Preprocess the data
X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
X_test = X_test.reshape(X_test.shape[0], -1) / 255.0

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Define the sweep configuration
sweep_config = {
    'method': 'random',
    'metric': {
        'name': 'val_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'num_epochs': {
            'values': [5, 10]
        },
        'num_hidden_layers': {
            'values': [3, 4, 5]
        },
        'hidden_layer_size': {
            'values': [32, 64, 128]
        },
        'weight_decay': {
            'values': [0, 0.0005, 0.5]
        },
        'learning_rate': {
            'values': [1e-3, 1e-4]
        },
        'optimizer': {
            'values': ['sgd', 'momentum', 'nesterov', 'rmsprop', 'adam', 'nadam']
        },
        'batch_size': {
            'values': [16, 32, 64]
        },
        'weight_initialization': {
            'values': ['random', 'xavier']
        },
        'activation_function': {
            'values': ['sigmoid', 'tanh', 'relu']
        }
    }
}

sweep_id = wandb.sweep(sweep_config, project="da6401_assignment1")

# Define the training function
def train():
    # Initialize wandb run
    wandb.init()

    # Get hyperparameters from wandb config
    config = wandb.config

    # Define the optimizer
    if config.optimizer == 'sgd':
        optimizer = SGD(learning_rate=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer == 'momentum':
        optimizer = Momentum(learning_rate=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer == 'nesterov':
        optimizer = Nesterov(learning_rate=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer == 'rmsprop':
        optimizer = RMSprop(learning_rate=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer == 'adam':
        optimizer = Adam(learning_rate=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer == 'nadam':
        optimizer = Nadam(learning_rate=config.learning_rate, weight_decay=config.weight_decay)

    # Define the activation function
    if config.activation_function == 'sigmoid':
        activation = Sigmoid()
    elif config.activation_function == 'tanh':
        activation = Tanh()
    elif config.activation_function == 'relu':
        activation = ReLU()

    # Define the weight initialization
    if config.weight_initialization == 'random':
        weight_init = 'random'
    elif config.weight_initialization == 'xavier':
        weight_init = 'xavier'

    # Define the model
    layers = []
    input_size = 784
    for _ in range(config.num_hidden_layers):
        layers.append(Layer(input_size, config.hidden_layer_size, weight_init=weight_init))
        layers.append(activation)
        input_size = config.hidden_layer_size
    layers.append(Layer(input_size, 10, weight_init=weight_init))
    layers.append(Softmax())

    model = NeuralNetwork(layers, optimizer)

    # Define the loss
    loss = CrossEntropyLoss()

    # Training loop
    num_classes = 10
    for epoch in range(config.num_epochs):
        for i in range(0, X_train.shape[0], config.batch_size):
            # Get the next batch of data
            X_batch = X_train[i:i+config.batch_size]
            y_batch = y_train[i:i+config.batch_size]

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

        # Log the loss
        wandb.log({"train_loss": loss_val, "val_loss": val_loss})

# Run the sweep
wandb.agent(sweep_id, train)