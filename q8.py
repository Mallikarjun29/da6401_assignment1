import wandb
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from keras.datasets import fashion_mnist
from Layer import Layer
from Activation import ReLU, Softmax, Sigmoid, Tanh
from Optimizer import SGD, Momentum, Nesterov, RMSprop, Adam, Nadam
from Loss import CrossEntropyLoss, MSELoss
from NeuralNetwork import NeuralNetwork
from sklearn.metrics import classification_report

def one_hot_encode(y, num_classes):
    return np.eye(num_classes)[y]

def train_and_evaluate(model, loss_fn, X_train, y_train, X_val, y_val, X_test, y_test, config):
    num_classes = 10
    train_losses = []
    val_losses = []
    pre_final_layer_gradients = []
    
    for epoch in range(config['num_epochs']):
        epoch_loss = 0
        num_batches = 0
        epoch_gradients = []
        
        for i in range(0, X_train.shape[0], config['batch_size']):
            X_batch = X_train[i:i+config['batch_size']]
            y_batch = y_train[i:i+config['batch_size']]
            y_batch_one_hot = one_hot_encode(y_batch, num_classes)

            # Forward pass
            y_pred = model.forward(X_batch)
            loss_val = loss_fn.forward(y_pred, y_batch_one_hot)
            
            # Backward pass
            loss_grad = loss_fn.backward()
            model.backward(loss_grad)
            
            # Capture gradients of the pre-final layer
            pre_final_layer_grad = model.layers[-2].weights_grad
            epoch_gradients.append(pre_final_layer_grad)
            
            epoch_loss += loss_val
            num_batches += 1
        
        # Compute training and validation losses
        train_loss = epoch_loss / num_batches
        train_losses.append(train_loss)
        
        val_pred = model.forward(X_val)
        val_true = one_hot_encode(y_val, num_classes)
        val_loss = loss_fn.forward(val_pred, val_true)
        val_losses.append(val_loss)
        
        # Store the average gradient of the pre-final layer for the epoch
        avg_grad = np.mean(np.array(epoch_gradients), axis=0)
        pre_final_layer_gradients.append(avg_grad)
        
        print(f"Epoch {epoch + 1}/{config['num_epochs']}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # Evaluate the model on the test set
    test_accuracy, cm = evaluate_best_model(model, X_test, y_test)
    
    return train_losses, val_losses, pre_final_layer_gradients, test_accuracy

def evaluate_best_model(model, X_test, y_test):
    # Class names for Fashion MNIST
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    # Get predictions
    y_pred = model.forward(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Calculate accuracy
    test_accuracy = np.mean(y_pred_classes == y_test)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred_classes)
    
    # Create figure for confusion matrix
    fig = plt.figure(figsize=(10, 8))
    
    # Plot Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    
    plt.tight_layout()
    
    # Log confusion matrix to wandb
    wandb.log({"confusion_matrix": wandb.Image(fig)})
    
    plt.close(fig)
    
    return test_accuracy, cm

# Load and preprocess data
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_test = X_test.reshape(X_test.shape[0], -1) / 255.0
X_train = X_train.reshape(X_train.shape[0], -1) / 255.0

# Split training data into train and validation
val_size = 5000
X_val = X_train[-val_size:]
y_val = y_train[-val_size:]
X_train = X_train[:-val_size]
y_train = y_train[:-val_size]

# Define the best configuration dictionary
best_config = {
    'num_hidden_layers': 2,
    'batch_size': 32,
    'hidden_layer_size': 128,
    'learning_rate': 0.001,
    'weight_decay': 0,
    'activation_function': ReLU(),
    'weight_initialization': 'random',
    'num_epochs': 10
}

# Define optimizer parameters
optimizer_params = {'learning_rate': best_config['learning_rate'], 'weight_decay': best_config['weight_decay']}

# Function to create and train the model with a given optimizer and loss function
def create_and_train_model(optimizer, loss_fn, optimizer_name, loss_name):
    print(f"\nTraining with {optimizer_name} and {loss_name} Loss")
    
    # Define the model
    layers = []
    input_size = 784
    for _ in range(best_config['num_hidden_layers']):
        layers.append(Layer(input_size, best_config['hidden_layer_size'], 
                           weight_init=best_config['weight_initialization']))
        layers.append(best_config['activation_function'])
        input_size = best_config['hidden_layer_size']
    layers.append(Layer(input_size, 10, weight_init=best_config['weight_initialization']))
    layers.append(Softmax())

    # Create the model with the correct optimizer
    model = NeuralNetwork(layers, optimizer, optimizer_params)
    
    # Train and evaluate the model
    train_losses, val_losses, pre_final_layer_gradients, test_accuracy = train_and_evaluate(model, loss_fn, X_train, y_train, X_val, y_val, X_test, y_test, best_config)
    
    # Log test accuracy and pre-final layer gradients to wandb
    wandb.log({
        "test_accuracy": test_accuracy,
        "pre_final_layer_gradients": [wandb.Histogram(np_histogram=np.histogram(g)) for g in pre_final_layer_gradients]
    })
    
    return train_losses, val_losses, pre_final_layer_gradients, test_accuracy

# Initialize wandb
wandb.login()
wandb.init(project="sweep_experiment_final", name="model_comparison_with_MSE")

# Train and evaluate with Cross Entropy Loss and Nadam
train_losses_ce_nadam, val_losses_ce_nadam, gradients_ce_nadam, test_acc_ce_nadam = create_and_train_model(Nadam, CrossEntropyLoss(), "Nadam", "Cross Entropy")

# Train and evaluate with Squared Error Loss and Nadam
train_losses_se_nadam, val_losses_se_nadam, gradients_se_nadam, test_acc_se_nadam = create_and_train_model(Nadam, MSELoss(), "Nadam", "Squared Error")

# Train and evaluate with Cross Entropy Loss and SGD
train_losses_ce_sgd, val_losses_ce_sgd, gradients_ce_sgd, test_acc_ce_sgd = create_and_train_model(SGD, CrossEntropyLoss(), "SGD", "Cross Entropy")

# Train and evaluate with Squared Error Loss and SGD
train_losses_se_sgd, val_losses_se_sgd, gradients_se_sgd, test_acc_se_sgd = create_and_train_model(SGD, MSELoss(), "SGD", "Squared Error")

# Plot the training and validation losses for all combinations
plt.figure(figsize=(12, 6))
plt.plot(train_losses_ce_nadam, label='Cross Entropy - Train Loss (Nadam)')
plt.plot(val_losses_ce_nadam, label='Cross Entropy - Val Loss (Nadam)')
plt.plot(train_losses_se_nadam, label='Squared Error - Train Loss (Nadam)')
plt.plot(val_losses_se_nadam, label='Squared Error - Val Loss (Nadam)')
plt.plot(train_losses_ce_sgd, label='Cross Entropy - Train Loss (SGD)')
plt.plot(val_losses_ce_sgd, label='Cross Entropy - Val Loss (SGD)')
plt.plot(train_losses_se_sgd, label='Squared Error - Train Loss (SGD)')
plt.plot(val_losses_se_sgd, label='Squared Error - Val Loss (SGD)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Losses for Different Loss Functions and Optimizers')
plt.legend()
plt.show()

# Plot the gradients of the pre-final layer for all combinations
plt.figure(figsize=(12, 6))
plt.plot([np.mean(np.abs(g)) for g in gradients_ce_nadam], label='Cross Entropy - Gradients (Nadam)', color='red', linestyle='--')
plt.plot([np.mean(np.abs(g)) for g in gradients_se_nadam], label='Squared Error - Gradients (Nadam)', color='black', linestyle='--')
plt.plot([np.mean(np.abs(g)) for g in gradients_ce_sgd], label='Cross Entropy - Gradients (SGD)', color='red', linestyle=':')
plt.plot([np.mean(np.abs(g)) for g in gradients_se_sgd], label='Squared Error - Gradients (SGD)', color='black', linestyle=':')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Gradient')
plt.title('Gradients of the Pre-Final Layer for Different Loss Functions and Optimizers')
plt.legend()
plt.show()

# Log the gradients plot to wandb
fig_gradients = plt.figure(figsize=(12, 6))
plt.plot([np.mean(np.abs(g)) for g in gradients_ce_nadam], label='Cross Entropy - Gradients (Nadam)', color='red', linestyle='--')
plt.plot([np.mean(np.abs(g)) for g in gradients_se_nadam], label='Squared Error - Gradients (Nadam)', color='black', linestyle='--')
plt.plot([np.mean(np.abs(g)) for g in gradients_ce_sgd], label='Cross Entropy - Gradients (SGD)', color='red', linestyle=':')
plt.plot([np.mean(np.abs(g)) for g in gradients_se_sgd], label='Squared Error - Gradients (SGD)', color='black', linestyle=':')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Gradient')
plt.title('Gradients of the Pre-Final Layer for Different Loss Functions and Optimizers')
plt.legend()
wandb.log({"pre_final_layer_gradients_plot": wandb.Image(fig_gradients)})
plt.close(fig_gradients)

# Plot the test accuracy for all combinations
test_accuracies = {
    'Cross Entropy (Nadam)': test_acc_ce_nadam,
    'Squared Error (Nadam)': test_acc_se_nadam,
    'Cross Entropy (SGD)': test_acc_ce_sgd,
    'Squared Error (SGD)': test_acc_se_sgd
}

plt.figure(figsize=(12, 6))
plt.bar(test_accuracies.keys(), test_accuracies.values(), color=['red', 'black', 'red', 'black'])
plt.xlabel('Model')
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy for Different Loss Functions and Optimizers')
plt.ylim(0, 1)
plt.show()

# Log the test accuracy plot to wandb
fig_accuracy = plt.figure(figsize=(12, 6))
plt.bar(test_accuracies.keys(), test_accuracies.values(), color=['red', 'black', 'red', 'black'])
plt.xlabel('Model')
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy for Different Loss Functions and Optimizers')
plt.ylim(0, 1)
wandb.log({"test_accuracy_plot": wandb.Image(fig_accuracy)})
plt.close(fig_accuracy)

# Finish wandb run
wandb.finish()