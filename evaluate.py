import wandb
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from keras.datasets import fashion_mnist
from Layer import Layer
from Activation import ReLU, Softmax, Sigmoid, Tanh
from Optimizer import SGD, Momentum, Nesterov, RMSprop, Adam, Nadam
from Loss import CrossEntropyLoss
from NeuralNetwork import NeuralNetwork
from sklearn.metrics import classification_report

def one_hot_encode(y, num_classes):
    return np.eye(num_classes)[y]

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
    
    # Show misclassified examples in a separate figure
    incorrect_mask = y_pred_classes != y_test
    incorrect_indices = np.where(incorrect_mask)[0]
    
    if len(incorrect_indices) > 0:
        fig2 = plt.figure(figsize=(15, 5))
        plt.suptitle('Misclassified Examples', fontsize=14)
        
        for i, idx in enumerate(incorrect_indices[:5]):
            plt.subplot(1, 5, i + 1)
            img = X_test[idx].reshape(28, 28)
            plt.imshow(img, cmap='gray')
            plt.title(f'True: {class_names[y_test[idx]]}\nPred: {class_names[y_pred_classes[idx]]}')
            plt.axis('off')
        plt.tight_layout()
    
    # Log to wandb
    wandb.log({
        "test_accuracy": test_accuracy,
        "confusion_matrix": wandb.Image(fig),
        "misclassified_examples": wandb.Image(fig2)
    })
    
    plt.show()

# Load and preprocess test data
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_test = X_test.reshape(X_test.shape[0], -1) / 255.0
X_train = X_train.reshape(X_train.shape[0], -1) / 255.0

# Define the best configuration dictionary
best_config = {
    'num_hidden_layers': 4,
    'batch_size': 64,
    'hidden_layer_size': 128,
    'learning_rate': 0.001,
    'weight_decay': 0.0005,
    'optimizer': Nadam,
    'activation_function': ReLU(),  # Initialize the activation function
    'weight_initialization': 'xavier',
    'num_epochs': 5
}

# Initialize wandb
wandb.login()
wandb.init(project="da6401_assignment_1", name="best_model_evaluation")

# Define optimizer parameters
optimizer_params = {'learning_rate': best_config['learning_rate'], 'weight_decay': best_config['weight_decay']}

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
best_model = NeuralNetwork(layers, best_config['optimizer'], optimizer_params)

# Define the loss
loss = CrossEntropyLoss()

# Training loop
num_classes = 10
for epoch in range(best_config['num_epochs']):
    for i in range(0, X_train.shape[0], best_config['batch_size']):
        # Get the next batch of data
        X_batch = X_train[i:i+best_config['batch_size']]
        y_batch = y_train[i:i+best_config['batch_size']]

        # One-hot encode the labels
        y_batch_one_hot = one_hot_encode(y_batch, num_classes)

        # Forward pass
        y_pred = best_model.forward(X_batch)
        loss_val = loss.forward(y_pred, y_batch_one_hot)

        # Backward pass
        loss_grad = loss.backward()
        best_model.backward(loss_grad)

    # Compute the training accuracy
    y_pred_train = best_model.forward(X_train)
    y_train_one_hot = one_hot_encode(y_train, num_classes)

# Create and evaluate the model
evaluate_best_model(best_model, X_test, y_test)
wandb.finish()