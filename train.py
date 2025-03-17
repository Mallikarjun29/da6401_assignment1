import argparse
import wandb
import numpy as np
from keras.datasets import fashion_mnist, mnist
from Layer import Layer
from Activation import ReLU, Softmax, Sigmoid, Tanh
from Optimizer import SGD, Momentum, Nesterov, RMSprop, Adam, Nadam
from Loss import CrossEntropyLoss, MSELoss
from NeuralNetwork import NeuralNetwork
from tabulate import tabulate
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train a neural network with specified parameters')
    
    # Wandb arguments
    parser.add_argument('-wp', '--wandb_project', type=str, default='myprojectname',
                        help='Project name for Weights & Biases dashboard')
    parser.add_argument('-we', '--wandb_entity', type=str, default='myname',
                        help='Entity name for Weights & Biases dashboard')
    
    # Dataset and training parameters
    parser.add_argument('-d', '--dataset', type=str, default='fashion_mnist',
                        choices=['mnist', 'fashion_mnist'],
                        help='Dataset to use for training')
    parser.add_argument('-e', '--epochs', type=int, default=1,
                        help='Number of epochs to train')
    parser.add_argument('-b', '--batch_size', type=int, default=4,
                        help='Batch size for training')
    
    # Model architecture
    parser.add_argument('-nhl', '--num_layers', type=int, default=1,
                        help='Number of hidden layers')
    parser.add_argument('-sz', '--hidden_size', type=int, default=4,
                        help='Number of neurons in hidden layers')
    parser.add_argument('-a', '--activation', type=str, default='sigmoid',
                        choices=['identity', 'sigmoid', 'tanh', 'ReLU'],
                        help='Activation function')
    
    # Optimization parameters
    parser.add_argument('-l', '--loss', type=str, default='cross_entropy',
                        choices=['mean_squared_error', 'cross_entropy'],
                        help='Loss function')
    parser.add_argument('-o', '--optimizer', type=str, default='sgd',
                        choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'],
                        help='Optimizer')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.1,
                        help='Learning rate')
    parser.add_argument('-m', '--momentum', type=float, default=0.5,
                        help='Momentum coefficient')
    parser.add_argument('-beta', '--beta', type=float, default=0.5,
                        help='Beta parameter for RMSprop')
    parser.add_argument('-beta1', '--beta1', type=float, default=0.5,
                        help='Beta1 parameter for Adam/Nadam')
    parser.add_argument('-beta2', '--beta2', type=float, default=0.5,
                        help='Beta2 parameter for Adam/Nadam')
    parser.add_argument('-eps', '--epsilon', type=float, default=1e-6,
                        help='Epsilon parameter for optimizers')
    parser.add_argument('-w_d', '--weight_decay', type=float, default=0.0,
                        help='Weight decay parameter')
    parser.add_argument('-w_i', '--weight_init', type=str, default='random',
                        choices=['random', 'Xavier'],
                        help='Weight initialization method')
    
    return parser.parse_args()

def load_data(dataset_name):
    """Load and preprocess dataset."""
    if dataset_name == 'mnist':
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    else:
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    
    # Combine train and test data for random splitting
    X = np.concatenate([X_train, X_test])
    y = np.concatenate([y_train, y_test])
    
    # Reshape and normalize
    X = X.reshape(X.shape[0], -1) / 255.0
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42)
    
    # Split training data into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42)
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def get_activation(name):
    """Get activation function by name."""
    activations = {
        'identity': lambda: None,
        'sigmoid': Sigmoid,
        'tanh': Tanh,
        'ReLU': ReLU
    }
    return activations[name]()

def get_optimizer(name, args):
    """Get optimizer instance by name with appropriate parameters."""
    optimizers = {
        'sgd': lambda: SGD(args.learning_rate, args.weight_decay),
        'momentum': lambda: Momentum(args.learning_rate, args.momentum, args.weight_decay),
        'nag': lambda: Nesterov(args.learning_rate, args.momentum, args.weight_decay),
        'rmsprop': lambda: RMSprop(args.learning_rate, args.beta, args.epsilon, args.weight_decay),
        'adam': lambda: Adam(args.learning_rate, args.beta1, args.beta2, args.epsilon, args.weight_decay),
        'nadam': lambda: Nadam(args.learning_rate, args.beta1, args.beta2, args.epsilon, args.weight_decay)
    }
    return optimizers[name]()

def get_loss(name):
    """Get loss function by name."""
    losses = {
        'mean_squared_error': MSELoss,
        'cross_entropy': CrossEntropyLoss
    }
    return losses[name]()

def create_model(args, input_size=784, num_classes=10):
    """Create neural network model with specified architecture."""
    layers = []
    
    # Add input layer
    prev_size = input_size
    
    # Add hidden layers
    for _ in range(args.num_layers):
        layers.append(Layer(prev_size, args.hidden_size, args.weight_init))
        if args.activation != 'identity':
            layers.append(get_activation(args.activation))
        prev_size = args.hidden_size
    
    # Add output layer
    layers.append(Layer(prev_size, num_classes, args.weight_init))
    layers.append(Softmax())
    
    # Create optimizer
    optimizer = get_optimizer(args.optimizer, args)
    
    return NeuralNetwork(layers, optimizer.__class__, {
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay
    })

def train(model, loss_fn, data, args):
    """Train the model and log metrics to wandb."""
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = data
    
    for epoch in range(args.epochs):
        # Training loop
        train_loss = 0
        num_batches = 0
        
        for i in range(0, len(X_train), args.batch_size):
            batch_x = X_train[i:i + args.batch_size]
            batch_y = y_train[i:i + args.batch_size]
            
            # Forward pass
            pred = model.forward(batch_x)
            loss = loss_fn.forward(pred, np.eye(10)[batch_y])
            
            # Backward pass
            grad = loss_fn.backward()
            model.backward(grad)
            
            train_loss += loss
            num_batches += 1
        
        # Compute validation metrics
        val_pred = model.forward(X_val)
        val_loss = loss_fn.forward(val_pred, np.eye(10)[y_val])
        val_acc = np.mean(np.argmax(val_pred, axis=1) == y_val)
        
        # Log metrics
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss / num_batches,
            'val_loss': val_loss,
            'val_accuracy': val_acc
        })
        
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"Train Loss: {train_loss/num_batches:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Accuracy: {val_acc:.4f}")
    
    # Compute test metrics
    evaluate_model(model, X_test, y_test, args.dataset)


def evaluate_model(model, X_test, y_test, dataset_name):
    # Class names for Fashion MNIST
    if dataset_name == 'fashion_mnist':
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    else:
        class_names = [str(i) for i in range(10)]
    
    # Get predictions
    y_pred = model.forward(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Calculate accuracy
    test_accuracy = np.mean(y_pred_classes == y_test)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Get detailed metrics
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred_classes, average=None)
    
    # Create summary DataFrame
    metrics_df = pd.DataFrame({
        'Class': class_names,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Support': support
    })
    
    # Add average row
    avg_precision, avg_recall, avg_f1, _ = precision_recall_fscore_support(y_test, y_pred_classes, average='weighted')
    metrics_df.loc[len(metrics_df)] = ['Average', avg_precision, avg_recall, avg_f1, np.sum(support)]
    
    # Print table
    print("\nModel Performance Summary:")
    print(tabulate(metrics_df, headers='keys', tablefmt='grid', floatfmt='.4f'))
    
    # Create a styled table for plotting
    plt.figure(figsize=(12, 6))
    plt.axis('off')
    table = plt.table(cellText=metrics_df.values,
                     colLabels=metrics_df.columns,
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.2, 0.2, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)
    plt.title('Model Performance Summary', pad=20)
    
    # Log table to wandb
    wandb.log({"performance_summary": wandb.Table(dataframe=metrics_df)})
    
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

def main():
    """Main function to run the training pipeline."""
    args = parse_args()
    
    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=vars(args)
    )
    
    # Load data
    data = load_data(args.dataset)
    
    # Create model
    model = create_model(args)
    
    # Get loss function
    loss_fn = get_loss(args.loss)

    wandb.run.name = f"{args.dataset}dataset_{args.num_layers}layers_{args.hidden_size}units_{args.activation}a_{args.optimizer}o_{args.epochs}e_{args.batch_size}b_{args.learning_rate}lr_{args.weight_decay}wd_{args.weight_init}wi_{args.loss}loss"
    # Train model
    train(model, loss_fn, data, args)

    # Close wandb run
    wandb.finish()

if __name__ == "__main__":
    main()