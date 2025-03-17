# Neural Network Implementation for Fashion MNIST Classification

This repository contains a custom implementation of a feedforward neural network with various optimization algorithms and loss functions.

## Repo(rt) link
https://github.com/Mallikarjun29/da6401_assignment1.git

[Wandb report](https://wandb.ai/da24s009-indiam-institute-of-technology-madras/sweep_experiment_final/reports/da24s009-s-Assignment-1--VmlldzoxMTgzMzc0Nw?accessToken=ebnkhxglng1r6va8v12rm2t43nxjc2pm0dmd2jbhe7vte0iwglk9ucu9sghe4xrb)

## Setup

### Requirements

Install the required packages:

```bash
pip install numpy wandb keras tensorflow pandas tabulate matplotlib seaborn
```

### Project Structure

```
da6401_assignment1/
├── Layer.py           # Neural network layer implementation
├── Activation.py      # Activation functions (ReLU, Sigmoid, Tanh, Softmax)
├── Optimizer.py       # Optimization algorithms (SGD, Momentum, NAG, RMSprop, Adam, Nadam)
├── Loss.py           # Loss functions (Cross Entropy, MSE)
├── NeuralNetwork.py  # Neural network model
├── train.py          # Training script
├── q1.py
├── q2.py
├── q4.py
├── q7.py
├── q18py
└── README.md

```

## Training the Model

### Basic Usage

To train the model with default parameters:

```bash
python train.py --wandb_entity your_username --wandb_project your_project_name
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `-wp`, `--wandb_project` | myprojectname | Weights & Biases project name |
| `-we`, `--wandb_entity` | myname | Weights & Biases entity name |
| `-d`, `--dataset` | fashion_mnist | Dataset choice (mnist, fashion_mnist) |
| `-e`, `--epochs` | 1 | Number of training epochs |
| `-b`, `--batch_size` | 4 | Training batch size |
| `-l`, `--loss` | cross_entropy | Loss function (mean_squared_error, cross_entropy) |
| `-o`, `--optimizer` | sgd | Optimizer (sgd, momentum, nag, rmsprop, adam, nadam) |
| `-lr`, `--learning_rate` | 0.1 | Learning rate |
| `-m`, `--momentum` | 0.5 | Momentum coefficient |
| `-beta`, `--beta` | 0.5 | RMSprop beta parameter |
| `-beta1`, `--beta1` | 0.5 | Adam/Nadam beta1 parameter |
| `-beta2`, `--beta2` | 0.5 | Adam/Nadam beta2 parameter |
| `-eps`, `--epsilon` | 1e-6 | Optimizer epsilon parameter |
| `-w_d`, `--weight_decay` | 0.0 | Weight decay (L2 regularization) |
| `-w_i`, `--weight_init` | random | Weight initialization (random, Xavier) |
| `-nhl`, `--num_layers` | 1 | Number of hidden layers |
| `-sz`, `--hidden_size` | 4 | Number of neurons per hidden layer |
| `-a`, `--activation` | sigmoid | Activation function (identity, sigmoid, tanh, ReLU) |

### Example Commands

Train with Adam optimizer and ReLU activation:
```bash
python train.py \
    --wandb_entity your_username \
    --wandb_project your_project_name \
    --optimizer adam \
    --learning_rate 0.001 \
    --activation ReLU \
    --num_layers 2 \
    --hidden_size 128 \
    --batch_size 32 \
    --epochs 10
```

Train with Nadam optimizer and Xavier initialization:
```bash
python train.py \
    --wandb_entity your_username \
    --wandb_project your_project_name \
    --optimizer nadam \
    --weight_init Xavier \
    --num_layers 3 \
    --hidden_size 256 \
    --batch_size 64 \
    --epochs 20
```

## Monitoring Training

Training progress is logged to Weights & Biases. You can monitor:
- Training loss
- Validation loss
- Validation accuracy
- Test accuracy
- Confusion matrix
- Performance metrics per class

Visit your W&B dashboard at: `https://wandb.ai/your_username/your_project_name`

## Evaluation

The model automatically evaluates on the test set after training and reports:
- Test accuracy
- Confusion matrix
- Per-class precision, recall, and F1-score
- Support (number of samples) for each class

## Model Architecture

The network architecture is customizable through command line arguments:
- Input layer: 784 neurons (28x28 pixels)
- Hidden layers: Configurable number and size
- Output layer: 10 neurons (number of classes)
- Configurable activation functions
- Choice of weight initialization methods
