# Feedforward Neural Network from Scratch

This project implements a feedforward neural network using only NumPy. The goal is to provide a clear understanding of how neural networks work by building one from the ground up.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Overview

The feedforward neural network consists of multiple layers of neurons, where each layer is fully connected to the next. The network is capable of learning from data through a process of forward and backward propagation. This project includes the following components:

- **Model**: The core neural network implementation.
- **Layers**: Definitions for various types of layers, including dense and activation layers.
- **Utilities**: Functions for data normalization, loss calculation, and accuracy measurement.
- **Data Handling**: Functions for loading and preprocessing datasets.

## Installation

To install the required dependencies, you can use the following command:

```
pip install -r requirements.txt
```

This will install NumPy, which is the only dependency for this project.

## Usage

To run the feedforward neural network, execute the `main.py` file:

```
python src/main.py
```

This will initiate the training process and evaluate the model on the test dataset.

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.