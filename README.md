# MNIST-Digit-Classifier
This repository contains the solution of  two main tasks: solving the XOR problem and handwritten digit recognition using the MNIST dataset.

## XOR Problem

The XOR problem is a classic example in machine learning, where a neural network is trained to learn the XOR logic gate. In this assignment, NumPy is used to generate the XOR input and output data, and the mean squared error is used as the loss function. The XOR problem is solved for four cases, with a learning rate of 0.1, and the results are saved as `XOR_Solved_w`.

## Handwritten Digit Recognition

The MNIST dataset is a widely used dataset for handwritten digit recognition. It consists of a large collection of 28x28 grayscale images of handwritten digits from 0 to 9, along with their corresponding labels. The goal is to build a model that can accurately classify unseen handwritten digits.

### Model 1

The first model architecture consists of the following layers:

- Flatten layer
- Dense layer with 128 units and ReLU activation
- Dense layer with 10 units and Softmax activation

### Model 2

The second model architecture consists of the following layers:

- Flatten layer
- Dense layer with 64 units and ReLU activation
- Dense layer with 10 units and Softmax activation

### Model 3

The third model architecture consists of the following layers:

- Flatten layer
- Dense layer with 32 units and ReLU activation
- Dense layer with 10 units and Softmax activation

Each model is trained and evaluated, and the training and validation losses are plotted.

## Hyperparameter Tuning

Four different configurations of hyperparameters are explored, and the training and validation losses are plotted for each configuration. If the training does not improve after five steps, it is terminated and plotted.

## Requirements

- Python
- NumPy
- Keras
- Matplotlib (for plotting)

## Usage

1. Clone the repository.
2. Install the required dependencies.
3. Run the provided Python scripts to train the models and generate the plots.

## License

This project is licensed under the [MIT License](LICENSE).
