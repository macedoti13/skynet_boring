# SkyNet: A Numpy-powered Machine Learning Library 🚀

Welcome to `SkyNet`! This is my little pet project where I'm building a machine learning library using just numpy. It's a fun adventure to understand the nuts and bolts of ML, and I'm documenting it all here. So jump in!

![Banner/Image](images/skynet.png)

## Table of Contents
1. [Features](#features)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Acknowledgements](#acknowledgements)

## Features
- **Numpy Powered**: Built completely with numpy, ensuring an educational perspective to the ML processes.
- **Custom Neural Nets**: Create custom dense layers, adjust activations, and optimize away!
- **Transparency**: Detailed comments and docstrings to guide you through each part of the code.

## Installation
Clone this repository:
```bash
git clone https://github.com/macedoti13/skynet.git
```

## Usage
Getting started with SkyNet is pretty straightforward. Here's a step-by-step guide:

1. **Setup Your Environment:**

Before diving in, ensure you have numpy:
```bash
pip install numpy
```

2. **Initialize Neural Network Layers:** 

Navigate to the SkyNet directory:
```bash
cd skynet
```

Now, create your neural layers:
```bash
from skynet import MLP
from skynet.layers import Dense 

mlp = MLP()

# Add the first hidden layer with 3 neurons and sigmoid activation
mlp.add(Dense(2, 3, activation="sigmoid"))

# Add the output layer with 1 neuron and sigmoid activation
mlp.add(Dense(3, 1, activation="sigmoid"))
```

3. **Building and Training:**

Use the library's functions to construct a neural network model and train it using your data.
```bash
# compile the model with hyperparameters 
mlp.compile(epochs=1000, learning_rate=0.01, optimizer="vanilla", batch_size=1, loss="mse")

# create a training set (featueres as columns)
X = np.array([[0.5, 0.2], 
              [0.1, 0.6]])

y = np.array([[0.7, 0.8]])

# use the fit method to train the model
mlp.fit(X, y)
```

4. **Predictions:**

Once your model is trained, making predictions is easy!
```bash
trained_output = mlp.forward(X)
``` 

## Acknowledgements

Big thanks to Numpy for being the foundation of this project.
Grateful for every tutorial and resource that made this learning journey smoother.