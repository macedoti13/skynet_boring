import numpy as np
from typing import List, Tuple

class Sequential:

    def __init__(self) -> None:
        """
        Initialize an instance of the Sequential (Multi-Layer Perceptron) class.

        Attributes:
        - layers: List of layers added to the neural network.
        """
        self.layers: list = []


    def add(self, layer) -> None: 
        """
        Add a new layer to the neural network.

        Args:
        - layer: The layer to be added.
        """
        self.layers.append(layer)


    def compile(self, epochs: int, learning_rate: float, optimizer: str, batch_size: int, loss: str) -> None:
        """
        Configure the learning process before training starts.

        Args:
        - epochs: Number of epochs the model will be trained.
        - learning_rate: Learning rate for optimization.
        - optimizer: Optimization algorithm to use.
        - batch_size: Size of each mini-batch.
        - loss: Loss function to be used.
        """
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.loss = loss
        self.loss_epochs = []


    def forward(self, X: np.array) -> np.array:
        """
        Perform a forward pass through all the layers of the neural network.

        Args:
        - X: Input data.

        Returns:
        - X: Output from the last layer.
        """
        for layer in self.layers:
            X = layer.forward(X)
        return X 


    def backward(self, d_output: np.array) -> None:
        """
        Perform a backward pass (backpropagation) through all the layers of the neural network in reverse order.

        Args:
        - d_output: Gradient of the loss with respect to the output.
        """
        for layer in reversed(self.layers):
            d_output = layer.backward(d_output)  # Compute the gradient with respect to layer's input.
            layer.optimize(self.learning_rate, self.optimizer)  # Update the layer's weights and biases.


    def compute_loss(self, yhat: np.array, y: np.array) -> np.array:
        """
        Compute the loss between the predictions and true labels.

        Args:
        - yhat: Predictions.
        - y: True labels.

        Returns:
        - loss: Computed loss.
        """
        num_samples = y.shape[0]
        
        if self.loss == "mse":
            loss = np.mean(0.5 * (yhat - y)**2)
            return loss
        
        elif self.loss == "mae":
            loss = np.mean(np.abs(yhat - y))
            return loss
        
        elif self.loss == "cross_entropy":
            epsilon = 1e-15  # To avoid log(0)
            yhat = np.clip(yhat, epsilon, 1 - epsilon)
            loss = -np.sum(y * np.log(yhat)) / num_samples
            return loss

        elif self.loss == "binary_cross_entropy":
            epsilon = 1e-15  # To avoid log(0)
            yhat = np.clip(yhat, epsilon, 1 - epsilon)
            loss = -np.sum(y * np.log(yhat) + (1 - y) * np.log(1 - yhat)) / num_samples
            return loss


    def compute_gradient_of_loss(self, yhat: np.array, y: np.array) -> np.array:
        """
        Compute the gradient of the loss with respect to predictions.

        Args:
        - yhat: Predictions.
        - y: True labels.

        Returns:
        - Gradient of the loss.
        """
        if self.loss == "mse":
            return yhat - y 
        
        elif self.loss == "mae":
            return np.where(yhat > y, 1, np.where(yhat < y, -1, 0))
        
        elif self.loss == "cross_entropy":
            return yhat - y
        
        elif self.loss == "binary_cross_entropy":
            return (yhat - y) / (yhat * (1 - yhat))


    def create_mini_batches(self, X: np.array, y: np.array, batch_size: int) -> List[Tuple[np.array, np.array]]:
        """
        Create mini-batches from the provided data.

        Args:
        - X: Input data.
        - y: True labels.
        - batch_size: Desired size of each mini-batch.

        Returns:
        - mini_batches: List of tuples, where each tuple contains a mini-batch of data and corresponding labels.
        """
        # Create an array of indices from 0 to the number of samples.
        indices = np.arange(X.shape[1])
        np.random.shuffle(indices)  # Shuffle the indices.
        X = X[:, indices]  # Shuffle X using the shuffled indices.
        y = y[:, indices]  # Shuffle y using the shuffled indices.

        mini_batches = []

        total_batches = X.shape[1] // batch_size
        for i in range(total_batches):
            X_mini = X[:, i * batch_size: (i + 1) * batch_size]
            y_mini = y[:, i * batch_size: (i + 1) * batch_size]
            mini_batches.append((X_mini, y_mini))

        # Handle the end case (last mini-batch < mini_batch_size)
        if X.shape[1] % batch_size != 0:
            X_mini = X[:, total_batches * batch_size:]
            y_mini = y[:, total_batches * batch_size:]
            mini_batches.append((X_mini, y_mini))

        return mini_batches


    def fit(self, X: np.array, y: np.array) -> None:
        """
        Train the neural network using the provided data.

        Args:
        - X: Input data for training.
        - y: True labels for training.
        """
        for _ in range(self.epochs):
            mini_batches = self.create_mini_batches(X, y, self.batch_size)
            loss_batches = []  # List to store loss of each mini-batch.

            for mini_batch in mini_batches:
                X_mini, y_mini = mini_batch  # Unpack mini-batch data and labels.

                yhat_mini = self.forward(X_mini)  # Forward pass: compute predictions.

                loss_mini = self.compute_loss(yhat_mini, y_mini)  # Compute loss of the mini-batch.

                loss_gradient_mini = self.compute_gradient_of_loss(yhat_mini, y_mini)  # Compute gradient of the loss.

                loss_batches.append(loss_mini)  # Store the loss.

                self.backward(loss_gradient_mini)  # Backward pass: update weights and biases.

            epoch_loss = np.mean(loss_batches)  # Compute average loss for the epoch.

            print(f"Epoch Loss: {epoch_loss}")
            self.loss_epochs.append(epoch_loss)  # Store the average epoch loss.
            
            
    def predict(self, X):
        """
        Use the neural network to predict a given input matrix X
        
        Args:
        - X: Input data for prediction
        """
        return self.forward(X)
