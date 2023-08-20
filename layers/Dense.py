import numpy as np

class Dense:
    """
    Fully Connected Neural Network Layer (Dense Layer).

    This layer is characterized by a weights matrix, bias vector, and an optional activation function.
    The weights and biases are initialized during the object instantiation and can be optimized using backpropagation.
    """

    def __init__(self, input_size: int, output_size: int, activation: str = None) -> None:
        """
        Initializes the Dense layer with randomly generated weights and zeros biases.

        Args:
            input_size (int): The number of input features.
            output_size (int): The number of output neurons or units.
            activation (str, optional): The activation function to be applied after the linear transformation.
                Supported values are "sigmoid", "relu", or None for no activation. Defaults to None.
        """
        self.weights = np.random.randn(output_size, input_size) * 0.01
        self.bias = np.zeros((output_size, 1))
        self.activation = activation

    def forward(self, input_data: np.array) -> np.array:
        """
        Performs the forward pass of the neural network.

        This involves a linear transformation using the weights and biases followed by an activation function if specified.

        Args:
            input_data (np.array): The input data for the forward pass.

        Returns:
            np.array: The resulting output after applying the linear transformation and activation function.
        """
        self.input_data = input_data
        self.z = np.dot(self.weights, self.input_data) + self.bias

        if self.activation is not None:
            self.a = self.apply_activation(self.z)
            return self.a

        return self.z 

    def apply_activation(self, z: np.array) -> np.array:
        """
        Applies the specified activation function to the input.

        Args:
            z (np.array): The input data after linear transformation.

        Returns:
            np.array: The output after applying the activation function.
        """
        if self.activation == "sigmoid":
            return 1 / (1 + np.exp(-z))
        elif self.activation == "relu":
            return np.maximum(0, z)

        return z # default is linear activation

    def backward(self, d_output: np.array) -> np.array:
        """
        For the last layer (layer L), the `d_output` is essentially the gradient of the loss with respect to the output (activation) of this layer. In other words, d_output represents the partial derivative of the loss with respect to the activation of layer L, often noted as "d_loss/d_a_L". To get the delta (or error term) for this last layer, it's necessary to multiply it element-wise with the gradient of the activation function. This relationship can be expressed as "delta_L = d_loss/d_a_L times the derivative of a_L with respect to z_L". This is achieved by the line: 
            `d_output = d_output * (sigmoid_output * (1 - sigmoid_output))`, 
            where `sigmoid_output` is the activation of this layer.

            For any preceding layer (l), where l is less than L, the error term or delta of that layer is determined by multiplying the delta from the next layer by the transposed weight matrix of the next layer and the derivative of the activation function for layer l. Specifically, the equation for this is "delta_l = (weights of layer l+1 transposed dot product with delta of layer l+1) times the derivative of the activation function at z_l". 

            The line `d_input = np.dot(d_output, self.weights.T)` computes the first part of that equation, which will then serve as the `d_output` for the next backward pass through layer l-1. 

            With the computed delta, the gradients with respect to the weights and biases for the current layer are determined using:
            `d_weights = np.dot(self.input_data.T, d_output)` and 
            `d_biases = np.sum(d_output, axis=0, keepdims=True)`.
            
            These gradients are stored for optimization purposes. Finally, `d_input` is returned and serves as the `d_output` for the previous layer in the next invocation of the `backward` function, ensuring the error is backpropagated through the entire network.

        Args:
            d_output (np.array): 
                - For the last layer (L), it represents the gradient of the loss with respect to the network's output, often noted as "d_loss/d_a_L".
                - For any other layer (l) where l is less than L, it is the dot product of the transposed weights of the next layer and the delta of the next layer.

        Returns:
            np.array: The product of the current layer's delta and the transposed weight matrix, which will serve as the `d_output` for the previous layer.
        """
        if self.activation == 'relu':
            d_output = d_output * (self.z > 0)
        elif self.activation == 'sigmoid':
            sigmoid_output = self.apply_activation(self.z)
            d_output = d_output * (sigmoid_output * (1 - sigmoid_output))

        d_input = np.dot(self.weights.T, d_output)
        d_weights = np.dot(d_output, self.input_data.T)
        d_biases = np.sum(d_output, axis=1, keepdims=True)

        self.d_weights = d_weights
        self.d_biases = d_biases

        return d_input

    def optimize(self, learning_rate: float = 0.01, optimizer: str = "vanilla") -> None:
        """
        Optimizes the layer's weights and biases using the computed gradients.

        Currently, only vanilla gradient descent is implemented for optimization.

        Args:
            learning_rate (float, optional): The step size for gradient descent optimization. Defaults to 0.01.
            optimizer (str, optional): The optimization strategy. Defaults to "vanilla".
        """
        if optimizer == "vanilla":
            self.weights = self.weights - learning_rate * self.d_weights
            self.bias = self.bias - learning_rate * self.d_biases
