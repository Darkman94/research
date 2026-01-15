"""
Simple neural network implementation for use with simulated annealing.
"""
import numpy as np
from typing import List, Callable, Tuple


class NeuralNetwork:
    """
    A simple feedforward neural network (multi-layer perceptron).

    This implementation is designed to work with simulated annealing optimization,
    so it provides easy access to all parameters as a flat vector.
    """

    def __init__(self, layer_sizes: List[int], activation: str = 'tanh',
                 output_activation: str = 'linear', seed: int = None):
        """
        Initialize the neural network.

        Args:
            layer_sizes: List of layer sizes, e.g., [2, 4, 1] for input->hidden->output
            activation: Activation function for hidden layers ('sigmoid', 'tanh', 'relu')
            output_activation: Activation function for output layer ('linear', 'sigmoid', 'tanh')
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)

        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.activation_name = activation
        self.output_activation_name = output_activation

        # Initialize weights and biases with small random values
        self.weights = []
        self.biases = []

        for i in range(len(layer_sizes) - 1):
            # He initialization for ReLU, Xavier for tanh/sigmoid
            if activation == 'relu':
                scale = np.sqrt(2.0 / layer_sizes[i])
            else:
                scale = np.sqrt(1.0 / layer_sizes[i])

            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * scale
            b = np.zeros((1, layer_sizes[i + 1]))

            self.weights.append(w)
            self.biases.append(b)

        # Set activation functions
        self.activation = self._get_activation_fn(activation)
        self.output_activation = self._get_activation_fn(output_activation)

    def _get_activation_fn(self, name: str) -> Callable:
        """Get activation function by name."""
        activations = {
            'sigmoid': lambda x: 1 / (1 + np.exp(-np.clip(x, -500, 500))),
            'tanh': np.tanh,
            'relu': lambda x: np.maximum(0, x),
            'linear': lambda x: x
        }

        if name not in activations:
            raise ValueError(f"Unknown activation: {name}. Choose from {list(activations.keys())}")

        return activations[name]

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward propagation through the network.

        Args:
            X: Input data of shape (n_samples, n_features)

        Returns:
            Output predictions of shape (n_samples, n_outputs)
        """
        activation = X

        # Forward through all hidden layers
        for i in range(len(self.weights) - 1):
            z = np.dot(activation, self.weights[i]) + self.biases[i]
            activation = self.activation(z)

        # Output layer
        z = np.dot(activation, self.weights[-1]) + self.biases[-1]
        output = self.output_activation(z)

        return output

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Alias for forward propagation."""
        return self.forward(X)

    def get_parameters(self) -> np.ndarray:
        """
        Get all network parameters as a single flat vector.

        Returns:
            1D array containing all weights and biases
        """
        params = []

        for w, b in zip(self.weights, self.biases):
            params.append(w.flatten())
            params.append(b.flatten())

        return np.concatenate(params)

    def set_parameters(self, params: np.ndarray):
        """
        Set all network parameters from a flat vector.

        Args:
            params: 1D array containing all weights and biases
        """
        idx = 0

        for i in range(len(self.weights)):
            # Extract weights
            w_size = self.weights[i].size
            w_shape = self.weights[i].shape
            self.weights[i] = params[idx:idx + w_size].reshape(w_shape)
            idx += w_size

            # Extract biases
            b_size = self.biases[i].size
            b_shape = self.biases[i].shape
            self.biases[i] = params[idx:idx + b_size].reshape(b_shape)
            idx += b_size

    def get_num_parameters(self) -> int:
        """Get total number of parameters in the network."""
        return sum(w.size + b.size for w, b in zip(self.weights, self.biases))

    def compute_loss(self, X: np.ndarray, y: np.ndarray, loss_type: str = 'mse') -> float:
        """
        Compute loss on given data.

        Args:
            X: Input data
            y: Target values
            loss_type: Type of loss ('mse' for regression, 'binary_crossentropy' for binary classification)

        Returns:
            Loss value
        """
        predictions = self.forward(X)

        if loss_type == 'mse':
            return np.mean((predictions - y) ** 2)
        elif loss_type == 'mae':
            return np.mean(np.abs(predictions - y))
        elif loss_type == 'binary_crossentropy':
            # Clip predictions to avoid log(0)
            predictions = np.clip(predictions, 1e-7, 1 - 1e-7)
            return -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def copy(self) -> 'NeuralNetwork':
        """Create a deep copy of the network."""
        new_nn = NeuralNetwork(self.layer_sizes, self.activation_name,
                              self.output_activation_name)
        new_nn.set_parameters(self.get_parameters().copy())
        return new_nn

    def __repr__(self) -> str:
        """String representation of the network."""
        return (f"NeuralNetwork(layers={self.layer_sizes}, "
                f"activation={self.activation_name}, "
                f"output_activation={self.output_activation_name}, "
                f"params={self.get_num_parameters()})")
