# Neural Network Training with Simulated Annealing

A toolset for training neural networks using simulated annealing optimization instead of traditional gradient-based methods.

## Overview

While neural networks are typically trained using gradient-based optimization algorithms (SGD, Adam, etc.), they can in principle be trained with any optimization algorithm. This project implements a complete toolset for training neural networks using **simulated annealing**, a probabilistic optimization technique inspired by the physical process of annealing in metallurgy.

## Why Simulated Annealing?

**Advantages:**
- **No gradient computation required**: Works with non-differentiable activation functions or loss functions
- **Global optimization**: Better at escaping local minima compared to gradient descent
- **Simplicity**: No need to tune learning rates, momentum, or other optimizer hyperparameters
- **Robustness**: Can handle noisy or discontinuous objective functions

**Disadvantages:**
- **Slower convergence**: Typically requires many more function evaluations than gradient-based methods
- **Computational cost**: Each iteration evaluates the full objective function
- **Not suitable for large networks**: Scales poorly with the number of parameters

## Components

### 1. Neural Network (`neural_network.py`)

A simple feedforward neural network (multi-layer perceptron) with:
- Configurable architecture (layer sizes)
- Multiple activation functions (sigmoid, tanh, ReLU, linear)
- Easy parameter access (get/set all weights as a flat vector)
- Support for different loss functions (MSE, MAE, binary cross-entropy)

```python
from neural_network import NeuralNetwork

# Create a network with 2 inputs, two hidden layers (8, 4 neurons), and 1 output
network = NeuralNetwork(
    layer_sizes=[2, 8, 4, 1],
    activation='tanh',
    output_activation='sigmoid'
)

# Make predictions
predictions = network.predict(X)

# Compute loss
loss = network.compute_loss(X, y, loss_type='mse')
```

### 2. Simulated Annealing Optimizer (`simulated_annealing.py`)

A flexible simulated annealing implementation with:
- Configurable temperature schedule (initial temp, final temp, cooling rate)
- Adaptive perturbation scaling based on acceptance rate
- Comprehensive training history tracking
- Callback support for custom monitoring

```python
from simulated_annealing import SimulatedAnnealing

# Create optimizer
optimizer = SimulatedAnnealing(
    initial_temperature=2.0,
    final_temperature=0.01,
    cooling_rate=0.95,
    iterations_per_temp=50,
    perturbation_scale=0.5,
    adaptive_perturbation=True
)

# Optimize any objective function
best_params, best_cost = optimizer.optimize(objective_fn, initial_params)
```

### 3. Training Utilities

Helper function for training neural networks:

```python
from simulated_annealing import train_neural_network

network, optimizer, history = train_neural_network(
    network,
    X_train, y_train,
    loss_type='mse',
    initial_temperature=2.0,
    final_temperature=0.01,
    cooling_rate=0.95,
    iterations_per_temp=50
)
```

## Examples

### Regression Example

Train a neural network to fit a sinusoidal function:

```bash
cd projects/neural-network-simulated-annealing
python example_regression.py
```

This demonstrates:
- Training a network on continuous data
- Visualizing the learned function
- Monitoring training progress (loss, temperature, acceptance rate)

### Classification Example (XOR Problem)

Train a neural network to solve the classic XOR problem:

```bash
python example_classification.py
```

This demonstrates:
- Binary classification
- Decision boundary visualization
- Handling non-linearly separable data

### Comparison with Gradient Descent

Compare simulated annealing with traditional gradient descent:

```bash
python comparison_with_gradient_descent.py
```

This demonstrates:
- Side-by-side comparison of both methods
- Trade-offs between convergence speed and solution quality
- When to prefer each approach

## Algorithm Details

### Simulated Annealing Process

1. **Initialize**: Start with random parameters and high temperature
2. **Iterate**: At each temperature level:
   - Generate a neighboring solution by perturbing current parameters
   - Evaluate the objective function
   - Accept the new solution if it's better
   - Accept worse solutions with probability `exp(-ΔE/T)` (Metropolis criterion)
3. **Cool down**: Gradually decrease temperature: `T_new = cooling_rate * T_old`
4. **Terminate**: Stop when temperature reaches the final threshold

### Key Parameters

- **initial_temperature**: Higher values allow more exploration (typically 1-10)
- **final_temperature**: Lower values ensure convergence (typically 0.001-0.1)
- **cooling_rate**: Controls how fast temperature decreases (typically 0.9-0.99)
- **iterations_per_temp**: Number of iterations at each temperature (typically 50-200)
- **perturbation_scale**: Size of random parameter perturbations (adaptive)

### Adaptive Perturbation

The implementation includes adaptive perturbation scaling that adjusts the step size based on the acceptance rate:
- **Target acceptance rate**: ~44% (theoretically optimal for many problems)
- **Too high**: Increase perturbation to explore more
- **Too low**: Decrease perturbation to improve acceptance

## Requirements

```
numpy
matplotlib
```

Install with:
```bash
pip install -r ../../requirements.txt
```

## Limitations and Future Work

**Current Limitations:**
- Not suitable for large-scale networks (>1000 parameters)
- Slower than gradient-based methods on differentiable problems
- No mini-batch support
- Basic cooling schedule (exponential)

**Potential Improvements:**
- Parallel tempering for better exploration
- Adaptive cooling schedules
- Hybrid approaches (SA for initialization + gradient descent for fine-tuning)
- Population-based methods (genetic algorithms, particle swarm)
- GPU acceleration for fitness evaluation

## Theoretical Background

Simulated annealing is based on the Metropolis-Hastings algorithm and is guaranteed to find the global optimum given:
1. Sufficiently slow cooling schedule (logarithmic)
2. Infinite time

In practice, we use faster cooling schedules and finite time, which gives good approximate solutions.

## References

- Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983). "Optimization by Simulated Annealing"
- Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). "Learning representations by back-propagating errors"

## License

MIT License - See root LICENSE file for details
