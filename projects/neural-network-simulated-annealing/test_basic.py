"""
Basic tests to verify the neural network and simulated annealing implementation.
"""
import numpy as np
from neural_network import NeuralNetwork
from simulated_annealing import SimulatedAnnealing, train_neural_network


def test_neural_network():
    """Test basic neural network functionality."""
    print("Testing Neural Network...")

    # Create network
    network = NeuralNetwork([2, 4, 1], activation='tanh', seed=42)

    # Test forward pass
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    predictions = network.predict(X)
    assert predictions.shape == (4, 1), "Prediction shape mismatch"

    # Test loss computation
    loss = network.compute_loss(X, y, loss_type='mse')
    assert isinstance(loss, (float, np.floating)), "Loss should be a scalar"
    assert loss >= 0, "Loss should be non-negative"

    # Test parameter get/set
    params = network.get_parameters()
    assert params.ndim == 1, "Parameters should be 1D"

    num_params = network.get_num_parameters()
    assert len(params) == num_params, "Parameter count mismatch"

    # Test parameter modification
    new_params = params + 0.1
    network.set_parameters(new_params)
    retrieved_params = network.get_parameters()
    assert np.allclose(retrieved_params, new_params), "Parameter set/get mismatch"

    print("✓ Neural Network tests passed")


def test_simulated_annealing():
    """Test simulated annealing optimizer."""
    print("Testing Simulated Annealing...")

    # Simple quadratic function: minimize (x-3)^2 + (y-4)^2
    def objective(params):
        x, y = params
        return (x - 3) ** 2 + (y - 4) ** 2

    # Create optimizer
    optimizer = SimulatedAnnealing(
        initial_temperature=1.0,
        final_temperature=0.01,
        cooling_rate=0.9,
        iterations_per_temp=20,
        perturbation_scale=0.5,
        seed=42,
        verbose=False
    )

    # Optimize
    initial_params = np.array([0.0, 0.0])
    best_params, best_cost = optimizer.optimize(objective, initial_params)

    # Check convergence
    assert best_cost < 1.0, "Should converge to low cost"
    assert abs(best_params[0] - 3.0) < 1.0, "Should converge to x=3"
    assert abs(best_params[1] - 4.0) < 1.0, "Should converge to y=4"

    # Check history tracking
    history = optimizer.get_history()
    assert len(history['cost']) > 0, "Should have cost history"
    assert len(history['temperature']) > 0, "Should have temperature history"

    print("✓ Simulated Annealing tests passed")


def test_training():
    """Test neural network training with simulated annealing."""
    print("Testing Neural Network Training...")

    # Simple dataset
    np.random.seed(42)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)  # XOR

    # Create and train network
    network = NeuralNetwork([2, 4, 1], activation='tanh', output_activation='sigmoid', seed=42)

    initial_loss = network.compute_loss(X, y, loss_type='mse')

    network, optimizer, history = train_neural_network(
        network, X, y,
        loss_type='mse',
        initial_temperature=1.0,
        final_temperature=0.01,
        cooling_rate=0.9,
        iterations_per_temp=20,
        perturbation_scale=0.5,
        seed=42,
        verbose=False
    )

    final_loss = network.compute_loss(X, y, loss_type='mse')

    # Check improvement
    assert final_loss < initial_loss, "Training should improve loss"
    assert len(history['cost']) > 0, "Should have training history"

    print(f"  Initial loss: {initial_loss:.6f}")
    print(f"  Final loss: {final_loss:.6f}")
    print(f"  Improvement: {(1 - final_loss/initial_loss)*100:.1f}%")
    print("✓ Neural Network Training tests passed")


def main():
    print("=" * 70)
    print("Running Basic Tests")
    print("=" * 70)
    print()

    test_neural_network()
    test_simulated_annealing()
    test_training()

    print()
    print("=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)


if __name__ == '__main__':
    main()
