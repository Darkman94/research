"""
Example: Comparing simulated annealing with gradient descent for neural network training.
"""
import numpy as np
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork
from simulated_annealing import train_neural_network
import time


def generate_data(n_samples=200, noise=0.1):
    """Generate synthetic data for regression."""
    np.random.seed(42)
    X = np.linspace(-3, 3, n_samples).reshape(-1, 1)
    y = np.sin(X) + 0.5 * np.cos(2 * X) + noise * np.random.randn(n_samples, 1)
    return X, y


def train_with_gradient_descent(network, X, y, learning_rate=0.01, epochs=100, verbose=False):
    """
    Simple gradient descent training using numerical gradients.

    Note: This is a basic implementation for comparison purposes.
    In practice, use frameworks like PyTorch or TensorFlow.
    """
    history = {'loss': []}

    for epoch in range(epochs):
        # Forward pass
        predictions = network.predict(X)
        loss = network.compute_loss(X, y, loss_type='mse')
        history['loss'].append(loss)

        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"Epoch {epoch:3d} | Loss: {loss:.6f}")

        # Compute numerical gradients
        params = network.get_parameters()
        gradients = np.zeros_like(params)
        epsilon = 1e-5

        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += epsilon
            network.set_parameters(params_plus)
            loss_plus = network.compute_loss(X, y, loss_type='mse')

            params_minus = params.copy()
            params_minus[i] -= epsilon
            network.set_parameters(params_minus)
            loss_minus = network.compute_loss(X, y, loss_type='mse')

            gradients[i] = (loss_plus - loss_minus) / (2 * epsilon)

        # Update parameters
        network.set_parameters(params)
        new_params = params - learning_rate * gradients
        network.set_parameters(new_params)

    return network, history


def main():
    print("=" * 70)
    print("Comparison: Simulated Annealing vs Gradient Descent")
    print("=" * 70)
    print()

    # Generate data
    print("Generating synthetic data...")
    X_train, y_train = generate_data(n_samples=100, noise=0.1)
    print()

    # ========================================================================
    # Train with Simulated Annealing
    # ========================================================================
    print("Training with SIMULATED ANNEALING...")
    print("-" * 70)

    network_sa = NeuralNetwork(
        layer_sizes=[1, 8, 1],
        activation='tanh',
        output_activation='linear',
        seed=42
    )

    start_time = time.time()
    network_sa, optimizer_sa, history_sa = train_neural_network(
        network_sa,
        X_train, y_train,
        loss_type='mse',
        initial_temperature=2.0,
        final_temperature=0.01,
        cooling_rate=0.95,
        iterations_per_temp=30,
        perturbation_scale=0.5,
        adaptive_perturbation=True,
        seed=42,
        verbose=True
    )
    sa_time = time.time() - start_time

    sa_loss = network_sa.compute_loss(X_train, y_train, loss_type='mse')
    print()

    # ========================================================================
    # Train with Gradient Descent
    # ========================================================================
    print("Training with GRADIENT DESCENT...")
    print("-" * 70)

    network_gd = NeuralNetwork(
        layer_sizes=[1, 8, 1],
        activation='tanh',
        output_activation='linear',
        seed=42
    )

    start_time = time.time()
    network_gd, history_gd = train_with_gradient_descent(
        network_gd,
        X_train, y_train,
        learning_rate=0.01,
        epochs=100,
        verbose=True
    )
    gd_time = time.time() - start_time

    gd_loss = network_gd.compute_loss(X_train, y_train, loss_type='mse')
    print()

    # ========================================================================
    # Compare Results
    # ========================================================================
    print("=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    print(f"{'Method':<25} {'Final Loss':<15} {'Time (s)':<15}")
    print("-" * 70)
    print(f"{'Simulated Annealing':<25} {sa_loss:<15.6f} {sa_time:<15.2f}")
    print(f"{'Gradient Descent':<25} {gd_loss:<15.6f} {gd_time:<15.2f}")
    print()

    if sa_loss < gd_loss:
        print(f"Simulated Annealing achieved {(1 - sa_loss/gd_loss)*100:.1f}% lower loss")
    else:
        print(f"Gradient Descent achieved {(1 - gd_loss/sa_loss)*100:.1f}% lower loss")
    print()

    # ========================================================================
    # Plot Comparison
    # ========================================================================
    print("Generating comparison plots...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Predictions comparison
    ax = axes[0, 0]
    ax.scatter(X_train, y_train, alpha=0.5, label='True data', s=20, c='gray')
    pred_sa = network_sa.predict(X_train)
    pred_gd = network_gd.predict(X_train)

    # Sort for better line plots
    sort_idx = np.argsort(X_train.ravel())
    ax.plot(X_train[sort_idx], pred_sa[sort_idx], 'r-', linewidth=2, label='Simulated Annealing')
    ax.plot(X_train[sort_idx], pred_gd[sort_idx], 'b--', linewidth=2, label='Gradient Descent')
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.set_title('Model Predictions Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: SA training progress
    ax = axes[0, 1]
    ax.plot(history_sa['cost'], alpha=0.3, label='Current cost')
    ax.plot(history_sa['best_cost'], 'r-', linewidth=2, label='Best cost')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('Simulated Annealing Progress')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: GD training progress
    ax = axes[1, 0]
    ax.plot(history_gd['loss'], 'b-', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('Gradient Descent Progress')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # Plot 4: Loss comparison on log scale
    ax = axes[1, 1]

    # Normalize iterations for comparison
    sa_iters = np.arange(len(history_sa['best_cost']))
    gd_iters = np.linspace(0, len(history_sa['best_cost']), len(history_gd['loss']))

    ax.plot(sa_iters, history_sa['best_cost'], 'r-', linewidth=2, label='Simulated Annealing')
    ax.plot(gd_iters, history_gd['loss'], 'b-', linewidth=2, label='Gradient Descent')
    ax.set_xlabel('Iteration (normalized)')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('Training Progress Comparison')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    import os
    save_path = os.path.join(os.path.dirname(__file__), 'comparison_results.png')
    plt.savefig(save_path, dpi=150)
    print(f"Saved plot to: {save_path}")
    print()


if __name__ == '__main__':
    main()
