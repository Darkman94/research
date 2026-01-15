"""
Example: Training a neural network on a simple regression task using simulated annealing.
"""
import numpy as np
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork
from simulated_annealing import train_neural_network


def generate_data(n_samples=200, noise=0.1):
    """Generate synthetic data for regression."""
    np.random.seed(42)
    X = np.linspace(-3, 3, n_samples).reshape(-1, 1)
    y = np.sin(X) + 0.5 * np.cos(2 * X) + noise * np.random.randn(n_samples, 1)
    return X, y


def main():
    print("=" * 70)
    print("Neural Network Training with Simulated Annealing - Regression Example")
    print("=" * 70)
    print()

    # Generate data
    print("Generating synthetic data...")
    X_train, y_train = generate_data(n_samples=200, noise=0.1)

    # Create neural network
    print("Creating neural network...")
    network = NeuralNetwork(
        layer_sizes=[1, 10, 10, 1],
        activation='tanh',
        output_activation='linear',
        seed=42
    )
    print(network)
    print()

    # Train with simulated annealing
    print("Training with simulated annealing...")
    print()

    network, optimizer, history = train_neural_network(
        network,
        X_train, y_train,
        loss_type='mse',
        initial_temperature=2.0,
        final_temperature=0.01,
        cooling_rate=0.95,
        iterations_per_temp=50,
        perturbation_scale=0.5,
        adaptive_perturbation=True,
        seed=42,
        verbose=True
    )

    print()
    print("=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print()

    # Evaluate
    predictions = network.predict(X_train)
    final_loss = network.compute_loss(X_train, y_train, loss_type='mse')
    print(f"Final training MSE: {final_loss:.6f}")
    print()

    # Plot results
    print("Generating plots...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Data and predictions
    ax = axes[0, 0]
    ax.scatter(X_train, y_train, alpha=0.5, label='True data', s=20)
    ax.plot(X_train, predictions, 'r-', linewidth=2, label='NN prediction')
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.set_title('Neural Network Fit')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Cost over iterations
    ax = axes[0, 1]
    ax.plot(history['cost'], alpha=0.3, label='Current cost')
    ax.plot(history['best_cost'], 'r-', linewidth=2, label='Best cost')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cost (MSE)')
    ax.set_title('Training Progress')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Temperature schedule
    ax = axes[1, 0]
    ax.plot(history['temperature'], 'b-', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Temperature')
    ax.set_title('Temperature Schedule')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # Plot 4: Acceptance rate and perturbation scale
    ax = axes[1, 1]
    iterations_per_temp = 50
    x_temps = np.arange(len(history['acceptance_rate'])) * iterations_per_temp
    ax1 = ax
    ax2 = ax.twinx()

    line1 = ax1.plot(x_temps, history['acceptance_rate'], 'g-', linewidth=2, label='Acceptance rate')
    line2 = ax2.plot(x_temps, history['perturbation_scale'], 'orange', linewidth=2, label='Perturbation scale')

    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Acceptance Rate', color='g')
    ax2.set_ylabel('Perturbation Scale', color='orange')
    ax1.set_title('Acceptance Rate and Perturbation Scale')
    ax1.tick_params(axis='y', labelcolor='g')
    ax2.tick_params(axis='y', labelcolor='orange')
    ax1.grid(True, alpha=0.3)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='best')

    plt.tight_layout()

    import os
    save_path = os.path.join(os.path.dirname(__file__), 'regression_results.png')
    plt.savefig(save_path, dpi=150)
    print(f"Saved plot to: {save_path}")
    print()


if __name__ == '__main__':
    main()
