"""
Example: Training a neural network on XOR classification using simulated annealing.
"""
import numpy as np
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork
from simulated_annealing import train_neural_network


def generate_xor_data(n_samples_per_class=100, noise=0.1):
    """Generate XOR dataset with some noise."""
    np.random.seed(42)

    # Class 1: (0,0) and (1,1) -> 0
    X1 = np.random.randn(n_samples_per_class, 2) * noise + np.array([0, 0])
    X2 = np.random.randn(n_samples_per_class, 2) * noise + np.array([1, 1])

    # Class 2: (0,1) and (1,0) -> 1
    X3 = np.random.randn(n_samples_per_class, 2) * noise + np.array([0, 1])
    X4 = np.random.randn(n_samples_per_class, 2) * noise + np.array([1, 0])

    X = np.vstack([X1, X2, X3, X4])
    y = np.array([0] * (2 * n_samples_per_class) + [1] * (2 * n_samples_per_class)).reshape(-1, 1)

    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    return X, y


def plot_decision_boundary(network, X, y, ax=None):
    """Plot decision boundary of the neural network."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # Create mesh
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    # Predict on mesh
    Z = network.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot
    ax.contourf(xx, yy, Z, levels=20, cmap='RdYlBu', alpha=0.6)
    ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)

    # Plot data points
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='RdYlBu',
                        edgecolors='black', s=50, linewidths=1.5)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_title('Decision Boundary')

    return ax


def main():
    print("=" * 70)
    print("Neural Network Training with Simulated Annealing - XOR Classification")
    print("=" * 70)
    print()

    # Generate data
    print("Generating XOR dataset...")
    X_train, y_train = generate_xor_data(n_samples_per_class=50, noise=0.1)
    print(f"Dataset: {len(X_train)} samples")
    print(f"Class distribution: {np.sum(y_train == 0)} class 0, {np.sum(y_train == 1)} class 1")
    print()

    # Create neural network
    print("Creating neural network...")
    network = NeuralNetwork(
        layer_sizes=[2, 8, 1],
        activation='tanh',
        output_activation='sigmoid',
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
        loss_type='binary_crossentropy',
        initial_temperature=5.0,
        final_temperature=0.01,
        cooling_rate=0.95,
        iterations_per_temp=50,
        perturbation_scale=1.0,
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
    pred_classes = (predictions > 0.5).astype(int)
    accuracy = np.mean(pred_classes == y_train)
    final_loss = network.compute_loss(X_train, y_train, loss_type='binary_crossentropy')

    print(f"Final training loss: {final_loss:.6f}")
    print(f"Training accuracy: {accuracy:.2%}")
    print()

    # Plot results
    print("Generating plots...")

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # Plot 1: Decision boundary
    ax = fig.add_subplot(gs[:, 0])
    plot_decision_boundary(network, X_train, y_train, ax)

    # Plot 2: Cost over iterations
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(history['cost'], alpha=0.3, label='Current cost')
    ax.plot(history['best_cost'], 'r-', linewidth=2, label='Best cost')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cost (Binary Cross-Entropy)')
    ax.set_title('Training Progress')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Temperature schedule
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(history['temperature'], 'b-', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Temperature')
    ax.set_title('Temperature Schedule')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # Plot 4: Acceptance rate
    ax = fig.add_subplot(gs[1, 1])
    iterations_per_temp = 50
    x_temps = np.arange(len(history['acceptance_rate'])) * iterations_per_temp
    ax.plot(x_temps, history['acceptance_rate'], 'g-', linewidth=2)
    ax.axhline(y=0.44, color='r', linestyle='--', label='Target rate (0.44)')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Acceptance Rate')
    ax.set_title('Acceptance Rate Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 5: Perturbation scale
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(x_temps, history['perturbation_scale'], 'orange', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Perturbation Scale')
    ax.set_title('Adaptive Perturbation Scale')
    ax.grid(True, alpha=0.3)

    import os
    save_path = os.path.join(os.path.dirname(__file__), 'classification_results.png')
    plt.savefig(save_path, dpi=150)
    print(f"Saved plot to: {save_path}")
    print()


if __name__ == '__main__':
    main()
