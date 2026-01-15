"""
Simulated annealing optimizer for neural network training.
"""
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple
import time


class SimulatedAnnealing:
    """
    Simulated Annealing optimizer.

    This optimizer can be used to train neural networks or optimize any
    objective function that takes a parameter vector and returns a scalar cost.
    """

    def __init__(self,
                 initial_temperature: float = 1.0,
                 final_temperature: float = 0.01,
                 cooling_rate: float = 0.95,
                 iterations_per_temp: int = 100,
                 perturbation_scale: float = 0.1,
                 adaptive_perturbation: bool = True,
                 min_perturbation: float = 0.001,
                 max_perturbation: float = 1.0,
                 seed: int = None,
                 verbose: bool = True):
        """
        Initialize the simulated annealing optimizer.

        Args:
            initial_temperature: Starting temperature
            final_temperature: Final temperature (stopping criterion)
            cooling_rate: Rate at which temperature decreases (0 < rate < 1)
            iterations_per_temp: Number of iterations at each temperature
            perturbation_scale: Scale of random perturbations to parameters
            adaptive_perturbation: Whether to adapt perturbation scale based on acceptance rate
            min_perturbation: Minimum perturbation scale (if adaptive)
            max_perturbation: Maximum perturbation scale (if adaptive)
            seed: Random seed for reproducibility
            verbose: Whether to print progress
        """
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.cooling_rate = cooling_rate
        self.iterations_per_temp = iterations_per_temp
        self.perturbation_scale = perturbation_scale
        self.adaptive_perturbation = adaptive_perturbation
        self.min_perturbation = min_perturbation
        self.max_perturbation = max_perturbation
        self.verbose = verbose

        if seed is not None:
            np.random.seed(seed)

        # Tracking
        self.history = {
            'cost': [],
            'best_cost': [],
            'temperature': [],
            'acceptance_rate': [],
            'perturbation_scale': []
        }

    def _perturb_parameters(self, params: np.ndarray) -> np.ndarray:
        """
        Generate a neighboring solution by perturbing parameters.

        Args:
            params: Current parameter vector

        Returns:
            Perturbed parameter vector
        """
        perturbation = np.random.randn(*params.shape) * self.perturbation_scale
        return params + perturbation

    def _acceptance_probability(self, current_cost: float, new_cost: float,
                                temperature: float) -> float:
        """
        Calculate probability of accepting a new solution.

        Args:
            current_cost: Cost of current solution
            new_cost: Cost of new solution
            temperature: Current temperature

        Returns:
            Acceptance probability (0 to 1)
        """
        if new_cost < current_cost:
            return 1.0
        else:
            # Boltzmann probability
            delta = new_cost - current_cost
            return np.exp(-delta / temperature)

    def _update_perturbation_scale(self, acceptance_rate: float):
        """
        Adapt perturbation scale based on acceptance rate.

        Target acceptance rate is around 0.44 (optimal for many problems).
        """
        if not self.adaptive_perturbation:
            return

        target_acceptance = 0.44

        if acceptance_rate > target_acceptance:
            # Too many acceptances, increase perturbation
            self.perturbation_scale *= 1.05
        else:
            # Too few acceptances, decrease perturbation
            self.perturbation_scale *= 0.95

        # Clip to bounds
        self.perturbation_scale = np.clip(self.perturbation_scale,
                                          self.min_perturbation,
                                          self.max_perturbation)

    def optimize(self,
                 objective_fn: Callable[[np.ndarray], float],
                 initial_params: np.ndarray,
                 callback: Optional[Callable] = None) -> Tuple[np.ndarray, float]:
        """
        Run simulated annealing optimization.

        Args:
            objective_fn: Function that takes parameters and returns cost to minimize
            initial_params: Initial parameter vector
            callback: Optional callback function called after each temperature step
                     Signature: callback(iteration, temperature, current_cost, best_cost)

        Returns:
            Tuple of (best_parameters, best_cost)
        """
        # Initialize
        current_params = initial_params.copy()
        current_cost = objective_fn(current_params)

        best_params = current_params.copy()
        best_cost = current_cost

        temperature = self.initial_temperature
        iteration = 0
        start_time = time.time()

        if self.verbose:
            print(f"Starting simulated annealing optimization")
            print(f"Initial cost: {current_cost:.6f}")
            print(f"Parameters: {len(initial_params)}")
            print("-" * 70)

        # Main optimization loop
        while temperature > self.final_temperature:
            acceptances = 0

            # Run multiple iterations at this temperature
            for _ in range(self.iterations_per_temp):
                iteration += 1

                # Generate neighbor
                new_params = self._perturb_parameters(current_params)
                new_cost = objective_fn(new_params)

                # Decide whether to accept
                accept_prob = self._acceptance_probability(current_cost, new_cost, temperature)

                if np.random.rand() < accept_prob:
                    current_params = new_params
                    current_cost = new_cost
                    acceptances += 1

                    # Update best solution
                    if current_cost < best_cost:
                        best_params = current_params.copy()
                        best_cost = current_cost

                # Track history
                self.history['cost'].append(current_cost)
                self.history['best_cost'].append(best_cost)
                self.history['temperature'].append(temperature)

            # Calculate acceptance rate for this temperature
            acceptance_rate = acceptances / self.iterations_per_temp
            self.history['acceptance_rate'].append(acceptance_rate)
            self.history['perturbation_scale'].append(self.perturbation_scale)

            # Adapt perturbation scale
            self._update_perturbation_scale(acceptance_rate)

            # Print progress
            if self.verbose:
                elapsed = time.time() - start_time
                print(f"Iter {iteration:5d} | Temp {temperature:.4f} | "
                      f"Cost {current_cost:.6f} | Best {best_cost:.6f} | "
                      f"Accept {acceptance_rate:.2%} | "
                      f"Perturb {self.perturbation_scale:.4f} | "
                      f"Time {elapsed:.1f}s")

            # Call callback
            if callback is not None:
                callback(iteration, temperature, current_cost, best_cost)

            # Cool down
            temperature *= self.cooling_rate

        if self.verbose:
            elapsed = time.time() - start_time
            print("-" * 70)
            print(f"Optimization complete in {elapsed:.1f}s")
            print(f"Final best cost: {best_cost:.6f}")
            print(f"Total iterations: {iteration}")

        return best_params, best_cost

    def get_history(self) -> Dict[str, List]:
        """Return optimization history."""
        return self.history


def train_neural_network(network, X_train, y_train, X_val=None, y_val=None,
                         loss_type='mse', **sa_kwargs):
    """
    Train a neural network using simulated annealing.

    Args:
        network: NeuralNetwork instance to train
        X_train: Training input data
        y_train: Training target data
        X_val: Validation input data (optional)
        y_val: Validation target data (optional)
        loss_type: Loss function to use ('mse', 'mae', 'binary_crossentropy')
        **sa_kwargs: Additional arguments for SimulatedAnnealing

    Returns:
        Tuple of (trained_network, optimizer, training_history)
    """
    # Create optimizer
    sa_optimizer = SimulatedAnnealing(**sa_kwargs)

    # Define objective function
    def objective(params):
        network.set_parameters(params)
        return network.compute_loss(X_train, y_train, loss_type=loss_type)

    # Track validation loss if provided
    val_losses = []

    def callback(iteration, temperature, current_cost, best_cost):
        if X_val is not None and y_val is not None:
            network.set_parameters(sa_optimizer.history['best_cost'][-1] * 0 +
                                  sa_optimizer.get_history()['cost'][-1] * 0)
            # Need to track the best params, let's do it differently
            pass

    # Get initial parameters
    initial_params = network.get_parameters()

    # Run optimization
    best_params, best_cost = sa_optimizer.optimize(objective, initial_params, callback=None)

    # Set network to best parameters
    network.set_parameters(best_params)

    # Build training history
    history = sa_optimizer.get_history()

    return network, sa_optimizer, history
