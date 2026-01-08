"""
Augmented Kalman Filter for bodyweight tracking with autocorrelated noise.

This module implements a 2D Kalman filter that models both:
1. True bodyweight (slowly changing state)
2. Autocorrelated measurement noise (water retention, glycogen, etc.)

Unlike the simple Kalman filter, this handles multi-day fluctuation patterns
caused by water retention, DOMS, hormones, etc.
"""

import numpy as np
from typing import Optional, Tuple


class AugmentedBodyweightKalmanFilter:
    """
    An augmented Kalman filter that models autocorrelated measurement noise.

    State vector: [true_weight, noise_level]

    The noise_level represents the current water retention / temporary fluctuation,
    which persists across multiple days following an AR(1) process.
    """

    def __init__(self,
                 autocorrelation: float = 0.7,
                 process_variance: float = 1e-5,
                 noise_variance: float = 0.3,
                 measurement_noise: float = 0.01,
                 initial_estimate: Optional[float] = None,
                 initial_error: float = 1.0,
                 initial_noise: float = 0.0,
                 initial_noise_error: float = 1.0):
        """
        Initialize the augmented Kalman filter.

        Args:
            autocorrelation: ρ, persistence of noise day-to-day (typical: 0.7-0.9)
            process_variance: How much true weight changes per day (kg²)
            noise_variance: Variance of AR(1) innovations σ_v² (kg²)
            measurement_noise: Direct measurement error from scale (kg²), default 0.01
                             (0.1 kg std, protects against one-off bad measurements)
            initial_estimate: Initial true weight estimate (kg)
            initial_error: Initial true weight error (kg²)
            initial_noise: Initial noise level estimate (kg)
            initial_noise_error: Initial noise level error (kg²)
        """
        self.rho = autocorrelation
        self.process_variance = process_variance
        self.noise_variance = noise_variance

        # State: [true_weight, noise_level]
        if initial_estimate is not None:
            self.state = np.array([initial_estimate, initial_noise])
        else:
            self.state = None

        # Error covariance matrix (2x2)
        self.P = np.array([
            [initial_error, 0],
            [0, initial_noise_error]
        ])

        # State transition matrix
        self.F = np.array([
            [1.0, 0.0],      # true_weight(t) = true_weight(t-1) + process_noise
            [0.0, self.rho]  # noise(t) = rho * noise(t-1) + new_noise
        ])

        # Process noise covariance
        self.Q = np.array([
            [self.process_variance, 0],
            [0, self.noise_variance]
        ])

        # Measurement matrix: measurement = true_weight + noise
        self.H = np.array([[1.0, 1.0]])

        # Measurement noise variance (scale error, protocol variation)
        # Typical: 0.01 kg² = 0.1 kg std for good home scales
        self.R = np.array([[measurement_noise]])

        # History
        self.measurements = []
        self.true_weight_estimates = []
        self.noise_estimates = []
        self.true_weight_errors = []
        self.noise_errors = []

    def update(self, measurement: float) -> Tuple[float, float]:
        """
        Update the filter with a new measurement.

        Args:
            measurement: New bodyweight measurement (kg)

        Returns:
            Tuple of (estimated_true_weight, estimated_noise_level)
        """
        # Initialize state with first measurement
        if self.state is None:
            self.state = np.array([measurement, 0.0])
            self.measurements.append(measurement)
            self.true_weight_estimates.append(measurement)
            self.noise_estimates.append(0.0)
            self.true_weight_errors.append(np.sqrt(self.P[0, 0]))
            self.noise_errors.append(np.sqrt(self.P[1, 1]))
            return measurement, 0.0

        # Prediction step
        state_pred = self.F @ self.state
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # Update step
        # Innovation (measurement residual)
        y = measurement - (self.H @ state_pred)[0]

        # Innovation covariance
        S = (self.H @ P_pred @ self.H.T + self.R)[0, 0]

        # Kalman gain
        K = (P_pred @ self.H.T) / S

        # Update state
        self.state = state_pred + K.flatten() * y

        # Update covariance using Joseph form (maintains symmetry and PSD)
        I_KH = np.eye(2) - K @ self.H
        self.P = I_KH @ P_pred @ I_KH.T + K @ self.R @ K.T

        # Store history
        self.measurements.append(measurement)
        self.true_weight_estimates.append(self.state[0])
        self.noise_estimates.append(self.state[1])
        self.true_weight_errors.append(np.sqrt(self.P[0, 0]))
        self.noise_errors.append(np.sqrt(self.P[1, 1]))

        return self.state[0], self.state[1]

    def batch_filter(self, measurements: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a batch of measurements.

        Args:
            measurements: Array of bodyweight measurements (kg)

        Returns:
            Tuple of (true_weights, noise_levels)
        """
        true_weights = []
        noise_levels = []

        for measurement in measurements:
            true_weight, noise = self.update(measurement)
            true_weights.append(true_weight)
            noise_levels.append(noise)

        return np.array(true_weights), np.array(noise_levels)

    def predict_next(self, days_ahead: int = 1) -> Tuple[float, float, float]:
        """
        Predict future measurements.

        Args:
            days_ahead: Number of days to predict forward

        Returns:
            Tuple of (predicted_measurement, predicted_true_weight, predicted_uncertainty)
        """
        if self.state is None:
            return None, None, None

        state_pred = self.state.copy()
        P_pred = self.P.copy()

        for _ in range(days_ahead):
            state_pred = self.F @ state_pred
            P_pred = self.F @ P_pred @ self.F.T + self.Q

        # Predicted measurement
        pred_measurement = (self.H @ state_pred)[0]
        pred_variance = (self.H @ P_pred @ self.H.T + self.R)[0, 0]
        pred_std = np.sqrt(pred_variance)

        return pred_measurement, state_pred[0], pred_std

    def get_results(self):
        """
        Get the full history of estimates.

        Returns:
            Dictionary containing:
                - measurements: Raw measurements
                - true_weight: Estimated true weight
                - noise: Estimated noise (water retention, etc.)
                - true_weight_std: Standard error of true weight
                - noise_std: Standard error of noise estimate
        """
        return {
            'measurements': np.array(self.measurements),
            'true_weight': np.array(self.true_weight_estimates),
            'noise': np.array(self.noise_estimates),
            'true_weight_std': np.array(self.true_weight_errors),
            'noise_std': np.array(self.noise_errors)
        }

    def get_current_estimate(self) -> Tuple[float, float, float, float]:
        """
        Get current estimates.

        Returns:
            Tuple of (true_weight, true_weight_std, noise_level, noise_std)
        """
        if self.state is None:
            return None, None, None, None

        return (self.state[0], np.sqrt(self.P[0, 0]),
                self.state[1], np.sqrt(self.P[1, 1]))


def estimate_autocorrelation(measurements: np.ndarray) -> float:
    """
    Estimate the autocorrelation coefficient from raw measurements.

    This uses a simple approach:
    1. Smooth measurements to estimate true weight trend
    2. Calculate residuals
    3. Compute lag-1 autocorrelation of residuals

    Args:
        measurements: Array of bodyweight measurements

    Returns:
        Estimated autocorrelation coefficient (ρ)
    """
    if len(measurements) < 10:
        print("  Warning: Less than 10 measurements, using default ρ=0.7")
        return 0.7  # Default assumption

    # Simple detrending: use 7-day moving average as estimate of true weight
    from scipy.ndimage import uniform_filter1d

    # Pad for edge effects
    padded = np.pad(measurements, (3, 3), mode='edge')
    smoothed = uniform_filter1d(padded, size=7)[3:-3]

    # Residuals (measurement - trend)
    residuals = measurements - smoothed

    # Compute autocorrelation at lag 1
    var = np.var(residuals)

    if var < 1e-6:
        print("  Warning: No variance in residuals, using ρ=0.0")
        return 0.0

    # Lag-1 autocorrelation
    autocorr = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]

    # Warn if autocorrelation seems unrealistic
    if autocorr < -0.3:
        print(f"  Warning: Negative autocorrelation ({autocorr:.3f}) detected - may indicate issues")
    elif autocorr > 0.95:
        print(f"  Warning: Very high autocorrelation ({autocorr:.3f}) - clamping to 0.95")
        autocorr = 0.95

    return autocorr


def auto_tune_filter(measurements: np.ndarray) -> AugmentedBodyweightKalmanFilter:
    """
    Automatically tune filter parameters from data.

    Estimates parameters for AR(1) noise model:
    - ρ: autocorrelation coefficient (from lag-1 correlation of residuals)
    - σ_v²: innovation variance (from variance of differences and ρ)

    For AR(1): n_t = ρ*n_{t-1} + v_t where v_t ~ N(0, σ_v²)
    Var(n_t - n_{t-1}) = 2*σ_v²/(1+ρ)
    Therefore: σ_v² = Var(differences) * (1+ρ) / 2

    Args:
        measurements: Array of bodyweight measurements

    Returns:
        Configured AugmentedBodyweightKalmanFilter
    """
    # Estimate autocorrelation from detrended residuals
    rho = estimate_autocorrelation(measurements)

    # Estimate innovation variance for AR(1) process
    # First, detrend to get residuals
    from scipy.ndimage import uniform_filter1d
    padded = np.pad(measurements, (3, 3), mode='edge')
    smoothed = uniform_filter1d(padded, size=7)[3:-3]
    residuals = measurements - smoothed

    # Variance of residual differences
    residual_diffs = np.diff(residuals)
    var_diffs = np.var(residual_diffs)

    # For AR(1): Var(n_t - n_{t-1}) = 2*σ_v²/(1+ρ)
    # Therefore: σ_v² = Var(diffs) * (1+ρ) / 2
    noise_var = var_diffs * (1 + rho) / 2

    # Sanity check
    if noise_var < 0.01:
        print(f"  Warning: Estimated noise variance very small ({noise_var:.4f}), using 0.1")
        noise_var = 0.1

    # Process variance should be much smaller (true weight changes slowly)
    # Assume true weight changes ~0.1 kg/week = 0.014 kg/day std
    process_var = 0.0002  # (0.014 kg)²

    # Measurement noise from scale (typical: 0.05-0.15 kg std)
    measurement_noise = 0.01  # 0.1 kg std

    print(f"Auto-tuned parameters:")
    print(f"  Autocorrelation (ρ): {rho:.3f}")
    print(f"  Innovation variance (σ_v²): {noise_var:.3f} kg²")
    print(f"  Process variance: {process_var:.6f} kg²")
    print(f"  Measurement noise (R): {measurement_noise:.4f} kg²")

    return AugmentedBodyweightKalmanFilter(
        autocorrelation=rho,
        process_variance=process_var,
        noise_variance=noise_var,
        measurement_noise=measurement_noise,
        initial_estimate=measurements[0],
        initial_noise=0.0
    )
