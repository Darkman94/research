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
                 initial_estimate: Optional[float] = None,
                 initial_error: float = 1.0,
                 initial_noise: float = 0.0,
                 initial_noise_error: float = 1.0):
        """
        Initialize the augmented Kalman filter.

        Args:
            autocorrelation: ρ, persistence of noise day-to-day (0-1)
                           0 = iid noise, 0.5 = moderate, 0.7-0.9 = typical for bodyweight
            process_variance: How much true weight changes per day (kg²)
            noise_variance: Variance of new noise each day (kg²)
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

        # Measurement noise variance (should be small since most variance is in state)
        self.R = np.array([[1e-6]])

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

        # Update covariance
        I_KH = np.eye(2) - K @ self.H
        self.P = I_KH @ P_pred

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


def estimate_autocorrelation(measurements: np.ndarray,
                             initial_true_weight: Optional[float] = None,
                             max_lag: int = 7) -> float:
    """
    Estimate the autocorrelation coefficient from raw measurements.

    This uses a simple approach:
    1. Smooth measurements to estimate true weight trend
    2. Calculate residuals
    3. Compute lag-1 autocorrelation of residuals

    Args:
        measurements: Array of bodyweight measurements
        initial_true_weight: Optional initial weight estimate
        max_lag: Maximum lag to consider (default 7 days)

    Returns:
        Estimated autocorrelation coefficient (ρ)
    """
    if len(measurements) < 10:
        return 0.7  # Default assumption

    # Simple detrending: use 7-day moving average as estimate of true weight
    from scipy.ndimage import uniform_filter1d

    # Pad for edge effects
    padded = np.pad(measurements, (3, 3), mode='edge')
    smoothed = uniform_filter1d(padded, size=7)[3:-3]

    # Residuals (measurement - trend)
    residuals = measurements - smoothed

    # Compute autocorrelation at lag 1
    n = len(residuals)
    mean = np.mean(residuals)
    var = np.var(residuals)

    if var < 1e-6:
        return 0.0

    # Lag-1 autocorrelation
    autocorr = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]

    # Clip to valid range
    autocorr = np.clip(autocorr, 0.0, 0.95)

    return autocorr


def auto_tune_filter(measurements: np.ndarray) -> AugmentedBodyweightKalmanFilter:
    """
    Automatically tune filter parameters from data.

    Args:
        measurements: Array of bodyweight measurements

    Returns:
        Configured AugmentedBodyweightKalmanFilter
    """
    # Estimate autocorrelation
    rho = estimate_autocorrelation(measurements)

    # Estimate measurement variance
    # Use variance of day-to-day changes as proxy
    daily_changes = np.diff(measurements)
    noise_var = np.var(daily_changes) / 2  # Divide by 2 for AR(1) process

    # Process variance should be much smaller (true weight changes slowly)
    process_var = noise_var / 100

    print(f"Auto-tuned parameters:")
    print(f"  Autocorrelation (ρ): {rho:.3f}")
    print(f"  Noise variance: {noise_var:.3f} kg²")
    print(f"  Process variance: {process_var:.6f} kg²")

    return AugmentedBodyweightKalmanFilter(
        autocorrelation=rho,
        process_variance=process_var,
        noise_variance=noise_var,
        initial_estimate=measurements[0],
        initial_noise=0.0
    )
