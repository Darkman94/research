"""
Velocity-augmented Kalman Filter for bodyweight tracking.

This module implements a 3D Kalman filter that models:
1. True bodyweight (changing state)
2. Weight velocity/trend (kg/day - handles diets/bulks)
3. Autocorrelated measurement noise (water retention, glycogen, etc.)

State vector: [weight, velocity, noise]
"""

import numpy as np
from typing import Optional, Tuple


class VelocityBodyweightKalmanFilter:
    """
    A velocity-augmented Kalman filter for bodyweight tracking.

    State vector: [weight, velocity, noise]
    - weight: True bodyweight (kg)
    - velocity: Rate of weight change (kg/day)
    - noise: Autocorrelated fluctuation (water retention, etc.)

    This model handles:
    - Diet/bulk trends (via velocity state)
    - Multi-day water retention patterns (via AR(1) noise)
    - Scale measurement errors
    """

    def __init__(self,
                 autocorrelation: float = 0.7,
                 weight_process_variance: float = 1e-6,
                 velocity_process_variance: float = 1e-5,
                 noise_variance: float = 0.3,
                 measurement_noise: float = 0.01,
                 dt: float = 1.0,
                 initial_weight: Optional[float] = None,
                 initial_velocity: float = 0.0,
                 initial_noise: float = 0.0,
                 initial_weight_error: float = 1.0,
                 initial_velocity_error: float = 0.01,
                 initial_noise_error: float = 1.0):
        """
        Initialize the velocity-augmented Kalman filter.

        Args:
            autocorrelation: ρ, persistence of noise day-to-day (typical: 0.7-0.9)
            weight_process_variance: Process noise for weight (kg²)
                                    Very small since weight follows velocity
            velocity_process_variance: Process noise for velocity (kg²/day²)
                                      Controls how quickly trend can change
            noise_variance: Variance of AR(1) innovations σ_v² (kg²)
            measurement_noise: Direct measurement error from scale (kg²)
            dt: Time step (days), default 1.0
            initial_weight: Initial weight estimate (kg)
            initial_velocity: Initial velocity estimate (kg/day), default 0
            initial_noise: Initial noise level estimate (kg)
            initial_weight_error: Initial weight error variance (kg²)
            initial_velocity_error: Initial velocity error variance (kg²/day²)
            initial_noise_error: Initial noise error variance (kg²)
        """
        self.rho = autocorrelation
        self.dt = dt
        self.weight_process_variance = weight_process_variance
        self.velocity_process_variance = velocity_process_variance
        self.noise_variance = noise_variance

        # State: [weight, velocity, noise]
        if initial_weight is not None:
            self.state = np.array([initial_weight, initial_velocity, initial_noise])
        else:
            self.state = None

        # Error covariance matrix (3x3)
        self.P = np.array([
            [initial_weight_error, 0, 0],
            [0, initial_velocity_error, 0],
            [0, 0, initial_noise_error]
        ])

        # State transition matrix F
        # weight(t) = weight(t-1) + velocity(t-1) * dt
        # velocity(t) = velocity(t-1)  (random walk)
        # noise(t) = ρ * noise(t-1)
        self.F = np.array([
            [1.0, self.dt, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, self.rho]
        ])

        # Process noise covariance Q
        self.Q = np.array([
            [self.weight_process_variance, 0, 0],
            [0, self.velocity_process_variance, 0],
            [0, 0, self.noise_variance]
        ])

        # Measurement matrix H: measurement = weight + noise
        self.H = np.array([[1.0, 0.0, 1.0]])

        # Measurement noise variance R
        self.R = np.array([[measurement_noise]])

        # History
        self.measurements = []
        self.weight_estimates = []
        self.velocity_estimates = []
        self.noise_estimates = []
        self.weight_errors = []
        self.velocity_errors = []
        self.noise_errors = []

    def update(self, measurement: float) -> Tuple[float, float, float]:
        """
        Update the filter with a new measurement.

        Args:
            measurement: New bodyweight measurement (kg)

        Returns:
            Tuple of (estimated_weight, estimated_velocity, estimated_noise)
        """
        # Initialize state with first measurement
        if self.state is None:
            self.state = np.array([measurement, 0.0, 0.0])
            self.measurements.append(measurement)
            self.weight_estimates.append(measurement)
            self.velocity_estimates.append(0.0)
            self.noise_estimates.append(0.0)
            self.weight_errors.append(np.sqrt(self.P[0, 0]))
            self.velocity_errors.append(np.sqrt(self.P[1, 1]))
            self.noise_errors.append(np.sqrt(self.P[2, 2]))
            return measurement, 0.0, 0.0

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
        I_KH = np.eye(3) - K @ self.H
        self.P = I_KH @ P_pred @ I_KH.T + K @ self.R @ K.T

        # Store history
        self.measurements.append(measurement)
        self.weight_estimates.append(self.state[0])
        self.velocity_estimates.append(self.state[1])
        self.noise_estimates.append(self.state[2])
        self.weight_errors.append(np.sqrt(self.P[0, 0]))
        self.velocity_errors.append(np.sqrt(self.P[1, 1]))
        self.noise_errors.append(np.sqrt(self.P[2, 2]))

        return self.state[0], self.state[1], self.state[2]

    def batch_filter(self, measurements: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Process a batch of measurements.

        Args:
            measurements: Array of bodyweight measurements (kg)

        Returns:
            Tuple of (weights, velocities, noise_levels)
        """
        weights = []
        velocities = []
        noise_levels = []

        for measurement in measurements:
            weight, velocity, noise = self.update(measurement)
            weights.append(weight)
            velocities.append(velocity)
            noise_levels.append(noise)

        return np.array(weights), np.array(velocities), np.array(noise_levels)

    def predict_future(self, days_ahead: int = 7) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict future measurements based on current state.

        Args:
            days_ahead: Number of days to predict forward

        Returns:
            Tuple of (predicted_measurements, predicted_weights, predicted_uncertainties)
        """
        if self.state is None:
            return None, None, None

        predictions = []
        weights = []
        uncertainties = []

        state_pred = self.state.copy()
        P_pred = self.P.copy()

        for _ in range(days_ahead):
            state_pred = self.F @ state_pred
            P_pred = self.F @ P_pred @ self.F.T + self.Q

            # Predicted measurement
            pred_measurement = (self.H @ state_pred)[0]
            pred_variance = (self.H @ P_pred @ self.H.T + self.R)[0, 0]

            predictions.append(pred_measurement)
            weights.append(state_pred[0])
            uncertainties.append(np.sqrt(pred_variance))

        return np.array(predictions), np.array(weights), np.array(uncertainties)

    def get_results(self):
        """
        Get the full history of estimates.

        Returns:
            Dictionary containing:
                - measurements: Raw measurements
                - weight: Estimated true weight
                - velocity: Estimated weight velocity (kg/day)
                - noise: Estimated noise (water retention, etc.)
                - weight_std: Standard error of weight
                - velocity_std: Standard error of velocity
                - noise_std: Standard error of noise estimate
        """
        return {
            'measurements': np.array(self.measurements),
            'weight': np.array(self.weight_estimates),
            'velocity': np.array(self.velocity_estimates),
            'noise': np.array(self.noise_estimates),
            'weight_std': np.array(self.weight_errors),
            'velocity_std': np.array(self.velocity_errors),
            'noise_std': np.array(self.noise_errors)
        }

    def get_current_estimate(self) -> Tuple[float, float, float, float, float, float]:
        """
        Get current estimates.

        Returns:
            Tuple of (weight, weight_std, velocity, velocity_std, noise, noise_std)
        """
        if self.state is None:
            return None, None, None, None, None, None

        return (self.state[0], np.sqrt(self.P[0, 0]),
                self.state[1], np.sqrt(self.P[1, 1]),
                self.state[2], np.sqrt(self.P[2, 2]))


def auto_tune_velocity_filter(measurements: np.ndarray) -> VelocityBodyweightKalmanFilter:
    """
    Automatically tune velocity filter parameters from data.

    Estimates:
    - ρ: autocorrelation coefficient
    - σ_v²: AR(1) innovation variance
    - velocity process variance: from observed weight changes

    Args:
        measurements: Array of bodyweight measurements

    Returns:
        Configured VelocityBodyweightKalmanFilter
    """
    from scipy.ndimage import uniform_filter1d

    # Estimate autocorrelation
    if len(measurements) < 10:
        print("  Warning: Less than 10 measurements, using default ρ=0.7")
        rho = 0.7
    else:
        # Detrend
        padded = np.pad(measurements, (3, 3), mode='edge')
        smoothed = uniform_filter1d(padded, size=7)[3:-3]
        residuals = measurements - smoothed

        # Compute autocorrelation
        var = np.var(residuals)
        if var < 1e-6:
            print("  Warning: No variance in residuals, using ρ=0.0")
            rho = 0.0
        else:
            rho = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
            if rho < -0.3:
                print(f"  Warning: Negative autocorrelation ({rho:.3f})")
            elif rho > 0.95:
                print(f"  Warning: Very high autocorrelation ({rho:.3f}), clamping to 0.95")
                rho = 0.95

    # Estimate AR(1) innovation variance
    if len(measurements) >= 10:
        padded = np.pad(measurements, (3, 3), mode='edge')
        smoothed = uniform_filter1d(padded, size=7)[3:-3]
        residuals = measurements - smoothed
        residual_diffs = np.diff(residuals)
        var_diffs = np.var(residual_diffs)
        noise_var = var_diffs * (1 + rho) / 2

        if noise_var < 0.01:
            print(f"  Warning: Estimated noise variance very small ({noise_var:.4f}), using 0.1")
            noise_var = 0.1
    else:
        noise_var = 0.3

    # Estimate velocity process variance from observed weight changes
    # Use detrended smoothed data to estimate typical velocity changes
    if len(measurements) >= 14:
        # Smooth more heavily for velocity estimation
        padded = np.pad(measurements, (7, 7), mode='edge')
        heavy_smooth = uniform_filter1d(padded, size=14)[7:-7]

        # Estimate velocities from smoothed data
        velocities = np.diff(heavy_smooth)
        velocity_changes = np.diff(velocities)
        velocity_var = np.var(velocity_changes)

        # Clamp to reasonable range
        velocity_var = np.clip(velocity_var, 1e-6, 0.01)
    else:
        # Default: allows ~0.01 kg/day change in velocity
        velocity_var = 1e-5

    # Weight process variance (very small, weight follows velocity)
    weight_var = 1e-6

    # Measurement noise
    measurement_noise = 0.01  # 0.1 kg std

    print(f"Auto-tuned parameters:")
    print(f"  Autocorrelation (ρ): {rho:.3f}")
    print(f"  AR(1) innovation variance: {noise_var:.3f} kg²")
    print(f"  Velocity process variance: {velocity_var:.6f} (kg/day)²")
    print(f"  Weight process variance: {weight_var:.6f} kg²")
    print(f"  Measurement noise (R): {measurement_noise:.4f} kg²")

    return VelocityBodyweightKalmanFilter(
        autocorrelation=rho,
        weight_process_variance=weight_var,
        velocity_process_variance=velocity_var,
        noise_variance=noise_var,
        measurement_noise=measurement_noise,
        initial_weight=measurements[0],
        initial_velocity=0.0,
        initial_noise=0.0
    )
