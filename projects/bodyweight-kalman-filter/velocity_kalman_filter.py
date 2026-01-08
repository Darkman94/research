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

    Process model:
    - weight(t) = weight(t-1) + velocity(t-1)*dt + integrated_acceleration_noise
    - velocity(t) = velocity(t-1) + acceleration_noise
    - noise(t) = ρ*noise(t-1) + innovation

    Uses proper integrated white noise acceleration for (weight, velocity) coupling,
    not independent random walks. This avoids redundancy and improves estimates.
    """

    def __init__(self,
                 autocorrelation: float = 0.7,
                 acceleration_variance: float = 1e-5,
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
            acceleration_variance: σ_a², white noise acceleration variance (kg²/day⁴)
                                  Controls how bendable the weight trend is
            noise_variance: Variance of AR(1) innovations σ_u² (kg²)
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
        self.acceleration_variance = acceleration_variance
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
        # velocity(t) = velocity(t-1)  (integrated from acceleration noise)
        # noise(t) = ρ * noise(t-1)
        self.F = np.array([
            [1.0, self.dt, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, self.rho]
        ])

        # Process noise covariance Q
        # For (weight, velocity) block: integrated white noise acceleration
        # Q_wv = σ_a² * [[dt⁴/4, dt³/2], [dt³/2, dt²]]
        # This properly couples weight and velocity uncertainty
        dt2 = self.dt * self.dt
        dt3 = dt2 * self.dt
        dt4 = dt3 * self.dt

        self.Q = np.array([
            [dt4/4 * self.acceleration_variance,  dt3/2 * self.acceleration_variance,  0],
            [dt3/2 * self.acceleration_variance,  dt2 * self.acceleration_variance,    0],
            [0,                                   0,                                    self.noise_variance]
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


def auto_tune_velocity_filter(measurements: np.ndarray, dt: float = 1.0) -> VelocityBodyweightKalmanFilter:
    """
    Automatically tune velocity filter parameters from data.

    Uses robust methods:
    - ρ, σ_u²: AR(1) regression on detrended residuals (not differencing)
    - σ_a²: Empirical Bayes from innovation variance in first-pass filter
    - R: Fixed at 0.01 kg² (0.1 kg std for typical scales)

    This avoids fragile smoothing heuristics and hard clamps that fail
    across diet phases.

    Args:
        measurements: Array of bodyweight measurements
        dt: Time step (days), default 1.0

    Returns:
        Configured VelocityBodyweightKalmanFilter
    """
    from scipy.ndimage import uniform_filter1d

    if len(measurements) < 14:
        print("  Warning: Less than 14 measurements, using conservative defaults")
        return VelocityBodyweightKalmanFilter(
            autocorrelation=0.7,
            acceleration_variance=1e-5,
            noise_variance=0.3,
            measurement_noise=0.01,
            dt=dt,
            initial_weight=measurements[0],
            initial_velocity=0.0,
            initial_noise=0.0
        )

    # Step 1: Detrend to get residuals
    # Use linear regression to remove trend (preserves autocorrelation better than MA)
    t = np.arange(len(measurements))
    # Fit linear trend: y = a + b*t
    A = np.vstack([np.ones(len(t)), t]).T
    coeffs = np.linalg.lstsq(A, measurements, rcond=None)[0]
    trend = coeffs[0] + coeffs[1] * t
    residuals = measurements - trend

    # Step 2: Fit AR(1) by regression: u_t = ρ*u_{t-1} + e_t
    # This is more robust than correlation + differencing
    u_t = residuals[1:]  # t = 2..N
    u_tm1 = residuals[:-1]  # t-1 = 1..N-1

    # OLS: ρ = (u_t' * u_{t-1}) / (u_{t-1}' * u_{t-1})
    numerator = np.sum(u_t * u_tm1)
    denominator = np.sum(u_tm1 * u_tm1)

    if denominator < 1e-10:
        print("  Warning: No variance in residuals, using ρ=0.0")
        rho = 0.0
        noise_var = 0.3
    else:
        rho = numerator / denominator

        # Sanity checks
        if rho < -0.3:
            print(f"  Warning: Negative AR(1) coefficient ({rho:.3f}) - may indicate issues")
        if rho > 0.98:
            print(f"  Warning: Very high AR(1) coefficient ({rho:.3f}), clamping to 0.95")
            rho = 0.95
        elif rho < 0:
            rho = max(rho, 0.0)  # Clamp negative to 0

        # Innovation variance: σ_u² = Var(u_t - ρ*u_{t-1})
        innovations = u_t - rho * u_tm1
        noise_var = np.var(innovations)

        # Sanity check
        if noise_var < 0.01:
            print(f"  Warning: Estimated AR(1) innovation variance very small ({noise_var:.4f}), using 0.1")
            noise_var = 0.1

    # Step 3: Tune acceleration variance via empirical Bayes
    # Run a first-pass filter with conservative σ_a², then use innovation variance
    print("  Running first-pass filter for acceleration tuning...")

    # Conservative initial guess
    accel_var_init = 1e-5
    kf_init = VelocityBodyweightKalmanFilter(
        autocorrelation=rho,
        acceleration_variance=accel_var_init,
        noise_variance=noise_var,
        measurement_noise=0.01,
        dt=dt,
        initial_weight=measurements[0],
        initial_velocity=0.0,
        initial_noise=0.0
    )

    # Collect innovations from first pass
    innovations_list = []
    innovation_vars = []

    for i, meas in enumerate(measurements):
        if kf_init.state is None:
            kf_init.update(meas)
            continue

        # Predict
        state_pred = kf_init.F @ kf_init.state
        P_pred = kf_init.F @ kf_init.P @ kf_init.F.T + kf_init.Q

        # Innovation
        y = meas - (kf_init.H @ state_pred)[0]
        S = (kf_init.H @ P_pred @ kf_init.H.T + kf_init.R)[0, 0]

        innovations_list.append(y)
        innovation_vars.append(S)

        # Update
        kf_init.update(meas)

    # Estimate acceleration variance from innovation statistics
    # If innovations are larger than predicted by S, increase σ_a²
    innovations_arr = np.array(innovations_list[5:])  # Skip first few (transient)
    predicted_std = np.sqrt(np.mean(innovation_vars[5:]))
    empirical_std = np.std(innovations_arr)

    # Ratio tells us if we need more/less process noise
    ratio = (empirical_std / predicted_std) ** 2

    # Adjust acceleration variance
    # If ratio > 1: innovations larger than expected → increase σ_a²
    # If ratio < 1: innovations smaller than expected → decrease σ_a²
    accel_var = accel_var_init * ratio

    # Sanity bounds (wider than before to allow adaptation across phases)
    accel_var = np.clip(accel_var, 1e-7, 0.1)

    # Measurement noise (fixed)
    measurement_noise = 0.01  # 0.1 kg std

    print(f"Auto-tuned parameters (robust methods):")
    print(f"  Autocorrelation (ρ): {rho:.3f} [AR(1) regression]")
    print(f"  AR(1) innovation variance (σ_u²): {noise_var:.3f} kg² [regression residuals]")
    print(f"  Acceleration variance (σ_a²): {accel_var:.6f} (kg/day²)² [empirical Bayes]")
    print(f"  Measurement noise (R): {measurement_noise:.4f} kg² [fixed]")

    return VelocityBodyweightKalmanFilter(
        autocorrelation=rho,
        acceleration_variance=accel_var,
        noise_variance=noise_var,
        measurement_noise=measurement_noise,
        dt=dt,
        initial_weight=measurements[0],
        initial_velocity=0.0,
        initial_noise=0.0
    )
