"""
Kalman Filter implementation for bodyweight tracking.

This module implements a 1D Kalman filter to smooth bodyweight measurements
and estimate the "true" bodyweight by filtering out daily fluctuations.
"""

import numpy as np


class BodyweightKalmanFilter:
    """
    A Kalman filter for tracking bodyweight over time.

    The filter models bodyweight as having:
    - A slowly changing true weight (state)
    - Daily measurement noise (water retention, food, etc.)
    - Gradual weight changes over time (process noise)
    """

    def __init__(self, process_variance=1e-5, measurement_variance=0.5,
                 initial_estimate=None, initial_error=1.0):
        """
        Initialize the Kalman filter.

        Args:
            process_variance: How much the true weight can change per day (kg²)
                             Small value = weight changes slowly
            measurement_variance: Expected variance in daily measurements (kg²)
                                 Larger value = less trust in measurements
            initial_estimate: Initial weight estimate (kg). If None, uses first measurement
            initial_error: Initial estimation error (kg²)
        """
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance

        # State: estimated true weight
        self.state_estimate = initial_estimate
        self.error_estimate = initial_error

        # Store history for analysis
        self.measurements = []
        self.filtered_estimates = []
        self.error_estimates = []

    def update(self, measurement):
        """
        Update the filter with a new measurement.

        Args:
            measurement: New bodyweight measurement (kg)

        Returns:
            Filtered weight estimate (kg)
        """
        # Initialize state with first measurement if needed
        if self.state_estimate is None:
            self.state_estimate = measurement
            self.measurements.append(measurement)
            self.filtered_estimates.append(measurement)
            self.error_estimates.append(self.error_estimate)
            return measurement

        # Prediction step
        # Predict next state (weight doesn't change much day-to-day)
        predicted_state = self.state_estimate
        predicted_error = self.error_estimate + self.process_variance

        # Update step
        # Calculate Kalman gain (how much to trust new measurement vs prediction)
        kalman_gain = predicted_error / (predicted_error + self.measurement_variance)

        # Update estimate with measurement
        self.state_estimate = predicted_state + kalman_gain * (measurement - predicted_state)

        # Update error estimate
        self.error_estimate = (1 - kalman_gain) * predicted_error

        # Store history
        self.measurements.append(measurement)
        self.filtered_estimates.append(self.state_estimate)
        self.error_estimates.append(self.error_estimate)

        return self.state_estimate

    def batch_filter(self, measurements):
        """
        Process a batch of measurements.

        Args:
            measurements: List or array of bodyweight measurements (kg)

        Returns:
            Array of filtered weight estimates (kg)
        """
        filtered = []
        for measurement in measurements:
            filtered.append(self.update(measurement))
        return np.array(filtered)

    def get_results(self):
        """
        Get the full history of measurements and estimates.

        Returns:
            Dictionary containing:
                - measurements: Raw measurements
                - filtered: Filtered estimates
                - errors: Estimation errors (standard deviation)
        """
        return {
            'measurements': np.array(self.measurements),
            'filtered': np.array(self.filtered_estimates),
            'errors': np.sqrt(np.array(self.error_estimates))  # Convert variance to std
        }

    def get_current_estimate(self):
        """
        Get the current best estimate of true weight.

        Returns:
            Tuple of (estimate, standard_error)
        """
        if self.state_estimate is None:
            return None, None
        return self.state_estimate, np.sqrt(self.error_estimate)
