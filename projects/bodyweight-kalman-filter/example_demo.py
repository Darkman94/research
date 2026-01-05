#!/usr/bin/env python3
"""
Demo script showing Kalman filter with synthetic bodyweight data.

This script generates realistic bodyweight measurements and demonstrates
the Kalman filter's ability to extract the true weight signal from noisy data.
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from kalman_filter import BodyweightKalmanFilter


def generate_synthetic_data(days=90, true_weight_start=75.0, trend=-0.05,
                           daily_noise_std=0.7, seed=42):
    """
    Generate synthetic bodyweight data.

    Args:
        days: Number of days to generate
        true_weight_start: Starting true weight (kg)
        trend: Daily weight change (kg/day)
        daily_noise_std: Standard deviation of daily fluctuations (kg)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (dates, true_weights, measurements)
    """
    np.random.seed(seed)

    dates = [datetime.now() - timedelta(days=days-i-1) for i in range(days)]

    # True underlying weight (with gradual trend)
    true_weights = true_weight_start + trend * np.arange(days)

    # Add realistic daily fluctuations
    # Includes both random noise and some autocorrelation (yesterday's noise affects today)
    noise = np.random.normal(0, daily_noise_std, days)
    autocorrelated_noise = np.zeros(days)
    autocorrelated_noise[0] = noise[0]
    for i in range(1, days):
        # 30% carryover from previous day (simulating water retention patterns)
        autocorrelated_noise[i] = 0.3 * autocorrelated_noise[i-1] + 0.7 * noise[i]

    measurements = true_weights + autocorrelated_noise

    return dates, true_weights, measurements


def plot_comparison(dates, true_weights, measurements, filtered_weights):
    """Plot the true weights, measurements, and filtered estimates."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Top plot: All three signals
    axes[0].plot(dates, measurements, 'o', alpha=0.4, markersize=4,
                label='Noisy measurements', color='lightblue')
    axes[0].plot(dates, true_weights, '-', linewidth=2,
                label='True weight (unknown)', color='green')
    axes[0].plot(dates, filtered_weights, '-', linewidth=2,
                label='Kalman filtered estimate', color='darkblue')
    axes[0].set_ylabel('Weight (kg)')
    axes[0].set_title('Kalman Filter Performance on Synthetic Bodyweight Data')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Bottom plot: Errors
    measurement_error = measurements - true_weights
    filter_error = filtered_weights - true_weights

    axes[1].plot(dates, measurement_error, 'o', alpha=0.4, markersize=4,
                label='Measurement error', color='lightcoral')
    axes[1].plot(dates, filter_error, '-', linewidth=2,
                label='Filter error', color='darkred')
    axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Error (kg)')
    axes[1].set_title('Estimation Errors')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def main():
    """Run the demo."""
    print("="*70)
    print("BODYWEIGHT KALMAN FILTER DEMONSTRATION")
    print("="*70)
    print("\nGenerating synthetic bodyweight data...")

    # Generate data: 90 days, starting at 75kg, losing ~0.35 kg/week
    dates, true_weights, measurements = generate_synthetic_data(
        days=90,
        true_weight_start=75.0,
        trend=-0.05,  # -0.05 kg/day = -0.35 kg/week
        daily_noise_std=0.7
    )

    print(f"  Days: {len(dates)}")
    print(f"  True weight: {true_weights[0]:.1f} kg → {true_weights[-1]:.1f} kg")
    print(f"  True change: {true_weights[-1] - true_weights[0]:.2f} kg")
    print(f"  Measurement noise: ±{0.7:.1f} kg std")

    # Apply Kalman filter
    print("\nApplying Kalman filter...")
    kf = BodyweightKalmanFilter(
        process_variance=1e-5,
        measurement_variance=0.5
    )

    filtered_weights = kf.batch_filter(measurements)

    # Calculate performance metrics
    measurement_rmse = np.sqrt(np.mean((measurements - true_weights)**2))
    filter_rmse = np.sqrt(np.mean((filtered_weights - true_weights)**2))
    improvement = (1 - filter_rmse / measurement_rmse) * 100

    print("\nResults:")
    print(f"  Raw measurements RMSE:  {measurement_rmse:.3f} kg")
    print(f"  Filtered estimate RMSE: {filter_rmse:.3f} kg")
    print(f"  Improvement: {improvement:.1f}%")

    # Estimate weight change from filtered data
    estimated_change = filtered_weights[-1] - filtered_weights[0]
    true_change = true_weights[-1] - true_weights[0]
    change_error = estimated_change - true_change

    print(f"\n  True weight change:      {true_change:.2f} kg")
    print(f"  Estimated weight change: {estimated_change:.2f} kg")
    print(f"  Error: {change_error:.2f} kg ({abs(change_error/true_change)*100:.1f}%)")

    # Latest estimate
    latest_measurement = measurements[-1]
    latest_estimate = filtered_weights[-1]
    latest_true = true_weights[-1]

    print(f"\nLatest values:")
    print(f"  Measurement: {latest_measurement:.2f} kg (error: {latest_measurement - latest_true:+.2f} kg)")
    print(f"  Estimate:    {latest_estimate:.2f} kg (error: {latest_estimate - latest_true:+.2f} kg)")
    print(f"  True weight: {latest_true:.2f} kg")

    print("\n" + "="*70)
    print("Generating visualization...")

    # Create plot
    fig = plot_comparison(dates, true_weights, measurements, filtered_weights)

    # Save plot
    output_path = 'kalman_filter_demo.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

    print("\nDemo complete! The Kalman filter successfully extracted the true")
    print("weight signal from noisy measurements.")
    print("="*70)


if __name__ == '__main__':
    main()
