#!/usr/bin/env python3
"""
Comparison of simple vs augmented Kalman filters.

This script demonstrates the improvement from modeling autocorrelated noise.
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from kalman_filter import BodyweightKalmanFilter
from augmented_kalman_filter import AugmentedBodyweightKalmanFilter


def generate_autocorrelated_data(days=90, true_weight_start=75.0, trend=-0.05,
                                 noise_std=0.5, autocorr=0.75, seed=42):
    """
    Generate synthetic bodyweight data with autocorrelated noise.

    Args:
        days: Number of days
        true_weight_start: Starting true weight (kg)
        trend: Daily weight change (kg/day)
        noise_std: Standard deviation of noise innovations (kg)
        autocorr: Autocorrelation coefficient (0-1)
        seed: Random seed

    Returns:
        Tuple of (dates, true_weights, measurements, noise_process)
    """
    np.random.seed(seed)

    dates = [datetime.now() - timedelta(days=days-i-1) for i in range(days)]

    # True underlying weight
    true_weights = true_weight_start + trend * np.arange(days)

    # Generate AR(1) noise process
    # n_t = ρ * n_{t-1} + v_t, where v_t ~ N(0, σ²)
    noise = np.zeros(days)
    innovations = np.random.normal(0, noise_std, days)

    for i in range(days):
        if i == 0:
            noise[i] = innovations[i]
        else:
            noise[i] = autocorr * noise[i-1] + innovations[i]

    measurements = true_weights + noise

    return dates, true_weights, measurements, noise


def plot_filter_comparison(dates, true_weights, measurements,
                          simple_results, augmented_results, actual_noise):
    """Create comprehensive comparison plots."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Plot 1: Measurements and estimates
    ax = axes[0]
    ax.plot(dates, measurements, 'o', alpha=0.3, markersize=3,
           label='Measurements', color='lightblue')
    ax.plot(dates, true_weights, '-', linewidth=2,
           label='True weight (ground truth)', color='green')
    ax.plot(dates, simple_results['filtered'], '-', linewidth=2,
           label='Simple Kalman (iid assumption)', color='orange')
    ax.plot(dates, augmented_results['true_weight'], '-', linewidth=2,
           label='Augmented Kalman (AR model)', color='darkblue')

    ax.set_ylabel('Weight (kg)')
    ax.set_title('Filter Comparison: Simple vs Augmented Kalman Filter')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Plot 2: Errors in true weight estimation
    ax = axes[1]
    simple_error = simple_results['filtered'] - true_weights
    augmented_error = augmented_results['true_weight'] - true_weights

    ax.plot(dates, simple_error, '-', alpha=0.7, linewidth=1.5,
           label=f'Simple Kalman (RMSE: {np.sqrt(np.mean(simple_error**2)):.3f} kg)',
           color='orange')
    ax.plot(dates, augmented_error, '-', alpha=0.7, linewidth=1.5,
           label=f'Augmented Kalman (RMSE: {np.sqrt(np.mean(augmented_error**2)):.3f} kg)',
           color='darkblue')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)

    ax.set_ylabel('Estimation Error (kg)')
    ax.set_title('True Weight Estimation Error')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Plot 3: Noise/water retention estimates
    ax = axes[2]
    ax.plot(dates, actual_noise, '-', linewidth=2, alpha=0.7,
           label='Actual noise (ground truth)', color='green')
    ax.plot(dates, augmented_results['noise'], '-', linewidth=2,
           label='Augmented Kalman noise estimate', color='darkblue')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)

    # Fill between to show water retention
    ax.fill_between(dates, 0, augmented_results['noise'],
                    where=augmented_results['noise']>0,
                    alpha=0.3, color='red', label='Estimated water retention')
    ax.fill_between(dates, 0, augmented_results['noise'],
                    where=augmented_results['noise']<0,
                    alpha=0.3, color='blue', label='Estimated dehydration')

    ax.set_xlabel('Date')
    ax.set_ylabel('Noise Level (kg)')
    ax.set_title('Water Retention / Temporary Fluctuations')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def main():
    """Run the comparison."""
    print("="*80)
    print("KALMAN FILTER COMPARISON: Simple (iid) vs Augmented (AR model)")
    print("="*80)

    # Generate data with strong autocorrelation (typical for bodyweight)
    print("\nGenerating synthetic data with autocorrelated noise...")
    days = 90
    autocorr = 0.75
    noise_std = 0.5

    dates, true_weights, measurements, actual_noise = generate_autocorrelated_data(
        days=days,
        true_weight_start=75.0,
        trend=-0.05,  # Losing weight
        noise_std=noise_std,
        autocorr=autocorr
    )

    print(f"  Days: {days}")
    print(f"  True autocorrelation: {autocorr}")
    print(f"  Noise std: {noise_std} kg")
    print(f"  True weight: {true_weights[0]:.1f} → {true_weights[-1]:.1f} kg")

    # Apply simple Kalman filter (iid assumption)
    print("\nApplying simple Kalman filter (assumes iid noise)...")
    simple_kf = BodyweightKalmanFilter(
        process_variance=1e-5,
        measurement_variance=0.5
    )
    simple_kf.batch_filter(measurements)
    simple_results = simple_kf.get_results()

    # Apply augmented Kalman filter (AR model)
    print("Applying augmented Kalman filter (models autocorrelation)...")
    # For AR(1): stationary variance = σ_v²/(1-ρ²), so σ_v² = σ²(1-ρ²)
    innovation_var = noise_std**2 * (1 - autocorr**2)
    augmented_kf = AugmentedBodyweightKalmanFilter(
        autocorrelation=autocorr,
        process_variance=1e-5,
        noise_variance=innovation_var,
        measurement_noise=0.01  # 0.1 kg scale error
    )
    augmented_kf.batch_filter(measurements)
    augmented_results = augmented_kf.get_results()

    # Calculate performance metrics
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON")
    print("="*80)

    # True weight estimation error
    simple_error = simple_results['filtered'] - true_weights
    augmented_error = augmented_results['true_weight'] - true_weights

    simple_rmse = np.sqrt(np.mean(simple_error**2))
    augmented_rmse = np.sqrt(np.mean(augmented_error**2))
    improvement = (1 - augmented_rmse / simple_rmse) * 100

    print("\nTrue Weight Estimation:")
    print(f"  Simple Kalman RMSE:    {simple_rmse:.4f} kg")
    print(f"  Augmented Kalman RMSE: {augmented_rmse:.4f} kg")
    print(f"  Improvement: {improvement:.1f}%")

    # Noise estimation (only augmented filter estimates this)
    noise_error = augmented_results['noise'] - actual_noise
    noise_rmse = np.sqrt(np.mean(noise_error**2))
    noise_corr = np.corrcoef(augmented_results['noise'], actual_noise)[0, 1]

    print("\nNoise/Water Retention Estimation (Augmented Only):")
    print(f"  RMSE: {noise_rmse:.4f} kg")
    print(f"  Correlation with actual: {noise_corr:.3f}")

    # Current state
    print("\nCurrent Estimate (Latest Day):")
    print(f"  Measurement: {measurements[-1]:.2f} kg")
    print(f"  True weight: {true_weights[-1]:.2f} kg")
    print(f"  Simple estimate: {simple_results['filtered'][-1]:.2f} kg (error: {simple_error[-1]:+.2f})")
    print(f"  Augmented estimate: {augmented_results['true_weight'][-1]:.2f} kg (error: {augmented_error[-1]:+.2f})")
    print(f"  Augmented noise est: {augmented_results['noise'][-1]:+.2f} kg (actual: {actual_noise[-1]:+.2f})")

    # Weight change estimation
    true_change = true_weights[-1] - true_weights[0]
    simple_change = simple_results['filtered'][-1] - simple_results['filtered'][0]
    augmented_change = augmented_results['true_weight'][-1] - augmented_results['true_weight'][0]

    print("\nWeight Change Over Period:")
    print(f"  True change:      {true_change:.3f} kg")
    print(f"  Simple estimate:  {simple_change:.3f} kg (error: {simple_change - true_change:+.3f})")
    print(f"  Augmented est:    {augmented_change:.3f} kg (error: {augmented_change - true_change:+.3f})")

    # Prediction demo
    print("\nNext-Day Prediction (from augmented filter):")
    pred_meas, pred_weight, pred_std = augmented_kf.predict_next(days_ahead=1)
    print(f"  Predicted measurement: {pred_meas:.2f} ± {pred_std:.2f} kg")
    print(f"  Predicted true weight: {pred_weight:.2f} kg")
    print(f"  (Accounts for {augmented_results['noise'][-1]:+.2f} kg water retention decay)")

    print("\n" + "="*80)
    print("Generating visualization...")

    # Create plots
    fig = plot_filter_comparison(dates, true_weights, measurements,
                                 simple_results, augmented_results, actual_noise)

    output_path = 'filter_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

    print("\nConclusion:")
    print("The augmented Kalman filter (AR model) significantly outperforms the")
    print("simple filter when noise is autocorrelated (which is the case for")
    print("bodyweight due to water retention, glycogen, DOMS, hormones, etc.)")
    print("="*80)


if __name__ == '__main__':
    main()
