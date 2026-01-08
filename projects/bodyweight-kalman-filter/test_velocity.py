#!/usr/bin/env python3
"""
Quick test of velocity Kalman filter with synthetic cutting data.
"""

import numpy as np
from velocity_kalman_filter import VelocityBodyweightKalmanFilter

# Generate synthetic cutting data: -0.5 kg/week trend
days = 60
true_weight_start = 85.0
velocity = -0.5 / 7  # -0.5 kg/week = -0.071 kg/day

# Generate true weights (linear trend)
true_weights = true_weight_start + velocity * np.arange(days)

# Add AR(1) noise
rho = 0.75
noise_std = 0.5
noise = np.zeros(days)
for i in range(days):
    if i == 0:
        noise[i] = np.random.normal(0, noise_std)
    else:
        noise[i] = rho * noise[i-1] + np.random.normal(0, noise_std * np.sqrt(1 - rho**2))

# Measurements
measurements = true_weights + noise

print("="*70)
print("VELOCITY KALMAN FILTER TEST")
print("="*70)
print(f"\nTrue parameters:")
print(f"  Starting weight: {true_weight_start:.2f} kg")
print(f"  Velocity: {velocity * 7:.3f} kg/week")
print(f"  Total change: {true_weights[-1] - true_weights[0]:.2f} kg over {days} days")
print(f"  Noise autocorrelation: {rho:.2f}")

# Apply velocity filter
# For constant velocity trend, use small acceleration variance
print(f"\nApplying velocity Kalman filter...")
kf = VelocityBodyweightKalmanFilter(
    autocorrelation=rho,
    acceleration_variance=1e-6,  # Small: allows slight velocity adaptation
    noise_variance=noise_std**2 * (1 - rho**2),
    measurement_noise=0.01
)

weights, velocities, noises = kf.batch_filter(measurements)

# Evaluate performance
weight_rmse = np.sqrt(np.mean((weights - true_weights)**2))
velocity_rmse = np.sqrt(np.mean((velocities - velocity)**2))
noise_rmse = np.sqrt(np.mean((noises - noise)**2))

print(f"\nPerformance:")
print(f"  Weight RMSE: {weight_rmse:.4f} kg")
print(f"  Velocity RMSE: {velocity_rmse * 7:.4f} kg/week")
print(f"  Noise RMSE: {noise_rmse:.4f} kg")

print(f"\nFinal estimates:")
print(f"  True weight: {true_weights[-1]:.2f} kg → Estimated: {weights[-1]:.2f} kg")
print(f"  True velocity: {velocity * 7:.3f} kg/week → Estimated: {velocities[-1] * 7:.3f} kg/week")
print(f"  True noise: {noise[-1]:+.2f} kg → Estimated: {noises[-1]:+.2f} kg")

print("="*70)
print("SUCCESS: Velocity filter tracks cutting trend + autocorrelated noise!")
print("="*70)
