# Autocorrelated Noise in Bodyweight Measurements

## The Problem

Standard Kalman filters assume **iid measurement noise** (independent and identically distributed). However, bodyweight measurement "noise" is strongly **autocorrelated** due to:

1. **Water Retention Waves**: Multi-day patterns from sodium, hormones, training
2. **Glycogen Storage**: Carb intake causes 3-4g water per 1g glycogen, persists for days
3. **DOMS & Inflammation**: Muscle damage causes localized water retention for 2-5 days
4. **TRT Effects**: Testosterone influences water retention, estrogen management
5. **Digestive Transit**: Food mass in GI tract (24-72 hour cycles)

If you weigh 2kg heavy today due to water retention, you're likely to be heavy tomorrow too. This violates the iid assumption.

## Mathematical Formulation

### Standard Model (Incorrect)
```
True weight:  x_t = x_{t-1} + w_t        (process noise)
Measurement:  y_t = x_t + v_t            (iid noise)
```

### Reality (Autocorrelated Noise)
```
True weight:  x_t = x_{t-1} + w_t
Noise:        n_t = ρ·n_{t-1} + v_t      (AR(1) process)
Measurement:  y_t = x_t + n_t
```

Where:
- `ρ` (rho) ∈ [0, 1] is the autocorrelation coefficient
- `ρ = 0`: iid noise (standard Kalman)
- `ρ = 0.5`: moderate autocorrelation
- `ρ = 0.8`: strong autocorrelation (typical for bodyweight)

## Proposed Solutions

### Solution 1: Augmented State Kalman Filter (Best)

**Idea**: Treat the autocorrelated noise as an additional state variable to estimate.

**State Vector**: `[true_weight, noise_level]`

**State Transition**:
```
[x_t  ]   [1  0] [x_{t-1}]   [w_t]
[n_t  ] = [0  ρ] [n_{t-1}] + [v_t]
```

**Measurement**:
```
y_t = [1  1] * [x_t, n_t]^T
```

**Advantages**:
- Principled probabilistic approach
- Estimates both true weight AND current water retention
- Can predict future measurements
- Proper uncertainty quantification

**Disadvantages**:
- Need to estimate ρ (can do from data)
- Slightly more complex

### Solution 2: AR(2) Noise Model

Extend to AR(2) for even more complex patterns:
```
n_t = ρ₁·n_{t-1} + ρ₂·n_{t-2} + v_t
```

Captures weekly cycles (e.g., weekend eating patterns, training splits).

### Solution 3: Moving Average Pre-filtering

**Idea**: Take 3-day or 7-day moving average before Kalman filtering.

**Advantages**: Very simple
**Disadvantages**: Loses information, introduces lag, not optimal

### Solution 4: Exponential Smoothing with Trend

**Idea**: Use Holt-Winters or similar methods.

**Disadvantages**: Not as theoretically sound, doesn't model autocorrelation explicitly

## Recommendation

Implement **Solution 1: Augmented State Kalman Filter with AR(1) noise**.

This gives you:
1. **True weight estimate**: The underlying trend
2. **Current water retention estimate**: How much of today's weight is "fake"
3. **Tomorrow's prediction**: What you'll likely weigh tomorrow given today's water retention

## Example Scenario

Day 1: Heavy training + high carbs
- Scale: 75.0 kg
- Estimated true weight: 73.5 kg
- Estimated water retention: +1.5 kg

Day 2: Rest day, moderate food
- Scale: 74.8 kg (still elevated due to autocorrelation)
- Estimated true weight: 73.5 kg
- Estimated water retention: +1.3 kg (decaying from +1.5)

Day 3: Rest day, lower carbs
- Scale: 74.0 kg
- Estimated true weight: 73.5 kg
- Estimated water retention: +0.5 kg (continuing to decay)

The augmented filter tracks this multi-day wave pattern.

## Implementation Plan

1. Create `AugmentedKalmanFilter` class
2. Auto-estimate ρ from data (using sample autocorrelation)
3. Provide 2D state estimates: (true_weight, water_retention)
4. Add visualization showing both components
5. Compare against simple Kalman filter to show improvement
