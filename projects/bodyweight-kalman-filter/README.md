# Bodyweight Tracking with Kalman Filter

A Python tool that pulls bodyweight data from Google Sheets and applies a Kalman filter to identify your "true" bodyweight by filtering out random daily fluctuations caused by water retention, food intake, and measurement variance.

## Overview

Daily bodyweight measurements can fluctuate significantly due to:
- Water retention and hydration levels
- Food in digestive system
- Time of day
- Measurement errors

This tool uses a **Kalman filter** to estimate your true underlying bodyweight by treating daily measurements as noisy observations of a slowly-changing true weight.

### Why Autocorrelation Matters

**Important**: Bodyweight fluctuations are NOT random (iid) noise - they're **autocorrelated**. If you're holding extra water today, you'll likely still be holding it tomorrow. This creates multi-day waves that violate standard Kalman filter assumptions.

This tool provides **two filter options**:
1. **Simple Kalman Filter**: Assumes iid noise (basic, less accurate)
2. **Augmented Kalman Filter** (⭐ RECOMMENDED): Models autocorrelated noise as an AR(1) process, providing:
   - More accurate true weight estimates
   - Estimates of current water retention/dehydration
   - Better handling of DOMS, hormonal effects, sodium intake patterns

For users on TRT or experiencing significant multi-day weight fluctuations, the **augmented filter is strongly recommended**.

## Features

- 📊 **Google Sheets Integration**: Automatically pulls data from your Google Sheets
- 🎯 **Kalman Filtering**: Statistically optimal filtering of noisy measurements
- 📈 **Visualization**: Plots showing raw measurements vs. filtered estimates
- 📉 **Trend Analysis**: Calculate weight change rates and statistics
- 💾 **CSV Export**: Save filtered results for further analysis

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Google Sheets API Setup

To access your Google Sheet, you need to set up API credentials:

#### Option A: OAuth 2.0 (For Personal Use)

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the **Google Sheets API**
4. Go to "Credentials" → "Create Credentials" → "OAuth client ID"
5. Choose "Desktop app" as the application type
6. Download the credentials JSON file
7. Rename it to `credentials.json` and place it in this directory

#### Option B: Service Account (For Automated Access)

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the **Google Sheets API**
4. Go to "Credentials" → "Create Credentials" → "Service Account"
5. Create a service account and download the JSON key
6. Rename it to `credentials.json` and place it in this directory
7. Share your Google Sheet with the service account email address

### 3. Prepare Your Google Sheet

Your Google Sheet should have at least two columns:
- **Date column**: Dates in any common format (YYYY-MM-DD, MM/DD/YYYY, etc.)
- **Weight column**: Bodyweight measurements in kg

Example:
```
Date       | Weight
2024-01-01 | 75.3
2024-01-02 | 75.8
2024-01-03 | 75.1
...
```

The script will automatically detect header rows and parse the data.

## Usage

### Recommended: Augmented Filter (Handles Autocorrelated Noise)

```bash
# Use augmented filter with auto-tuning (RECOMMENDED)
python bodyweight_tracker.py --spreadsheet YOUR_ID --augmented --auto-tune

# Or with manual autocorrelation parameter
python bodyweight_tracker.py --spreadsheet YOUR_ID --augmented --autocorr 0.75

# Using spreadsheet URL
python bodyweight_tracker.py --url "https://docs.google.com/spreadsheets/d/YOUR_ID/edit" --augmented
```

The augmented filter provides:
- More accurate true weight estimates by modeling multi-day fluctuation patterns
- Estimates of your current water retention/dehydration level
- Better performance for users on TRT, those with DOMS patterns, or high sodium variability

### Simple Filter (Basic, Assumes iid Noise)

```bash
# Simple filter (less accurate, but faster)
python bodyweight_tracker.py --spreadsheet YOUR_ID
```

### Additional Options

```bash
# Specify custom sheet range
python bodyweight_tracker.py --spreadsheet YOUR_ID --range "Data!A:B"

# Use custom credentials file
python bodyweight_tracker.py --spreadsheet YOUR_ID --credentials mycreds.json

# Save plot to file instead of displaying
python bodyweight_tracker.py --spreadsheet YOUR_ID --augmented --output bodyweight_plot.png

# Export filtered results to CSV
python bodyweight_tracker.py --spreadsheet YOUR_ID --augmented --csv results.csv

# Show summary only, no plot
python bodyweight_tracker.py --spreadsheet YOUR_ID --augmented --no-plot
```

### Demo and Comparison

```bash
# Run demo with synthetic data (no Google Sheets needed)
python example_demo.py

# Compare simple vs augmented filters
python compare_filters.py
```

### Filter Parameters

#### Augmented Filter Parameters

- `--autocorr`: Autocorrelation coefficient ρ (default: 0.7, range: 0-0.95)
  - How much noise persists day-to-day
  - 0.7-0.8: Typical for bodyweight
  - Higher for strong TRT effects or consistent training patterns
  - Use `--auto-tune` to estimate from your data

- `--noise-var`: Variance of daily noise innovations (default: 0.3 kg²)
- `--process-var`: True weight change variance (default: 1e-5 kg²)

#### Simple Filter Parameters

- `--measurement-var`: Expected measurement variance (default: 0.5 kg²)
  - Smaller = trusts measurements more
  - Larger = expects more noise

- `--process-var`: True weight change variance (default: 1e-5 kg²)
  - Smaller = smoother curve
  - Larger = more responsive to trends

## How It Works

### Simple Kalman Filter (iid assumption)

The basic Kalman filter assumes **iid noise**:
- **State**: True bodyweight (changes slowly)
- **Measurements**: Daily scale readings = true weight + random noise
- **Assumption**: Daily fluctuations are independent

This works reasonably well but **underestimates true weight changes** and **lags behind trends** when noise is actually autocorrelated.

### Augmented Kalman Filter (AR model) ⭐

The augmented filter models **autocorrelated noise** explicitly:

**State**: `[true_weight, noise_level]`
- True weight: Underlying trend
- Noise level: Current water retention/dehydration (persists across days)

**Noise dynamics**: `noise(t) = ρ × noise(t-1) + new_noise`
- ρ (autocorrelation): How much yesterday's fluctuation carries over
- If you're +2kg heavy today due to water, you'll likely be +1.4kg heavy tomorrow (with ρ=0.7)

**Why this matters**:
- Water retention from high sodium, carbs, training, or hormones lasts multiple days
- DOMS-related inflammation persists 2-5 days
- TRT can create multi-day water retention patterns
- Modeling this autocorrelation improves true weight estimates by 30-50%

**Output**:
1. **True weight estimate**: Your actual body composition trend
2. **Water retention estimate**: How much temporary fluctuation you're experiencing
3. **Tomorrow's prediction**: Expected weight based on current retention level

### Output Interpretation

The script produces:

1. **Summary Statistics**:
   - Latest measurement vs. estimated true weight
   - Daily fluctuation magnitude
   - Weight change trends

2. **Visualization**:
   - Top plot: Raw measurements (blue dots) vs. filtered estimate (dark blue line)
   - Shaded area: 95% confidence interval
   - Bottom plot: Daily residuals (measurement - estimate)

3. **CSV Export** (optional):
   - Date, raw weight, filtered weight, error estimates, residuals

## Project Structure

```
bodyweight-kalman-filter/
├── bodyweight_tracker.py        # Main script
├── kalman_filter.py             # Simple Kalman filter (iid)
├── augmented_kalman_filter.py   # Augmented filter (AR model)
├── sheets_integration.py        # Google Sheets API client
├── compare_filters.py           # Compare simple vs augmented
├── example_demo.py              # Demo with synthetic data
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── AUTOCORRELATION_ANALYSIS.md  # Technical analysis
├── .gitignore                   # Ignore credentials and outputs
└── credentials.json             # Your API credentials (not in git)
```

## Example Output

```
BODYWEIGHT TRACKING SUMMARY
============================================================
Total measurements: 90
Date range: 2023-10-01 to 2023-12-29

Latest measurement:    75.8 kg
Estimated true weight: 75.3 ± 0.15 kg
Daily fluctuation:     +0.5 kg

Average daily fluctuation: ±0.68 kg
Total weight change: -2.4 kg over 89 days
Average rate: -0.188 kg/week
============================================================
```

## Tips

- **Measure consistently**: Weigh yourself at the same time each day (e.g., morning after waking)
- **Track regularly**: The filter works best with consistent daily measurements
- **Be patient**: It takes several days for the filter to stabilize
- **Adjust parameters**: If the filtered line is too smooth or too jumpy, tune the filter parameters

## License

MIT License - See the root LICENSE file for details

## References

- [Kalman Filter](https://en.wikipedia.org/wiki/Kalman_filter)
- [Google Sheets API](https://developers.google.com/sheets/api)
