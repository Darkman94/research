# Bodyweight Tracking with Kalman Filter

A Python tool that pulls bodyweight data from Google Sheets and applies a Kalman filter to identify your "true" bodyweight by filtering out random daily fluctuations caused by water retention, food intake, and measurement variance.

## Overview

Daily bodyweight measurements can fluctuate significantly due to:
- Water retention and hydration levels
- Food in digestive system
- Time of day
- Measurement errors

This tool uses a **Kalman filter** to estimate your true underlying bodyweight by treating daily measurements as noisy observations of a slowly-changing true weight.

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

### Basic Usage

```bash
# Using spreadsheet URL
python bodyweight_tracker.py --url "https://docs.google.com/spreadsheets/d/YOUR_SPREADSHEET_ID/edit"

# Or using spreadsheet ID directly
python bodyweight_tracker.py --spreadsheet YOUR_SPREADSHEET_ID
```

### Advanced Options

```bash
# Specify custom sheet range
python bodyweight_tracker.py --spreadsheet YOUR_ID --range "Data!A:B"

# Use custom credentials file
python bodyweight_tracker.py --spreadsheet YOUR_ID --credentials mycreds.json

# Save plot to file instead of displaying
python bodyweight_tracker.py --spreadsheet YOUR_ID --output bodyweight_plot.png

# Export filtered results to CSV
python bodyweight_tracker.py --spreadsheet YOUR_ID --csv results.csv

# Adjust Kalman filter parameters
python bodyweight_tracker.py --spreadsheet YOUR_ID \
    --process-var 1e-4 \
    --measurement-var 1.0

# Show summary only, no plot
python bodyweight_tracker.py --spreadsheet YOUR_ID --no-plot
```

### Kalman Filter Parameters

The filter behavior can be tuned with two parameters:

- `--process-var`: How much your true weight can change per day (default: 1e-5)
  - Smaller = assumes weight changes slowly (smoother filtered curve)
  - Larger = allows faster weight changes (more responsive to trends)

- `--measurement-var`: Expected variance in daily measurements (default: 0.5)
  - Smaller = trusts measurements more (less smoothing)
  - Larger = expects more noise (more smoothing)

## How It Works

### The Kalman Filter

The Kalman filter is an optimal recursive algorithm that estimates the state of a system from noisy measurements. For bodyweight tracking:

- **State**: Your true bodyweight at any given time
- **Measurements**: Your daily scale readings (noisy observations)
- **Process model**: Weight changes slowly day-to-day
- **Measurement model**: Scale readings have random noise

The filter works in two steps:
1. **Prediction**: Estimate today's weight based on yesterday's estimate
2. **Update**: Combine the prediction with today's measurement, weighted by their uncertainties

This produces an optimal estimate that:
- Smooths out random daily fluctuations
- Responds to real weight changes over time
- Provides uncertainty estimates (confidence intervals)

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
├── bodyweight_tracker.py    # Main script
├── kalman_filter.py          # Kalman filter implementation
├── sheets_integration.py     # Google Sheets API client
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── .gitignore               # Ignore credentials and outputs
└── credentials.json         # Your Google API credentials (not in git)
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
