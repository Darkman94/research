#!/usr/bin/env python3
"""
Bodyweight Tracker with Kalman Filtering

This script pulls bodyweight data from Google Sheets and applies a Kalman filter
to estimate the "true" bodyweight, filtering out daily fluctuations.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from kalman_filter import BodyweightKalmanFilter
from sheets_integration import GoogleSheetsClient, get_spreadsheet_id_from_url


def plot_results(data: pd.DataFrame, results: dict, output_path: str = None):
    """
    Create a visualization of the raw and filtered bodyweight data.

    Args:
        data: DataFrame with date and weight columns
        results: Dictionary from BodyweightKalmanFilter.get_results()
        output_path: Optional path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    dates = data['date'].values

    # Top plot: Raw measurements vs filtered estimate
    ax1.plot(dates, results['measurements'], 'o-', alpha=0.5,
             label='Raw measurements', color='lightblue', markersize=4)
    ax1.plot(dates, results['filtered'], '-', linewidth=2,
             label='Filtered estimate (true weight)', color='darkblue')

    # Add confidence interval
    ax1.fill_between(dates,
                     results['filtered'] - 2 * results['errors'],
                     results['filtered'] + 2 * results['errors'],
                     alpha=0.2, color='darkblue',
                     label='95% confidence interval')

    ax1.set_ylabel('Weight (kg)')
    ax1.set_title('Bodyweight Tracking with Kalman Filter')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Bottom plot: Measurement residuals (noise)
    residuals = results['measurements'] - results['filtered']
    ax2.plot(dates, residuals, 'o-', alpha=0.6, color='coral', markersize=4)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Residual (kg)')
    ax2.set_title('Daily Fluctuations (Measurement - True Weight)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()


def print_summary(results: dict, data: pd.DataFrame):
    """Print a summary of the results."""
    current_estimate = results['filtered'][-1]
    current_error = results['errors'][-1]
    latest_measurement = results['measurements'][-1]

    print("\n" + "="*60)
    print("BODYWEIGHT TRACKING SUMMARY")
    print("="*60)
    print(f"Total measurements: {len(results['measurements'])}")
    print(f"Date range: {data['date'].min().strftime('%Y-%m-%d')} to "
          f"{data['date'].max().strftime('%Y-%m-%d')}")
    print()
    print(f"Latest measurement:    {latest_measurement:.2f} kg")
    print(f"Estimated true weight: {current_estimate:.2f} ± {current_error:.2f} kg")
    print(f"Daily fluctuation:     {latest_measurement - current_estimate:+.2f} kg")
    print()

    # Calculate statistics
    avg_noise = abs(results['measurements'] - results['filtered']).mean()
    weight_change = results['filtered'][-1] - results['filtered'][0]
    days = (data['date'].max() - data['date'].min()).days

    print(f"Average daily fluctuation: ±{avg_noise:.2f} kg")
    print(f"Total weight change: {weight_change:+.2f} kg over {days} days")
    if days > 0:
        print(f"Average rate: {weight_change / days * 7:+.3f} kg/week")
    print("="*60 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Track bodyweight from Google Sheets with Kalman filtering',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using spreadsheet ID:
  %(prog)s --spreadsheet YOUR_SPREADSHEET_ID

  # Using spreadsheet URL:
  %(prog)s --url "https://docs.google.com/spreadsheets/d/YOUR_ID/edit"

  # Specify custom range and credentials:
  %(prog)s --spreadsheet YOUR_ID --range "Data!A:B" --credentials mycreds.json

  # Save output plot:
  %(prog)s --spreadsheet YOUR_ID --output bodyweight_plot.png

  # Adjust filter parameters:
  %(prog)s --spreadsheet YOUR_ID --process-var 1e-4 --measurement-var 1.0
        """)

    parser.add_argument('--spreadsheet', '-s', type=str,
                       help='Google Sheets spreadsheet ID')
    parser.add_argument('--url', '-u', type=str,
                       help='Google Sheets URL (alternative to --spreadsheet)')
    parser.add_argument('--range', '-r', type=str, default='Sheet1!A:B',
                       help='Range to read in A1 notation (default: Sheet1!A:B)')
    parser.add_argument('--credentials', '-c', type=str, default='credentials.json',
                       help='Path to Google API credentials (default: credentials.json)')
    parser.add_argument('--token', '-t', type=str, default='token.json',
                       help='Path to store auth token (default: token.json)')
    parser.add_argument('--output', '-o', type=str,
                       help='Path to save output plot (if not specified, displays plot)')
    parser.add_argument('--process-var', type=float, default=1e-5,
                       help='Process variance for Kalman filter (default: 1e-5)')
    parser.add_argument('--measurement-var', type=float, default=0.5,
                       help='Measurement variance for Kalman filter (default: 0.5)')
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip plotting, only show summary')
    parser.add_argument('--csv', type=str,
                       help='Export results to CSV file')

    args = parser.parse_args()

    # Validate arguments
    if not args.spreadsheet and not args.url:
        parser.error('Either --spreadsheet or --url must be provided')

    # Get spreadsheet ID
    if args.url:
        spreadsheet_id = get_spreadsheet_id_from_url(args.url)
    else:
        spreadsheet_id = args.spreadsheet

    try:
        # Initialize Google Sheets client
        print("Authenticating with Google Sheets...")
        client = GoogleSheetsClient(
            credentials_path=args.credentials,
            token_path=args.token
        )

        # Fetch data
        print(f"Fetching data from spreadsheet {spreadsheet_id}...")
        data = client.get_bodyweight_data(spreadsheet_id, args.range)
        print(f"Loaded {len(data)} measurements")

        # Initialize Kalman filter
        kf = BodyweightKalmanFilter(
            process_variance=args.process_var,
            measurement_variance=args.measurement_var
        )

        # Process measurements
        print("Applying Kalman filter...")
        kf.batch_filter(data['weight'].values)
        results = kf.get_results()

        # Print summary
        print_summary(results, data)

        # Export to CSV if requested
        if args.csv:
            output_df = pd.DataFrame({
                'date': data['date'],
                'raw_weight': results['measurements'],
                'filtered_weight': results['filtered'],
                'std_error': results['errors'],
                'residual': results['measurements'] - results['filtered']
            })
            output_df.to_csv(args.csv, index=False)
            print(f"Results exported to {args.csv}")

        # Plot results
        if not args.no_plot:
            print("Generating plot...")
            plot_results(data, results, args.output)

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("\nMake sure you have downloaded your Google API credentials.", file=sys.stderr)
        print("See README.md for setup instructions.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
