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
from augmented_kalman_filter import AugmentedBodyweightKalmanFilter, auto_tune_filter
from velocity_kalman_filter import VelocityBodyweightKalmanFilter, auto_tune_velocity_filter
from sheets_integration import GoogleSheetsClient, get_spreadsheet_id_from_url


def plot_results(data: pd.DataFrame, results: dict, output_path: str = None,
                 filter_type: str = 'simple'):
    """
    Create a visualization of the raw and filtered bodyweight data.

    Args:
        data: DataFrame with date and weight columns
        results: Dictionary from filter.get_results()
        output_path: Optional path to save the plot
        filter_type: 'simple', 'augmented', or 'velocity'
    """
    dates = data['date'].values

    if filter_type == 'velocity':
        # Four-panel plot for velocity filter
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

        # Top: measurements and weight estimate
        ax1.plot(dates, results['measurements'], 'o-', alpha=0.5,
                label='Raw measurements', color='lightblue', markersize=4)
        ax1.plot(dates, results['weight'], '-', linewidth=2,
                label='True weight estimate', color='darkblue')
        ax1.fill_between(dates,
                        results['weight'] - 2 * results['weight_std'],
                        results['weight'] + 2 * results['weight_std'],
                        alpha=0.2, color='darkblue',
                        label='95% confidence interval')
        ax1.set_ylabel('Weight (kg)')
        ax1.set_title('Bodyweight Tracking with Velocity Kalman Filter')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Second: velocity (trend)
        ax2.plot(dates, results['velocity'] * 7, '-', linewidth=2, color='green')
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax2.fill_between(dates,
                        (results['velocity'] - 2 * results['velocity_std']) * 7,
                        (results['velocity'] + 2 * results['velocity_std']) * 7,
                        alpha=0.2, color='green')
        ax2.set_ylabel('Rate (kg/week)')
        ax2.set_title('Weight Change Velocity (Diet/Bulk Trend)')
        ax2.grid(True, alpha=0.3)

        # Third: water retention / noise estimate
        ax3.plot(dates, results['noise'], '-', linewidth=2, color='purple')
        ax3.fill_between(dates, 0, results['noise'],
                        where=results['noise']>0,
                        alpha=0.3, color='red', label='Water retention')
        ax3.fill_between(dates, 0, results['noise'],
                        where=results['noise']<0,
                        alpha=0.3, color='blue', label='Dehydration')
        ax3.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax3.set_ylabel('Noise Level (kg)')
        ax3.set_title('Estimated Water Retention / Temporary Fluctuations')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Bottom: residuals
        residuals = results['measurements'] - results['weight']
        ax4.plot(dates, residuals, 'o-', alpha=0.6, color='coral', markersize=4)
        ax4.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Residual (kg)')
        ax4.set_title('Measurement - True Weight')
        ax4.grid(True, alpha=0.3)

    elif filter_type == 'augmented':
        # Three-panel plot for augmented filter
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        # Top: measurements and true weight estimate
        ax1.plot(dates, results['measurements'], 'o-', alpha=0.5,
                label='Raw measurements', color='lightblue', markersize=4)
        ax1.plot(dates, results['true_weight'], '-', linewidth=2,
                label='True weight estimate', color='darkblue')
        ax1.fill_between(dates,
                        results['true_weight'] - 2 * results['true_weight_std'],
                        results['true_weight'] + 2 * results['true_weight_std'],
                        alpha=0.2, color='darkblue',
                        label='95% confidence interval')
        ax1.set_ylabel('Weight (kg)')
        ax1.set_title('Bodyweight Tracking with Augmented Kalman Filter (AR Model)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Middle: water retention / noise estimate
        ax2.plot(dates, results['noise'], '-', linewidth=2, color='purple')
        ax2.fill_between(dates, 0, results['noise'],
                        where=results['noise']>0,
                        alpha=0.3, color='red', label='Water retention')
        ax2.fill_between(dates, 0, results['noise'],
                        where=results['noise']<0,
                        alpha=0.3, color='blue', label='Dehydration')
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax2.set_ylabel('Noise Level (kg)')
        ax2.set_title('Estimated Water Retention / Temporary Fluctuations')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Bottom: residuals
        residuals = results['measurements'] - results['true_weight']
        ax3.plot(dates, residuals, 'o-', alpha=0.6, color='coral', markersize=4)
        ax3.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Residual (kg)')
        ax3.set_title('Measurement - True Weight')
        ax3.grid(True, alpha=0.3)

    else:
        # Two-panel plot for simple filter
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Top: measurements and filtered estimate
        ax1.plot(dates, results['measurements'], 'o-', alpha=0.5,
                label='Raw measurements', color='lightblue', markersize=4)
        ax1.plot(dates, results['filtered'], '-', linewidth=2,
                label='Filtered estimate (true weight)', color='darkblue')
        ax1.fill_between(dates,
                        results['filtered'] - 2 * results['errors'],
                        results['filtered'] + 2 * results['errors'],
                        alpha=0.2, color='darkblue',
                        label='95% confidence interval')
        ax1.set_ylabel('Weight (kg)')
        ax1.set_title('Bodyweight Tracking with Simple Kalman Filter')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Bottom: residuals
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


def print_summary(results: dict, data: pd.DataFrame, filter_type: str = 'simple'):
    """Print a summary of the results."""
    latest_measurement = results['measurements'][-1]

    if filter_type == 'velocity':
        current_estimate = results['weight'][-1]
        current_error = results['weight_std'][-1]
        current_velocity = results['velocity'][-1]
        current_velocity_std = results['velocity_std'][-1]
        current_noise = results['noise'][-1]
        weight_change = results['weight'][-1] - results['weight'][0]
        avg_residual = abs(results['measurements'] - results['weight']).mean()
        filter_name = 'Velocity (AR + trend)'
    elif filter_type == 'augmented':
        current_estimate = results['true_weight'][-1]
        current_error = results['true_weight_std'][-1]
        current_noise = results['noise'][-1]
        weight_change = results['true_weight'][-1] - results['true_weight'][0]
        avg_residual = abs(results['measurements'] - results['true_weight']).mean()
        filter_name = 'Augmented (AR model)'
    else:
        current_estimate = results['filtered'][-1]
        current_error = results['errors'][-1]
        weight_change = results['filtered'][-1] - results['filtered'][0]
        avg_residual = abs(results['measurements'] - results['filtered']).mean()
        filter_name = 'Simple (iid)'

    print("\n" + "="*60)
    print("BODYWEIGHT TRACKING SUMMARY")
    print("="*60)
    print(f"Filter type: {filter_name}")
    print(f"Total measurements: {len(results['measurements'])}")
    print(f"Date range: {data['date'].min().strftime('%Y-%m-%d')} to "
          f"{data['date'].max().strftime('%Y-%m-%d')}")
    print()
    print(f"Latest measurement:    {latest_measurement:.2f} kg")
    print(f"Estimated true weight: {current_estimate:.2f} ± {current_error:.2f} kg")

    if filter_type in ['velocity', 'augmented']:
        print(f"Estimated water retention: {current_noise:+.2f} kg")
        if current_noise > 0.3:
            print(f"  → Currently holding ~{current_noise:.1f} kg extra water")
        elif current_noise < -0.3:
            print(f"  → Currently ~{-current_noise:.1f} kg dehydrated")
        else:
            print(f"  → Near baseline hydration")

        if filter_type == 'velocity':
            print(f"Estimated velocity: {current_velocity * 7:+.3f} ± {current_velocity_std * 7:.3f} kg/week")
            if abs(current_velocity * 7) > 0.1:
                phase = "cutting" if current_velocity < 0 else "bulking"
                print(f"  → Currently {phase}")
            else:
                print(f"  → Near maintenance")
    else:
        print(f"Daily fluctuation:     {latest_measurement - current_estimate:+.2f} kg")

    print()

    # Calculate statistics
    days = (data['date'].max() - data['date'].min()).days

    print(f"Average daily fluctuation: ±{avg_residual:.2f} kg")
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

  # Use velocity filter (handles trends + autocorrelated noise - BEST):
  %(prog)s --spreadsheet YOUR_ID --velocity --auto-tune

  # Use augmented filter (handles autocorrelated noise):
  %(prog)s --spreadsheet YOUR_ID --augmented

  # Auto-tune filters from data:
  %(prog)s --spreadsheet YOUR_ID --velocity --auto-tune
  %(prog)s --spreadsheet YOUR_ID --augmented --auto-tune

  # Specify custom range and credentials:
  %(prog)s --spreadsheet YOUR_ID --range "Data!A:B" --credentials mycreds.json

  # Save output plot:
  %(prog)s --spreadsheet YOUR_ID --output bodyweight_plot.png

  # Adjust simple filter parameters:
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
    parser.add_argument('--velocity', '-v', action='store_true',
                       help='Use velocity Kalman filter (models trends + autocorrelated noise)')
    parser.add_argument('--augmented', '-a', action='store_true',
                       help='Use augmented Kalman filter (models autocorrelated noise)')
    parser.add_argument('--auto-tune', action='store_true',
                       help='Auto-tune filter parameters from data')
    parser.add_argument('--autocorr', type=float, default=0.7,
                       help='Autocorrelation coefficient for augmented filter (default: 0.7)')
    parser.add_argument('--process-var', type=float, default=1e-5,
                       help='Process variance for Kalman filter (default: 1e-5)')
    parser.add_argument('--measurement-var', type=float, default=0.5,
                       help='Measurement variance for simple filter (default: 0.5)')
    parser.add_argument('--noise-var', type=float, default=0.3,
                       help='AR(1) innovation variance for augmented filter (default: 0.3)')
    parser.add_argument('--measurement-noise', type=float, default=0.01,
                       help='Scale measurement error variance for augmented filter (default: 0.01)')
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

        # Determine filter type
        if args.velocity:
            filter_type = 'velocity'
        elif args.augmented:
            filter_type = 'augmented'
        else:
            filter_type = 'simple'

        # Initialize and run Kalman filter
        if filter_type == 'velocity':
            if args.auto_tune:
                print("Auto-tuning velocity Kalman filter from data...")
                kf = auto_tune_velocity_filter(data['weight'].values)
            else:
                print("Applying velocity Kalman filter (trend + AR noise model)...")
                kf = VelocityBodyweightKalmanFilter(
                    autocorrelation=args.autocorr,
                    acceleration_variance=args.process_var,
                    noise_variance=args.noise_var,
                    measurement_noise=args.measurement_noise
                )
                print(f"  Autocorrelation: {args.autocorr}")
                print(f"  Acceleration variance: {args.process_var} (kg/day²)²")
                print(f"  AR(1) innovation variance: {args.noise_var} kg²")
                print(f"  Measurement noise (R): {args.measurement_noise} kg²")

            kf.batch_filter(data['weight'].values)
            results = kf.get_results()

        elif filter_type == 'augmented':
            if args.auto_tune:
                print("Auto-tuning augmented Kalman filter from data...")
                kf = auto_tune_filter(data['weight'].values)
            else:
                print("Applying augmented Kalman filter (AR noise model)...")
                kf = AugmentedBodyweightKalmanFilter(
                    autocorrelation=args.autocorr,
                    process_variance=args.process_var,
                    noise_variance=args.noise_var,
                    measurement_noise=args.measurement_noise
                )
                print(f"  Autocorrelation: {args.autocorr}")
                print(f"  Process variance: {args.process_var}")
                print(f"  AR(1) innovation variance: {args.noise_var}")
                print(f"  Measurement noise (R): {args.measurement_noise}")

            kf.batch_filter(data['weight'].values)
            results = kf.get_results()

        else:  # simple
            print("Applying simple Kalman filter (iid assumption)...")
            kf = BodyweightKalmanFilter(
                process_variance=args.process_var,
                measurement_variance=args.measurement_var
            )
            kf.batch_filter(data['weight'].values)
            results = kf.get_results()

        # Print summary
        print_summary(results, data, filter_type=filter_type)

        # Export to CSV if requested
        if args.csv:
            if filter_type == 'velocity':
                output_df = pd.DataFrame({
                    'date': data['date'],
                    'raw_weight': results['measurements'],
                    'true_weight': results['weight'],
                    'velocity_kg_per_week': results['velocity'] * 7,
                    'water_retention': results['noise'],
                    'weight_std': results['weight_std'],
                    'velocity_std': results['velocity_std'],
                    'residual': results['measurements'] - results['weight']
                })
            elif filter_type == 'augmented':
                output_df = pd.DataFrame({
                    'date': data['date'],
                    'raw_weight': results['measurements'],
                    'true_weight': results['true_weight'],
                    'water_retention': results['noise'],
                    'true_weight_std': results['true_weight_std'],
                    'residual': results['measurements'] - results['true_weight']
                })
            else:
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
            plot_results(data, results, args.output, filter_type=filter_type)

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
