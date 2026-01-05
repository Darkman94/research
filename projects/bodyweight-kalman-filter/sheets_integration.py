"""
Google Sheets integration for bodyweight data.

This module handles authentication and data retrieval from Google Sheets.
"""

import os
from datetime import datetime
from typing import List, Tuple, Optional

from google.oauth2.credentials import Credentials
from google.oauth2 import service_account
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import pandas as pd


# If modifying these scopes, delete the token file.
SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']


class GoogleSheetsClient:
    """Client for reading bodyweight data from Google Sheets."""

    def __init__(self, credentials_path: str = 'credentials.json',
                 token_path: str = 'token.json'):
        """
        Initialize the Google Sheets client.

        Args:
            credentials_path: Path to OAuth2 credentials JSON file
            token_path: Path to store/load authentication token
        """
        self.credentials_path = credentials_path
        self.token_path = token_path
        self.service = None
        self._authenticate()

    def _authenticate(self):
        """Authenticate with Google Sheets API."""
        creds = None

        # Try to load existing token
        if os.path.exists(self.token_path):
            creds = Credentials.from_authorized_user_file(self.token_path, SCOPES)

        # If no valid credentials, authenticate
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                # Check if using service account
                if os.path.exists(self.credentials_path):
                    try:
                        creds = service_account.Credentials.from_service_account_file(
                            self.credentials_path, scopes=SCOPES)
                    except ValueError:
                        # Not a service account, use OAuth flow
                        flow = InstalledAppFlow.from_client_secrets_file(
                            self.credentials_path, SCOPES)
                        creds = flow.run_local_server(port=0)

                        # Save the credentials for the next run
                        with open(self.token_path, 'w') as token:
                            token.write(creds.to_json())

        self.service = build('sheets', 'v4', credentials=creds)

    def get_bodyweight_data(self, spreadsheet_id: str,
                           range_name: str = 'Sheet1!A:B',
                           date_column: int = 0,
                           weight_column: int = 1) -> pd.DataFrame:
        """
        Fetch bodyweight data from Google Sheets.

        Args:
            spreadsheet_id: The ID of the Google Sheet
            range_name: The A1 notation of the range to retrieve (e.g., 'Sheet1!A:B')
            date_column: Index of the date column (0-based)
            weight_column: Index of the weight column (0-based)

        Returns:
            DataFrame with 'date' and 'weight' columns, sorted by date
        """
        # Call the Sheets API
        sheet = self.service.spreadsheets()
        result = sheet.values().get(spreadsheetId=spreadsheet_id,
                                   range=range_name).execute()
        values = result.get('values', [])

        if not values:
            raise ValueError('No data found in spreadsheet')

        # Convert to DataFrame
        df = pd.DataFrame(values)

        # Handle header row if present
        if self._is_header_row(df.iloc[0]):
            df.columns = df.iloc[0]
            df = df[1:]
            df.reset_index(drop=True, inplace=True)
        else:
            # Use default column names
            df.columns = [f'col_{i}' for i in range(len(df.columns))]

        # Extract date and weight columns
        date_col_name = df.columns[date_column]
        weight_col_name = df.columns[weight_column]

        # Create clean DataFrame
        data = pd.DataFrame({
            'date': df[date_col_name],
            'weight': df[weight_col_name]
        })

        # Clean and convert data types
        data = self._clean_data(data)

        # Sort by date
        data = data.sort_values('date').reset_index(drop=True)

        return data

    def _is_header_row(self, row) -> bool:
        """Check if the first row appears to be a header."""
        # Simple heuristic: if first cell contains common header words
        first_cell = str(row.iloc[0]).lower()
        header_keywords = ['date', 'time', 'day', 'weight', 'mass', 'kg', 'lbs']
        return any(keyword in first_cell for keyword in header_keywords)

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate the data."""
        # Remove rows with missing values
        df = df.dropna()

        # Remove empty strings
        df = df[(df['date'].str.strip() != '') & (df['weight'].str.strip() != '')]

        # Parse dates
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

        # Convert weight to float
        df['weight'] = pd.to_numeric(df['weight'], errors='coerce')

        # Remove rows where parsing failed
        df = df.dropna()

        # Remove duplicate dates (keep first occurrence)
        df = df.drop_duplicates(subset=['date'], keep='first')

        return df


def get_spreadsheet_id_from_url(url: str) -> str:
    """
    Extract spreadsheet ID from a Google Sheets URL.

    Args:
        url: Google Sheets URL

    Returns:
        Spreadsheet ID
    """
    # URL format: https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}/edit...
    if '/d/' in url:
        return url.split('/d/')[1].split('/')[0]
    else:
        # Assume it's already just the ID
        return url
