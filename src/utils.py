import os
from pathlib import Path

# Base Directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data Paths
RAW_CLIENTS_PATH = BASE_DIR / "clients.csv"
RAW_PROPERTIES_PATH = BASE_DIR / "properties.csv"

PROCESSED_DATA_DIR = BASE_DIR / "data"
CLEANED_CLIENTS_PATH = PROCESSED_DATA_DIR / "cleaned_clients.csv"
CLEANED_PROPERTIES_PATH = PROCESSED_DATA_DIR / "cleaned_properties.csv"
BUYER_PROFILES_PATH = PROCESSED_DATA_DIR / "buyer_profiles.csv"

# Report Paths
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
DATA_QUALITY_REPORT_PATH = REPORTS_DIR / "data_quality_report.md"
EDA_REPORT_PATH = REPORTS_DIR / "eda_report.md"

# Reference Constants
# Dataset transactions span 2024-01-01 to 2025-12-01.
# Fixed reference date 2025-12-31 is used for standardizing buyer age calculations.
REFERENCE_DATE = "2025-12-31"

def ensure_directories():
    """Ensure output directories exist."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def format_currency(value: float) -> str:
    """Format float value to currency string."""
    return f"${value:,.2f}"
