"""
Centralized project configuration.

All configurable project paths and analytical parameters are loaded
from the .env file so individual analytical modules remain clean,
portable, and free from hard-coded configuration values.
"""

from pathlib import Path
import os

from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# PROJECT ROOT
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")


# ---------------------------------------------------------------------------
# PROJECT INFORMATION
# ---------------------------------------------------------------------------

PROJECT_NAME = os.getenv(
    "PROJECT_NAME",
    "Sustainable Manufacturing Analytics"
)


# ---------------------------------------------------------------------------
# DATA CONFIGURATION
# ---------------------------------------------------------------------------

DATA_FILE = PROJECT_ROOT / os.getenv(
    "DATA_FILE",
    "data/Sustainable_Manufacturing_Energy_Analytics_10000.xlsx"
)


# ---------------------------------------------------------------------------
# OUTPUT DIRECTORIES
# ---------------------------------------------------------------------------

OUTPUT_DIR = PROJECT_ROOT / os.getenv(
    "OUTPUT_DIR",
    "outputs"
)

CHARTS_DIR = PROJECT_ROOT / os.getenv(
    "CHARTS_DIR",
    "outputs/charts"
)

CLEANED_DATA_DIR = PROJECT_ROOT / os.getenv(
    "CLEANED_DATA_DIR",
    "outputs/cleaned_data"
)

BASELINE_DIR = PROJECT_ROOT / os.getenv(
    "BASELINE_DIR",
    "outputs/baseline"
)

INEFFICIENCY_DIR = PROJECT_ROOT / os.getenv(
    "INEFFICIENCY_DIR",
    "outputs/inefficiency"
)

FORECASTING_DIR = PROJECT_ROOT / os.getenv(
    "FORECASTING_DIR",
    "outputs/forecasting"
)

WHAT_IF_DIR = PROJECT_ROOT / os.getenv(
    "WHAT_IF_DIR",
    "outputs/what_if"
)

CO2_DIR = PROJECT_ROOT / os.getenv(
    "CO2_DIR",
    "outputs/co2_analysis"
)

OPTIMIZATION_DIR = PROJECT_ROOT / os.getenv(
    "OPTIMIZATION_DIR",
    "outputs/optimization"
)

RECOMMENDATIONS_DIR = PROJECT_ROOT / os.getenv(
    "RECOMMENDATIONS_DIR",
    "outputs/recommendations"
)


# ---------------------------------------------------------------------------
# ANALYTICAL PARAMETERS
# ---------------------------------------------------------------------------

EMISSION_FACTOR = float(
    os.getenv(
        "SYNTHETIC_EMISSION_FACTOR",
        "0.7"
    )
)

IDLE_ENERGY_REDUCTION = float(
    os.getenv(
        "IDLE_ENERGY_REDUCTION",
        "0.25"
    )
)

SCHEDULE_ENERGY_REDUCTION = float(
    os.getenv(
        "SCHEDULE_ENERGY_REDUCTION",
        "0.20"
    )
)

ENERGY_INTENSITY_IMPROVEMENT = float(
    os.getenv(
        "ENERGY_INTENSITY_IMPROVEMENT",
        "0.10"
    )
)


# ---------------------------------------------------------------------------
# DIRECTORY INITIALIZATION
# ---------------------------------------------------------------------------

ALL_DIRECTORIES = [
    OUTPUT_DIR,
    CHARTS_DIR,
    CLEANED_DATA_DIR,
    BASELINE_DIR,
    INEFFICIENCY_DIR,
    FORECASTING_DIR,
    WHAT_IF_DIR,
    CO2_DIR,
    OPTIMIZATION_DIR,
    RECOMMENDATIONS_DIR,
]


def initialize_directories():
    """
    Create all configured project directories if they do not exist.
    """

    for directory in ALL_DIRECTORIES:
        directory.mkdir(
            parents=True,
            exist_ok=True
        )