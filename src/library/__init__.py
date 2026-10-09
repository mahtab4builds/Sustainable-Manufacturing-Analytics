"""
Shared analytical library for the Sustainable Manufacturing Analytics project.

This package contains reusable configuration, logging, I/O,
validation, calculation, and plotting utilities.
"""

from .config import (
    PROJECT_ROOT,
    PROJECT_NAME,
    DATA_FILE,
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
    EMISSION_FACTOR,
)

from .logger import get_logger
from .io import (
    ensure_directory,
    load_excel,
    load_csv,
    save_csv,
    save_dataframe,
)
from .calculations import (
    safe_divide,
    calculate_percentage,
    calculate_energy_per_unit,
    calculate_co2e,
)
from .validation import (
    validate_required_columns,
    validate_record_count,
    validate_no_negative_values,
)