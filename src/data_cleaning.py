"""
Sustainable Manufacturing:
Energy Consumption & Carbon Footprint Optimization Analytics

Module:
    Data Cleaning & Validation Pipeline

Purpose:
    Clean, validate, classify, and prepare the synthetic manufacturing
    dataset for downstream analytics, forecasting, what-if analysis,
    CO2e analysis, optimization, and dashboard development.

Pipeline:
    Raw Data
        ↓
    Schema Validation
        ↓
    Duplicate Validation
        ↓
    Timestamp Validation
        ↓
    Environmental Data Treatment
        ↓
    State Consistency Validation
        ↓
    Production Consistency Validation
        ↓
    Energy Consistency Validation
        ↓
    Schedule Deviation Classification
        ↓
    Final Validation
        ↓
    Clean Analytical Dataset

Important:
    The original raw Excel workbook is NEVER modified.
"""

from pathlib import Path
from datetime import datetime
from library.data_source import load_data

import logging
import sys

import numpy as np
import pandas as pd


# ============================================================================
# SHARED LIBRARY
# ============================================================================

try:
    from library import constants as C
    from library import config as CFG

except ImportError as error:
    print(f"Shared library import failed: {error}")
    print("Make sure src/library exists and contains the shared modules.")
    sys.exit(1)


# ============================================================================
# PROJECT CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(
    getattr(
        CFG,
        "PROJECT_ROOT",
        Path(__file__).resolve().parent.parent
    )
)

DATA_DIR = Path(
    getattr(
        CFG,
        "DATA_DIR",
        PROJECT_ROOT / "data"
    )
)

OUTPUT_DIR = Path(
    getattr(
        CFG,
        "OUTPUT_DIR",
        PROJECT_ROOT / "outputs"
    )
)

CLEANED_DATA_DIR = (
    OUTPUT_DIR / "cleaned_data"
)

VALIDATION_DIR = (
    OUTPUT_DIR / "validation"
)

INPUT_FILE = Path(
    getattr(
        CFG,
        "INPUT_FILE",
        DATA_DIR / "Sustainable_Manufacturing_Energy_Analytics_10000.xlsx"
    )
)

DATA_SHEET = getattr(
    C,
    "DATA_SHEET",
    "Manufacturing_Energy_Data"
)

EXPECTED_RECORDS = int(
    getattr(
        C,
        "EXPECTED_RECORDS",
        10_000
    )
)

EXPECTED_MACHINES = set(
    getattr(
        C,
        "EXPECTED_MACHINES",
        {
            "M001",
            "M002",
            "M003",
            "M004",
            "M005"
        }
    )
)

VALID_STATES = set(
    getattr(
        C,
        "VALID_STATES",
        {
            "RUNNING",
            "IDLE",
            "OFF",
            "MAINTENANCE"
        }
    )
)

CLEANED_FILE = (
    CLEANED_DATA_DIR
    / "manufacturing_energy_cleaned.csv"
)

VALIDATION_FILE = (
    VALIDATION_DIR
    / "validation_issues.csv"
)

CLEANING_SUMMARY_FILE = (
    VALIDATION_DIR
    / "cleaning_summary.csv"
)

SCHEDULE_CLASSIFICATION_FILE = (
    VALIDATION_DIR
    / "schedule_deviation_classification.csv"
)

QUALITY_BEFORE_FILE = (
    VALIDATION_DIR
    / "data_quality_before.csv"
)

QUALITY_AFTER_FILE = (
    VALIDATION_DIR
    / "data_quality_after.csv"
)

FINAL_VALIDATION_FILE = (
    VALIDATION_DIR
    / "final_validation_results.csv"
)


# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)

logger = logging.getLogger("data_cleaning")


# ============================================================================
# REQUIRED SCHEMA
# ============================================================================

REQUIRED_COLUMNS = [
    "Record_ID",
    "Timestamp",
    "Date",
    "Time",
    "Machine_ID",
    "Machine_Name",
    "Machine_Type",
    "Planned_State",
    "Actual_State",
    "State_Deviation",
    "Running_Hours",
    "Idle_Hours",
    "Maintenance_Flag",
    "Planned_Production_Units",
    "Actual_Production_Units",
    "Production_Variance",
    "Production_Efficiency_Percent",
    "Energy_Consumption_kWh",
    "Baseline_Energy_kWh",
    "Energy_Deviation_kWh",
    "Energy_Deviation_Percent",
    "Energy_Per_Unit_kWh",
    "Idle_Energy_kWh",
    "Peak_Load_Flag",
    "High_Consumption_Flag",
    "Ambient_Temperature_C",
    "Relative_Humidity_Percent",
    "CO2e_kg",
    "CO2e_Baseline_kg",
    "Potential_CO2_Reduction_kg",
    "Known_Condition_Flag"
]


NUMERIC_COLUMNS = [
    "Running_Hours",
    "Idle_Hours",
    "Maintenance_Flag",
    "Planned_Production_Units",
    "Actual_Production_Units",
    "Production_Variance",
    "Production_Efficiency_Percent",
    "Energy_Consumption_kWh",
    "Baseline_Energy_kWh",
    "Energy_Deviation_kWh",
    "Energy_Deviation_Percent",
    "Energy_Per_Unit_kWh",
    "Idle_Energy_kWh",
    "Ambient_Temperature_C",
    "Relative_Humidity_Percent",
    "CO2e_kg",
    "CO2e_Baseline_kg",
    "Potential_CO2_Reduction_kg"
]


TEXT_COLUMNS = [
    "Record_ID",
    "Machine_ID",
    "Machine_Name",
    "Machine_Type",
    "Planned_State",
    "Actual_State",
    "State_Deviation",
    "Peak_Load_Flag",
    "High_Consumption_Flag",
    "Known_Condition_Flag"
]


# ============================================================================
# DIRECTORY MANAGEMENT
# ============================================================================

def initialize_directories():
    """Create all directories required by the cleaning pipeline."""

    CLEANED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    VALIDATION_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    logger.info(
        "Cleaning and validation directories initialized."
    )


# ============================================================================
# DATA LOADING
# ============================================================================

def load_raw_dataset():
    """Load the original manufacturing Excel dataset."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Raw dataset not found:\n{INPUT_FILE}"
        )

    logger.info(
        "Loading raw manufacturing dataset..."
    )

    logger.info(
        "Source: %s",
        INPUT_FILE
    )

    df = load_data()

    logger.info(
        "Raw dataset loaded: %s rows × %s columns",
        df.shape[0],
        df.shape[1]
    )

    return df


# ============================================================================
# DATA TYPE STANDARDIZATION
# ============================================================================

def standardize_data_types(df):
    """Convert dataset columns into appropriate analytical types."""

    logger.info(
        "Standardizing data types..."
    )

    df = df.copy()

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        errors="coerce"
    )

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    for column in NUMERIC_COLUMNS:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    for column in TEXT_COLUMNS:

        if column in df.columns:

            df[column] = (
                df[column]
                .astype("string")
                .str.strip()
            )

    logger.info(
        "Data types standardized successfully."
    )

    return df


# ============================================================================
# SCHEMA VALIDATION
# ============================================================================

def validate_schema(df):
    """Verify that the complete expected schema exists."""

    logger.info(
        "Validating dataset schema..."
    )

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Required columns are missing:\n"
            + "\n".join(missing_columns)
        )

    logger.info(
        "Schema validation passed."
    )


# ============================================================================
# RECORD COUNT VALIDATION
# ============================================================================

def validate_record_count(df):
    """Verify that exactly 10,000 records are available."""

    logger.info(
        "Validating record count..."
    )

    if len(df) != EXPECTED_RECORDS:

        raise ValueError(
            f"Expected exactly {EXPECTED_RECORDS:,} records, "
            f"but found {len(df):,}."
        )

    logger.info(
        "Record count validated: %s",
        f"{len(df):,}"
    )


# ============================================================================
# DUPLICATE VALIDATION
# ============================================================================

def validate_duplicates(df, issues):
    """Identify duplicate rows and duplicate Record_ID values."""

    logger.info(
        "Checking duplicate records..."
    )

    duplicate_rows = df.duplicated(
        keep=False
    )

    duplicate_ids = df["Record_ID"].duplicated(
        keep=False
    )

    duplicate_row_count = int(
        duplicate_rows.sum()
    )

    duplicate_id_count = int(
        duplicate_ids.sum()
    )

    logger.info(
        "Duplicate rows: %s",
        duplicate_row_count
    )

    logger.info(
        "Duplicate Record IDs: %s",
        duplicate_id_count
    )

    for index in df.index[duplicate_rows]:

        issues.append({
            "Record_ID": df.at[index, "Record_ID"],
            "Machine_ID": df.at[index, "Machine_ID"],
            "Timestamp": df.at[index, "Timestamp"],
            "Column_Name": "Record_ID",
            "Issue_Type": "Duplicate_Record",
            "Severity": "High",
            "Recommended_Action": (
                "Investigate duplicate record"
            )
        })

    for index in df.index[duplicate_ids]:

        issues.append({
            "Record_ID": df.at[index, "Record_ID"],
            "Machine_ID": df.at[index, "Machine_ID"],
            "Timestamp": df.at[index, "Timestamp"],
            "Column_Name": "Record_ID",
            "Issue_Type": "Duplicate_Record_ID",
            "Severity": "High",
            "Recommended_Action": (
                "Investigate duplicate identifier"
            )
        })


# ============================================================================
# TIMESTAMP VALIDATION
# ============================================================================

def validate_timestamps(df, issues):
    """Validate timestamps and Date/Timestamp consistency."""

    logger.info(
        "Validating timestamps..."
    )

    invalid_timestamp = df["Timestamp"].isna()

    for index in df.index[invalid_timestamp]:

        issues.append({
            "Record_ID": df.at[index, "Record_ID"],
            "Machine_ID": df.at[index, "Machine_ID"],
            "Timestamp": None,
            "Column_Name": "Timestamp",
            "Issue_Type": "Invalid_Timestamp",
            "Severity": "Critical",
            "Recommended_Action": (
                "Investigate timestamp"
            )
        })

    valid_timestamp = ~invalid_timestamp

    date_mismatch = (
        valid_timestamp
        &
        (
            df.loc[
                valid_timestamp,
                "Timestamp"
            ].dt.date
            !=
            df.loc[
                valid_timestamp,
                "Date"
            ].dt.date
        )
    )

    mismatch_indexes = (
        df.loc[
            valid_timestamp
        ]
        .index[date_mismatch]
    )

    for index in mismatch_indexes:

        issues.append({
            "Record_ID": df.at[index, "Record_ID"],
            "Machine_ID": df.at[index, "Machine_ID"],
            "Timestamp": df.at[index, "Timestamp"],
            "Column_Name": "Date",
            "Issue_Type": "Date_Timestamp_Mismatch",
            "Severity": "Medium",
            "Recommended_Action": (
                "Derive Date from Timestamp"
            )
        })

    logger.info(
        "Invalid timestamps: %s",
        int(invalid_timestamp.sum())
    )

    logger.info(
        "Date/timestamp mismatches: %s",
        len(mismatch_indexes)
    )


# ============================================================================
# MACHINE IDENTIFIER VALIDATION
# ============================================================================

def validate_machine_ids(df, issues):
    """Validate Machine_ID values against the machine master."""

    logger.info(
        "Validating machine identifiers..."
    )

    invalid_machines = (
        ~df["Machine_ID"].isin(
            EXPECTED_MACHINES
        )
    )

    for index in df.index[invalid_machines]:

        issues.append({
            "Record_ID": df.at[index, "Record_ID"],
            "Machine_ID": df.at[index, "Machine_ID"],
            "Timestamp": df.at[index, "Timestamp"],
            "Column_Name": "Machine_ID",
            "Issue_Type": "Invalid_Machine_ID",
            "Severity": "Critical",
            "Recommended_Action": (
                "Validate machine master mapping"
            )
        })

    logger.info(
        "Invalid machine identifiers: %s",
        int(invalid_machines.sum())
    )


# ============================================================================
# MACHINE STATE VALIDATION
# ============================================================================

def validate_states(df, issues):
    """Validate Planned_State and Actual_State values."""

    logger.info(
        "Validating machine states..."
    )

    for state_column in [
        "Planned_State",
        "Actual_State"
    ]:

        invalid_states = (
            ~df[state_column].isin(
                VALID_STATES
            )
        )

        for index in df.index[invalid_states]:

            issues.append({
                "Record_ID": df.at[index, "Record_ID"],
                "Machine_ID": df.at[index, "Machine_ID"],
                "Timestamp": df.at[index, "Timestamp"],
                "Column_Name": state_column,
                "Issue_Type": "Invalid_State",
                "Severity": "High",
                "Recommended_Action": (
                    "Validate machine-state value"
                )
            })

        logger.info(
            "%s invalid states: %s",
            state_column,
            int(invalid_states.sum())
        )


# ============================================================================
# ENVIRONMENTAL DATA IMPUTATION
# ============================================================================

def impute_environmental_data(df, issues):
    """
    Handle missing environmental measurements using time-series
    interpolation followed by directional filling where necessary.
    """

    logger.info(
        "Processing environmental sensor data..."
    )

    df = df.copy()

    environmental_columns = [
        "Ambient_Temperature_C",
        "Relative_Humidity_Percent"
    ]

    for column in environmental_columns:

        missing_before = int(
            df[column].isna().sum()
        )

        if missing_before == 0:
            continue

        logger.info(
            "%s missing values found in %s.",
            missing_before,
            column
        )

        missing_indexes = df.index[
            df[column].isna()
        ]

        for index in missing_indexes:

            issues.append({
                "Record_ID": df.at[index, "Record_ID"],
                "Machine_ID": df.at[index, "Machine_ID"],
                "Timestamp": df.at[index, "Timestamp"],
                "Column_Name": column,
                "Issue_Type": "Missing_Environmental_Value",
                "Severity": "Low",
                "Recommended_Action": (
                    "Time-series interpolation"
                )
            })

        ordered = (
            df.sort_values(
                "Timestamp"
            )
            .copy()
        )

        ordered[column] = (
            ordered[column]
            .interpolate(
                method="linear",
                limit_direction="both"
            )
        )

        df.loc[
            ordered.index,
            column
        ] = ordered[column]

        missing_after = int(
            df[column].isna().sum()
        )

        logger.info(
            "%s: %s missing before → %s after",
            column,
            missing_before,
            missing_after
        )

    return df


# ============================================================================
# DATE RECONCILIATION
# ============================================================================

def reconcile_date_column(df, issues):
    """Derive Date from Timestamp where inconsistencies exist."""

    logger.info(
        "Reconciling Date with Timestamp..."
    )

    valid_timestamp = (
        df["Timestamp"].notna()
    )

    expected_dates = pd.Series(
        pd.NaT,
        index=df.index
    )

    expected_dates.loc[
        valid_timestamp
    ] = (
        df.loc[
            valid_timestamp,
            "Timestamp"
        ].dt.normalize()
    )

    mismatch = (
        valid_timestamp
        &
        (
            df["Date"]
            !=
            expected_dates
        )
    )

    mismatch_count = int(
        mismatch.sum()
    )

    if mismatch_count > 0:

        logger.warning(
            "Correcting %s Date/Timestamp mismatches.",
            mismatch_count
        )

        for index in df.index[mismatch]:

            issues.append({
                "Record_ID": df.at[index, "Record_ID"],
                "Machine_ID": df.at[index, "Machine_ID"],
                "Timestamp": df.at[index, "Timestamp"],
                "Column_Name": "Date",
                "Issue_Type": "Date_Corrected",
                "Severity": "Medium",
                "Recommended_Action": (
                    "Date derived from Timestamp"
                )
            })

        df.loc[
            mismatch,
            "Date"
        ] = expected_dates.loc[
            mismatch
        ]

    return df


# ============================================================================
# PRODUCTION / STATE VALIDATION
# ============================================================================

def validate_production_state_consistency(
    df,
    issues
):
    """Identify production values inconsistent with machine state."""

    logger.info(
        "Validating production-state consistency..."
    )

    zero_production_states = {
        "IDLE",
        "OFF",
        "MAINTENANCE"
    }

    inconsistent = (
        df["Actual_State"].isin(
            zero_production_states
        )
        &
        (
            df["Actual_Production_Units"]
            > 0
        )
    )

    count = int(
        inconsistent.sum()
    )

    for index in df.index[inconsistent]:

        issues.append({
            "Record_ID": df.at[index, "Record_ID"],
            "Machine_ID": df.at[index, "Machine_ID"],
            "Timestamp": df.at[index, "Timestamp"],
            "Column_Name": "Actual_Production_Units",
            "Issue_Type": "Production_State_Inconsistency",
            "Severity": "Medium",
            "Recommended_Action": (
                "Investigate production reading"
            )
        })

    logger.info(
        "Production-state inconsistencies: %s",
        count
    )


# ============================================================================
# ENERGY / STATE VALIDATION
# ============================================================================

def validate_energy_state_consistency(
    df,
    issues
):
    """Validate energy and idle-energy measurements."""

    logger.info(
        "Validating energy-state consistency..."
    )

    negative_energy = (
        df["Energy_Consumption_kWh"]
        < 0
    )

    negative_idle_energy = (
        df["Idle_Energy_kWh"]
        < 0
    )

    for index in df.index[negative_energy]:

        issues.append({
            "Record_ID": df.at[index, "Record_ID"],
            "Machine_ID": df.at[index, "Machine_ID"],
            "Timestamp": df.at[index, "Timestamp"],
            "Column_Name": "Energy_Consumption_kWh",
            "Issue_Type": "Negative_Energy",
            "Severity": "Critical",
            "Recommended_Action": (
                "Investigate energy measurement"
            )
        })

    for index in df.index[negative_idle_energy]:

        issues.append({
            "Record_ID": df.at[index, "Record_ID"],
            "Machine_ID": df.at[index, "Machine_ID"],
            "Timestamp": df.at[index, "Timestamp"],
            "Column_Name": "Idle_Energy_kWh",
            "Issue_Type": "Negative_Idle_Energy",
            "Severity": "High",
            "Recommended_Action": (
                "Investigate idle-energy measurement"
            )
        })

    logger.info(
        "Negative energy observations: %s",
        int(negative_energy.sum())
    )

    logger.info(
        "Negative idle-energy observations: %s",
        int(negative_idle_energy.sum())
    )


# ============================================================================
# ENVIRONMENTAL RANGE VALIDATION
# ============================================================================

def validate_environmental_ranges(
    df,
    issues
):
    """Validate environmental measurements against broad ranges."""

    logger.info(
        "Validating environmental ranges..."
    )

    temperature_invalid = (
        (df["Ambient_Temperature_C"] < -30)
        |
        (df["Ambient_Temperature_C"] > 60)
    )

    humidity_invalid = (
        (df["Relative_Humidity_Percent"] < 0)
        |
        (df["Relative_Humidity_Percent"] > 100)
    )

    for index in df.index[temperature_invalid]:

        issues.append({
            "Record_ID": df.at[index, "Record_ID"],
            "Machine_ID": df.at[index, "Machine_ID"],
            "Timestamp": df.at[index, "Timestamp"],
            "Column_Name": "Ambient_Temperature_C",
            "Issue_Type": "Temperature_Out_of_Range",
            "Severity": "Medium",
            "Recommended_Action": (
                "Investigate environmental sensor"
            )
        })

    for index in df.index[humidity_invalid]:

        issues.append({
            "Record_ID": df.at[index, "Record_ID"],
            "Machine_ID": df.at[index, "Machine_ID"],
            "Timestamp": df.at[index, "Timestamp"],
            "Column_Name": "Relative_Humidity_Percent",
            "Issue_Type": "Humidity_Out_of_Range",
            "Severity": "Medium",
            "Recommended_Action": (
                "Investigate environmental sensor"
            )
        })

    logger.info(
        "Temperature range violations: %s",
        int(temperature_invalid.sum())
    )

    logger.info(
        "Humidity range violations: %s",
        int(humidity_invalid.sum())
    )


# ============================================================================
# NON-NEGATIVE VALIDATION
# ============================================================================

def validate_non_negative_metrics(
    df,
    issues
):
    """Validate analytical fields that cannot logically be negative."""

    logger.info(
        "Validating non-negative analytical metrics..."
    )

    columns = [
        "Energy_Consumption_kWh",
        "Baseline_Energy_kWh",
        "Idle_Energy_kWh",
        "Actual_Production_Units",
        "Planned_Production_Units",
        "CO2e_kg",
        "CO2e_Baseline_kg"
    ]

    for column in columns:

        negative = (
            df[column] < 0
        )

        count = int(
            negative.sum()
        )

        if count == 0:
            continue

        for index in df.index[negative]:

            issues.append({
                "Record_ID": df.at[index, "Record_ID"],
                "Machine_ID": df.at[index, "Machine_ID"],
                "Timestamp": df.at[index, "Timestamp"],
                "Column_Name": column,
                "Issue_Type": "Negative_Value",
                "Severity": "High",
                "Recommended_Action": (
                    "Investigate source measurement"
                )
            })

        logger.warning(
            "%s negative values found in %s.",
            count,
            column
        )


# ============================================================================
# ENERGY CALCULATION VALIDATION
# ============================================================================

def validate_energy_calculations(
    df,
    issues
):
    """Validate Energy_Deviation_kWh mathematically."""

    logger.info(
        "Validating energy calculations..."
    )

    expected_deviation = (
        df["Energy_Consumption_kWh"]
        -
        df["Baseline_Energy_kWh"]
    )

    difference = (
        df["Energy_Deviation_kWh"]
        -
        expected_deviation
    ).abs()

    mismatch = (
        difference > 0.01
    )

    for index in df.index[mismatch]:

        issues.append({
            "Record_ID": df.at[index, "Record_ID"],
            "Machine_ID": df.at[index, "Machine_ID"],
            "Timestamp": df.at[index, "Timestamp"],
            "Column_Name": "Energy_Deviation_kWh",
            "Issue_Type": "Energy_Deviation_Calculation_Mismatch",
            "Severity": "High",
            "Recommended_Action": (
                "Recalculate energy deviation"
            )
        })

    logger.info(
        "Energy calculation mismatches: %s",
        int(mismatch.sum())
    )


# ============================================================================
# CO2e CALCULATION VALIDATION
# ============================================================================

def validate_co2_calculations(
    df,
    issues
):
    """Validate CO2e calculations using the implied synthetic factor."""

    logger.info(
        "Validating CO2e calculations..."
    )

    valid_rows = (
        df["Energy_Consumption_kWh"].notna()
        &
        df["CO2e_kg"].notna()
        &
        (
            df["Energy_Consumption_kWh"]
            > 0
        )
    )

    if valid_rows.sum() == 0:

        logger.warning(
            "No valid rows available for CO2e validation."
        )

        return

    implied_factor = (
        df.loc[
            valid_rows,
            "CO2e_kg"
        ]
        /
        df.loc[
            valid_rows,
            "Energy_Consumption_kWh"
        ]
    )

    reference_factor = (
        implied_factor.median()
    )

    expected_co2 = (
        df["Energy_Consumption_kWh"]
        *
        reference_factor
    )

    difference = (
        df["CO2e_kg"]
        -
        expected_co2
    ).abs()

    mismatch = (
        difference > 0.05
    )

    mismatch &= valid_rows

    for index in df.index[mismatch]:

        issues.append({
            "Record_ID": df.at[index, "Record_ID"],
            "Machine_ID": df.at[index, "Machine_ID"],
            "Timestamp": df.at[index, "Timestamp"],
            "Column_Name": "CO2e_kg",
            "Issue_Type": "CO2e_Calculation_Mismatch",
            "Severity": "Medium",
            "Recommended_Action": (
                "Verify CO2e calculation and project parameter"
            )
        })

    logger.info(
        "Implied synthetic emission factor: %.6f",
        reference_factor
    )

    logger.info(
        "CO2e calculation mismatches: %s",
        int(mismatch.sum())
    )


# ============================================================================
# SCHEDULE DEVIATION CLASSIFICATION
# ============================================================================

def classify_schedule_deviations(df):
    """
    Classify planned-versus-actual machine-state deviations.

    The classification is performed using vectorized pandas operations
    so the function remains efficient, readable, and scalable.
    """

    logger.info(
        "Classifying schedule deviations..."
    )

    df = df.copy()

    planned = df["Planned_State"]
    actual = df["Actual_State"]

    df["Schedule_Deviation_Category"] = (
        "NORMAL_OPERATION"
    )

    df.loc[
        (planned == "RUNNING") & (actual == "IDLE"),
        "Schedule_Deviation_Category"
    ] = "POTENTIAL_IDLE_OPPORTUNITY"

    df.loc[
        (planned == "RUNNING") & (actual == "OFF"),
        "Schedule_Deviation_Category"
    ] = "POTENTIAL_EARLY_SHUTDOWN"

    df.loc[
        (planned == "RUNNING") & (actual == "MAINTENANCE"),
        "Schedule_Deviation_Category"
    ] = "MAINTENANCE_DEVIATION"

    df.loc[
        (planned == "IDLE") & (actual == "RUNNING"),
        "Schedule_Deviation_Category"
    ] = "POTENTIAL_UNSCHEDULED_OPERATION"

    df.loc[
        (planned == "IDLE") & (actual == "OFF"),
        "Schedule_Deviation_Category"
    ] = "LOW_IMPACT_SHUTDOWN_DEVIATION"

    df.loc[
        (planned == "IDLE") & (actual == "MAINTENANCE"),
        "Schedule_Deviation_Category"
    ] = "MAINTENANCE_DURING_IDLE"

    logger.info(
        "Schedule deviation classification completed."
    )

    return df


# ============================================================================
# SCHEDULE CLASSIFICATION REPORT
# ============================================================================

def create_schedule_classification_report(df):
    """Generate schedule-deviation summary statistics."""

    logger.info(
        "Creating schedule deviation classification report..."
    )

    deviations = df[
        df["Planned_State"]
        !=
        df["Actual_State"]
    ].copy()

    if deviations.empty:

        summary = pd.DataFrame(
            columns=[
                "Schedule_Deviation_Category",
                "Records",
                "Energy_kWh",
                "Production_Units",
                "Idle_Energy_kWh"
            ]
        )

    else:

        summary = (
            deviations
            .groupby(
                "Schedule_Deviation_Category",
                as_index=False
            )
            .agg(
                Records=(
                    "Record_ID",
                    "count"
                ),
                Energy_kWh=(
                    "Energy_Consumption_kWh",
                    "sum"
                ),
                Production_Units=(
                    "Actual_Production_Units",
                    "sum"
                ),
                Idle_Energy_kWh=(
                    "Idle_Energy_kWh",
                    "sum"
                )
            )
            .sort_values(
                "Records",
                ascending=False
            )
        )

    summary.to_csv(
        SCHEDULE_CLASSIFICATION_FILE,
        index=False
    )

    logger.info(
        "Schedule classification report saved."
    )

    return summary


# ============================================================================
# DATA QUALITY REPORT
# ============================================================================

def create_quality_report(df):
    """Generate column-level data-quality statistics."""

    return pd.DataFrame({
        "Column": df.columns,
        "Data_Type": [
            str(dtype)
            for dtype in df.dtypes
        ],
        "Missing_Count": [
            int(
                df[column].isna().sum()
            )
            for column in df.columns
        ],
        "Missing_Percentage": [
            round(
                df[column].isna().mean() * 100,
                4
            )
            for column in df.columns
        ],
        "Unique_Values": [
            int(
                df[column].nunique()
            )
            for column in df.columns
        ]
    })


def save_quality_report(
    quality,
    file_path
):
    """Save a quality report to CSV."""

    quality.to_csv(
        file_path,
        index=False
    )


# ============================================================================
# VALIDATION ISSUE REPORT
# ============================================================================

def save_validation_issues(issues):
    """Save all identified validation issues."""

    if issues:

        validation_df = pd.DataFrame(
            issues
        )

    else:

        validation_df = pd.DataFrame(
            columns=[
                "Record_ID",
                "Machine_ID",
                "Timestamp",
                "Column_Name",
                "Issue_Type",
                "Severity",
                "Recommended_Action"
            ]
        )

    validation_df.insert(
        0,
        "Issue_ID",
        [
            f"ISSUE{i:05d}"
            for i in range(
                1,
                len(validation_df) + 1
            )
        ]
    )

    validation_df.to_csv(
        VALIDATION_FILE,
        index=False
    )

    return validation_df


# ============================================================================
# FINAL VALIDATION
# ============================================================================

def final_validation(df):
    """Perform final validation checks after cleaning."""

    logger.info(
        "Running final validation..."
    )

    results = []

    duplicate_ids = int(
        df["Record_ID"]
        .duplicated()
        .sum()
    )

    invalid_machines = int(
        (
            ~df["Machine_ID"].isin(
                EXPECTED_MACHINES
            )
        ).sum()
    )

    invalid_timestamps = int(
        df["Timestamp"].isna().sum()
    )

    negative_energy = int(
        (
            df["Energy_Consumption_kWh"]
            < 0
        ).sum()
    )

    negative_production = int(
        (
            df["Actual_Production_Units"]
            < 0
        ).sum()
    )

    results.extend([
        {
            "Validation": "Record Count",
            "Expected": EXPECTED_RECORDS,
            "Actual": len(df),
            "Status": (
                "PASS"
                if len(df) == EXPECTED_RECORDS
                else "FAIL"
            )
        },
        {
            "Validation": "Duplicate Record IDs",
            "Expected": 0,
            "Actual": duplicate_ids,
            "Status": (
                "PASS"
                if duplicate_ids == 0
                else "FAIL"
            )
        },
        {
            "Validation": "Invalid Machine IDs",
            "Expected": 0,
            "Actual": invalid_machines,
            "Status": (
                "PASS"
                if invalid_machines == 0
                else "FAIL"
            )
        },
        {
            "Validation": "Invalid Timestamps",
            "Expected": 0,
            "Actual": invalid_timestamps,
            "Status": (
                "PASS"
                if invalid_timestamps == 0
                else "FAIL"
            )
        },
        {
            "Validation": "Negative Energy Values",
            "Expected": 0,
            "Actual": negative_energy,
            "Status": (
                "PASS"
                if negative_energy == 0
                else "FAIL"
            )
        },
        {
            "Validation": "Negative Production Values",
            "Expected": 0,
            "Actual": negative_production,
            "Status": (
                "PASS"
                if negative_production == 0
                else "FAIL"
            )
        }
    ])

    return pd.DataFrame(
        results
    )


# ============================================================================
# CLEANING SUMMARY
# ============================================================================

def create_cleaning_summary(
    raw_df,
    clean_df,
    validation_df,
    quality_before,
    quality_after
):
    """Create an executive cleaning-process summary."""

    severity_counts = (
        validation_df["Severity"]
        .value_counts()
        if not validation_df.empty
        else pd.Series(dtype=int)
    )

    missing_before = int(
        quality_before[
            "Missing_Count"
        ].sum()
    )

    missing_after = int(
        quality_after[
            "Missing_Count"
        ].sum()
    )

    summary = pd.DataFrame({
        "Metric": [
            "Raw Records",
            "Cleaned Records",
            "Raw Columns",
            "Cleaned Columns",
            "Missing Values Before",
            "Missing Values After",
            "Total Validation Issues",
            "Critical Issues",
            "High Severity Issues",
            "Medium Severity Issues",
            "Low Severity Issues",
            "Duplicate Record IDs After Cleaning",
            "Cleaning Timestamp"
        ],
        "Value": [
            len(raw_df),
            len(clean_df),
            len(raw_df.columns),
            len(clean_df.columns),
            missing_before,
            missing_after,
            len(validation_df),
            int(
                severity_counts.get(
                    "Critical",
                    0
                )
            ),
            int(
                severity_counts.get(
                    "High",
                    0
                )
            ),
            int(
                severity_counts.get(
                    "Medium",
                    0
                )
            ),
            int(
                severity_counts.get(
                    "Low",
                    0
                )
            ),
            int(
                clean_df[
                    "Record_ID"
                ].duplicated().sum()
            ),
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        ]
    })

    summary.to_csv(
        CLEANING_SUMMARY_FILE,
        index=False
    )

    return summary


# ============================================================================
# SAVE CLEANED DATASET
# ============================================================================

def save_cleaned_dataset(df):
    """Save the cleaned analytical dataset."""

    logger.info(
        "Saving cleaned analytical dataset..."
    )

    df.to_csv(
        CLEANED_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    logger.info(
        "Cleaned dataset saved: %s",
        CLEANED_FILE
    )


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Execute the complete data-cleaning and validation workflow."""

    logger.info("=" * 78)

    logger.info(
        "SUSTAINABLE MANUFACTURING - DATA CLEANING & VALIDATION"
    )

    logger.info("=" * 78)

    try:

        # Step 1: Initialize directories.
        initialize_directories()

        # Step 2: Load raw data.
        raw_df = load_raw_dataset()

        df = raw_df.copy()

        # Step 3: Standardize analytical data types.
        df = standardize_data_types(
            df
        )

        # Step 4: Validate schema and record count.
        validate_schema(
            df
        )

        validate_record_count(
            df
        )

        # Step 5: Capture data quality before cleaning.
        quality_before = create_quality_report(
            df
        )

        save_quality_report(
            quality_before,
            QUALITY_BEFORE_FILE
        )

        # Step 6: Initialize validation issue log.
        issues = []

        # Step 7: Structural validation.
        validate_duplicates(
            df,
            issues
        )

        validate_timestamps(
            df,
            issues
        )

        validate_machine_ids(
            df,
            issues
        )

        validate_states(
            df,
            issues
        )

        # Step 8: Reconcile Date against Timestamp.
        df = reconcile_date_column(
            df,
            issues
        )

        # Step 9: Treat environmental missing values.
        df = impute_environmental_data(
            df,
            issues
        )

        # Step 10: Operational consistency validation.
        validate_production_state_consistency(
            df,
            issues
        )

        validate_energy_state_consistency(
            df,
            issues
        )

        validate_environmental_ranges(
            df,
            issues
        )

        validate_non_negative_metrics(
            df,
            issues
        )

        # Step 11: Mathematical validation.
        validate_energy_calculations(
            df,
            issues
        )

        validate_co2_calculations(
            df,
            issues
        )

        # Step 12: Schedule classification.
        df = classify_schedule_deviations(
            df
        )

        schedule_summary = (
            create_schedule_classification_report(
                df
            )
        )

        # Step 13: Save validation issues.
        validation_df = save_validation_issues(
            issues
        )

        # Step 14: Capture data quality after cleaning.
        quality_after = create_quality_report(
            df
        )

        save_quality_report(
            quality_after,
            QUALITY_AFTER_FILE
        )

        # Step 15: Final validation.
        final_results = final_validation(
            df
        )

        final_results.to_csv(
            FINAL_VALIDATION_FILE,
            index=False
        )

        # Step 16: Save cleaned dataset.
        save_cleaned_dataset(
            df
        )

        # Step 17: Create cleaning summary.
        cleaning_summary = (
            create_cleaning_summary(
                raw_df=raw_df,
                clean_df=df,
                validation_df=validation_df,
                quality_before=quality_before,
                quality_after=quality_after
            )
        )

        # Step 18: Determine final status.
        validation_failures = int(
            (
                final_results["Status"]
                == "FAIL"
            ).sum()
        )

        logger.info("=" * 78)

        if validation_failures == 0:

            logger.info(
                "DATA CLEANING & VALIDATION COMPLETED SUCCESSFULLY"
            )

        else:

            logger.warning(
                "DATA CLEANING COMPLETED WITH "
                "VALIDATION FAILURES: %s",
                validation_failures
            )

        logger.info("=" * 78)

        logger.info(
            "Raw records             : %s",
            len(raw_df)
        )

        logger.info(
            "Cleaned records         : %s",
            len(df)
        )

        logger.info(
            "Missing values before   : %s",
            int(
                quality_before[
                    "Missing_Count"
                ].sum()
            )
        )

        logger.info(
            "Missing values after    : %s",
            int(
                quality_after[
                    "Missing_Count"
                ].sum()
            )
        )

        logger.info(
            "Validation issues       : %s",
            len(validation_df)
        )

        logger.info(
            "Schedule classifications: %s",
            len(schedule_summary)
        )

        logger.info(
            "Cleaned dataset         : %s",
            CLEANED_FILE
        )

        logger.info(
            "Validation directory    : %s",
            VALIDATION_DIR
        )

        print()
        print("=" * 78)
        print("DATA CLEANING EXECUTIVE SUMMARY")
        print("=" * 78)
        print(
            cleaning_summary.to_string(
                index=False
            )
        )

        print()
        print("=" * 78)
        print("FINAL VALIDATION")
        print("=" * 78)
        print(
            final_results.to_string(
                index=False
            )
        )

        print()
        print(
            "Data cleaning and validation completed."
        )

        print(
            f"Cleaned dataset: {CLEANED_FILE}"
        )

        print("=" * 78)

    except FileNotFoundError as error:

        logger.error(
            "File error: %s",
            error
        )

        sys.exit(1)

    except ValueError as error:

        logger.error(
            "Validation error: %s",
            error
        )

        sys.exit(1)

    except Exception as error:

        logger.exception(
            "Unexpected error during cleaning: %s",
            error
        )

        sys.exit(1)


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()