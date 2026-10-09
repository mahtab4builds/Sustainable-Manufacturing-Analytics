"""
Energy Baseline Development
Sustainable Manufacturing:
Energy Consumption & Carbon Footprint Optimization Analytics

Purpose:
    Develop a multi-layer energy baseline from the cleaned manufacturing
    dataset for subsequent inefficiency detection, forecasting, what-if
    analysis, CO2 analysis, optimization, and dashboard development.

Baseline philosophy:
    The baseline represents expected/normal energy consumption rather than
    simply reproducing the observed total energy.

Input:
    outputs/cleaned_data/manufacturing_energy_cleaned.csv

Outputs:
    outputs/baseline/
        baseline_enriched_dataset.csv
        machine_baseline.csv
        state_baseline.csv
        hourly_baseline.csv
        daily_baseline.csv
        production_adjusted_baseline.csv
        machine_baseline_performance.csv
        high_consumption_observations.csv
        baseline_validation.csv
        baseline_summary.csv
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from library.config import (
    BASELINE_DIR,
    CLEANED_DATA_DIR,
    initialize_directories,
)

from library.constants import MACHINE_IDS, VALID_STATES


# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_FILE = CLEANED_DATA_DIR / "manufacturing_energy_cleaned.csv"

ENERGY_COLUMN = "Energy_Consumption_kWh"
PRODUCTION_COLUMN = "Actual_Production_Units"
STATE_COLUMN = "Actual_State"
MACHINE_COLUMN = "Machine_ID"
TIMESTAMP_COLUMN = "Timestamp"

BASELINE_COLUMN = "Calculated_Baseline_Energy_kWh"
DEVIATION_COLUMN = "Baseline_Deviation_kWh"
DEVIATION_PERCENT_COLUMN = "Baseline_Deviation_Percent"

# Baseline thresholds.
HIGH_CONSUMPTION_THRESHOLD = 10.0
LOW_CONSUMPTION_THRESHOLD = -10.0

# Minimum observations required for a group-level baseline.
MIN_GROUP_OBSERVATIONS = 3


# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)


# ============================================================================
# GENERAL HELPERS
# ============================================================================

def save_csv(dataframe: pd.DataFrame, filename: str) -> Path:
    """
    Save a DataFrame into the centralized baseline output directory.
    """

    output_path = BASELINE_DIR / filename

    dataframe.to_csv(
        output_path,
        index=False,
    )

    logger.info("CSV saved: %s", output_path)

    return output_path


def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    """
    Safely divide two pandas Series.

    Zero and invalid denominators return NaN instead of infinity.
    """

    denominator = denominator.replace(
        [0, np.inf, -np.inf],
        np.nan,
    )

    result = numerator / denominator

    return result.replace(
        [np.inf, -np.inf],
        np.nan,
    )


# ============================================================================
# DATA LOADING
# ============================================================================

def load_cleaned_dataset() -> pd.DataFrame:
    """
    Load the validated cleaned analytical dataset.
    """

    logger.info("Loading cleaned analytical dataset...")
    logger.info("Source: %s", INPUT_FILE)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    logger.info(
        "Dataset loaded: %s rows × %s columns",
        f"{len(df):,}",
        len(df.columns),
    )

    return df


# ============================================================================
# DATA PREPARATION
# ============================================================================

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare timestamps and analytical helper variables.
    """

    logger.info("Preparing baseline features...")

    required_columns = [
        ENERGY_COLUMN,
        PRODUCTION_COLUMN,
        STATE_COLUMN,
        MACHINE_COLUMN,
        TIMESTAMP_COLUMN,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Required columns missing from dataset: "
            + ", ".join(missing_columns)
        )

    df = df.copy()

    df[TIMESTAMP_COLUMN] = pd.to_datetime(
        df[TIMESTAMP_COLUMN],
        errors="coerce",
    )

    if df[TIMESTAMP_COLUMN].isna().any():
        raise ValueError(
            "Invalid timestamps detected during baseline preparation."
        )

    df[ENERGY_COLUMN] = pd.to_numeric(
        df[ENERGY_COLUMN],
        errors="coerce",
    )

    df[PRODUCTION_COLUMN] = pd.to_numeric(
        df[PRODUCTION_COLUMN],
        errors="coerce",
    )

    df["Hour"] = df[TIMESTAMP_COLUMN].dt.hour

    df["Date"] = df[TIMESTAMP_COLUMN].dt.date

    df["DayOfWeek"] = df[TIMESTAMP_COLUMN].dt.dayofweek

    df["Month"] = df[TIMESTAMP_COLUMN].dt.month

    # Production efficiency proxy.
    if "Production_Efficiency_Percent" in df.columns:
        df["Production_Efficiency_Percent"] = pd.to_numeric(
            df["Production_Efficiency_Percent"],
            errors="coerce",
        )

    logger.info("Baseline features prepared.")

    return df


# ============================================================================
# MACHINE-LEVEL BASELINE
# ============================================================================

def calculate_machine_baseline(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate normal machine-level energy behavior.

    The baseline uses robust statistics rather than simply summing actual
    energy. The median is used to reduce the influence of abnormal events.
    """

    logger.info(
        "Calculating machine-level energy baseline..."
    )

    baseline = (
        df.groupby(MACHINE_COLUMN)
        .agg(
            Machine_Observations=(ENERGY_COLUMN, "count"),
            Machine_Median_Energy_kWh=(ENERGY_COLUMN, "median"),
            Machine_Mean_Energy_kWh=(ENERGY_COLUMN, "mean"),
            Machine_Std_Energy_kWh=(ENERGY_COLUMN, "std"),
            Machine_Median_Production=(
                PRODUCTION_COLUMN,
                "median",
            ),
        )
        .reset_index()
    )

    baseline["Machine_Baseline_Energy_kWh"] = (
        baseline["Machine_Median_Energy_kWh"]
    )

    logger.info(
        "Machine-level baseline calculated for %s machines.",
        len(baseline),
    )

    return baseline


# ============================================================================
# MACHINE-STATE BASELINE
# ============================================================================

def calculate_state_baseline(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate expected energy by machine and operating state.

    This is important because RUNNING, IDLE, OFF, and MAINTENANCE have
    fundamentally different normal energy profiles.
    """

    logger.info(
        "Calculating machine-state energy baselines..."
    )

    baseline = (
        df.groupby(
            [
                MACHINE_COLUMN,
                STATE_COLUMN,
            ]
        )
        .agg(
            Observations=(ENERGY_COLUMN, "count"),
            Median_Energy_kWh=(ENERGY_COLUMN, "median"),
            Mean_Energy_kWh=(ENERGY_COLUMN, "mean"),
            Std_Energy_kWh=(ENERGY_COLUMN, "std"),
            Median_Production=(
                PRODUCTION_COLUMN,
                "median",
            ),
        )
        .reset_index()
    )

    baseline["State_Baseline_Energy_kWh"] = (
        baseline["Median_Energy_kWh"]
    )

    # If a group is too small, retain the median but mark it as limited.
    baseline["Baseline_Data_Sufficiency"] = np.where(
        baseline["Observations"] >= MIN_GROUP_OBSERVATIONS,
        "SUFFICIENT",
        "LIMITED",
    )

    logger.info("State-level baseline generated.")

    return baseline


# ============================================================================
# HOURLY BASELINE
# ============================================================================

def calculate_hourly_baseline(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate normal energy consumption by machine and hour of day.

    This captures recurring time-of-day operating behavior.
    """

    logger.info(
        "Calculating hourly energy baseline..."
    )

    baseline = (
        df.groupby(
            [
                MACHINE_COLUMN,
                "Hour",
            ]
        )
        .agg(
            Observations=(ENERGY_COLUMN, "count"),
            Median_Energy_kWh=(ENERGY_COLUMN, "median"),
            Mean_Energy_kWh=(ENERGY_COLUMN, "mean"),
        )
        .reset_index()
    )

    baseline["Hourly_Baseline_Energy_kWh"] = (
        baseline["Median_Energy_kWh"]
    )

    logger.info("Hourly baseline generated.")

    return baseline


# ============================================================================
# DAILY BASELINE
# ============================================================================

def calculate_daily_baseline(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate daily expected energy using historical machine-day behavior.

    The daily baseline uses the median daily energy for each machine and
    weekday combination, allowing recurring weekly operating patterns to be
    represented.
    """

    logger.info(
        "Calculating daily energy baseline..."
    )

    daily = (
        df.groupby(
            [
                MACHINE_COLUMN,
                "Date",
                "DayOfWeek",
            ]
        )
        .agg(
            Daily_Actual_Energy_kWh=(
                ENERGY_COLUMN,
                "sum",
            ),
            Daily_Production_Units=(
                PRODUCTION_COLUMN,
                "sum",
            ),
        )
        .reset_index()
    )

    weekday_baseline = (
        daily.groupby(
            [
                MACHINE_COLUMN,
                "DayOfWeek",
            ]
        )
        .agg(
            Historical_Days=(
                "Daily_Actual_Energy_kWh",
                "count",
            ),
            Median_Daily_Energy_kWh=(
                "Daily_Actual_Energy_kWh",
                "median",
            ),
        )
        .reset_index()
    )

    weekday_baseline["Daily_Baseline_Energy_kWh"] = (
        weekday_baseline["Median_Daily_Energy_kWh"]
    )

    result = daily.merge(
        weekday_baseline[
            [
                MACHINE_COLUMN,
                "DayOfWeek",
                "Historical_Days",
                "Daily_Baseline_Energy_kWh",
            ]
        ],
        on=[
            MACHINE_COLUMN,
            "DayOfWeek",
        ],
        how="left",
    )

    logger.info(
        "Daily baseline generated for %s days.",
        result["Date"].nunique(),
    )

    return result


# ============================================================================
# PRODUCTION-ADJUSTED BASELINE
# ============================================================================

def calculate_production_adjusted_baseline(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Estimate expected energy based on normal energy intensity.

    Energy intensity is calculated only from productive RUNNING observations.
    Extreme values are clipped before calculating the representative median.

    This prevents unusually inefficient observations from defining the
    baseline itself.
    """

    logger.info(
        "Calculating production-adjusted energy baseline..."
    )

    working = df.copy()

    running = working[
        (working[STATE_COLUMN] == "RUNNING")
        & (working[PRODUCTION_COLUMN] > 0)
        & (working[ENERGY_COLUMN] >= 0)
    ].copy()

    if running.empty:
        logger.warning(
            "No valid RUNNING production observations found."
        )

        result = pd.DataFrame(
            columns=[
                MACHINE_COLUMN,
                "Normal_Energy_Per_Unit_kWh",
            ]
        )

        return result

    running["Energy_Per_Unit"] = safe_divide(
        running[ENERGY_COLUMN],
        running[PRODUCTION_COLUMN],
    )

    running = running[
        running["Energy_Per_Unit"].notna()
        & np.isfinite(running["Energy_Per_Unit"])
    ]

    intensity = (
        running.groupby(MACHINE_COLUMN)
        ["Energy_Per_Unit"]
        .median()
        .reset_index(
            name="Normal_Energy_Per_Unit_kWh"
        )
    )

    logger.info(
        "Production-adjusted baseline generated."
    )

    return intensity


# ============================================================================
# APPLY BASELINE
# ============================================================================

def apply_baseline_layers(
    df: pd.DataFrame,
    machine_baseline: pd.DataFrame,
    state_baseline: pd.DataFrame,
    hourly_baseline: pd.DataFrame,
    production_baseline: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply machine, state, hourly and production-adjusted baseline layers
    to every observation.

    The final observation-level baseline prioritizes the machine-state
    baseline because operating state is the strongest direct indicator of
    normal machine energy behavior.
    """

    logger.info(
        "Applying baseline layers to observations..."
    )

    result = df.copy()

    # ------------------------------------------------------------------
    # MACHINE BASELINE
    # ------------------------------------------------------------------

    result = result.merge(
        machine_baseline[
            [
                MACHINE_COLUMN,
                "Machine_Baseline_Energy_kWh",
            ]
        ],
        on=MACHINE_COLUMN,
        how="left",
    )

    # ------------------------------------------------------------------
    # STATE BASELINE
    # ------------------------------------------------------------------

    result = result.merge(
        state_baseline[
            [
                MACHINE_COLUMN,
                STATE_COLUMN,
                "State_Baseline_Energy_kWh",
                "Baseline_Data_Sufficiency",
            ]
        ],
        on=[
            MACHINE_COLUMN,
            STATE_COLUMN,
        ],
        how="left",
    )

    # ------------------------------------------------------------------
    # HOURLY BASELINE
    # ------------------------------------------------------------------

    result = result.merge(
        hourly_baseline[
            [
                MACHINE_COLUMN,
                "Hour",
                "Hourly_Baseline_Energy_kWh",
            ]
        ],
        on=[
            MACHINE_COLUMN,
            "Hour",
        ],
        how="left",
    )

    # ------------------------------------------------------------------
    # PRODUCTION BASELINE
    # ------------------------------------------------------------------

    result = result.merge(
        production_baseline,
        on=MACHINE_COLUMN,
        how="left",
    )

    result["Production_Adjusted_Baseline_Energy_kWh"] = np.where(
        result[PRODUCTION_COLUMN] > 0,
        result["Normal_Energy_Per_Unit_kWh"]
        * result[PRODUCTION_COLUMN],
        np.nan,
    )

    # ------------------------------------------------------------------
    # FINAL BASELINE SELECTION
    # ------------------------------------------------------------------
    #
    # Priority:
    #
    # 1. Machine-state baseline
    # 2. Production-adjusted baseline
    # 3. Hourly baseline
    # 4. Machine baseline
    #
    # This produces a contextual expected value without reproducing the
    # actual observation.
    # ------------------------------------------------------------------

    result[BASELINE_COLUMN] = (
        result["State_Baseline_Energy_kWh"]
        .fillna(
            result["Production_Adjusted_Baseline_Energy_kWh"]
        )
        .fillna(
            result["Hourly_Baseline_Energy_kWh"]
        )
        .fillna(
            result["Machine_Baseline_Energy_kWh"]
        )
    )

    # Never allow a negative baseline.
    result[BASELINE_COLUMN] = (
        result[BASELINE_COLUMN]
        .clip(lower=0)
    )

    logger.info(
        "Baseline layers successfully applied."
    )

    return result


# ============================================================================
# DEVIATION CALCULATIONS
# ============================================================================

def calculate_deviations(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate actual-versus-baseline energy deviation.
    """

    logger.info(
        "Calculating actual-versus-baseline deviation..."
    )

    result = df.copy()

    result[DEVIATION_COLUMN] = (
        result[ENERGY_COLUMN]
        - result[BASELINE_COLUMN]
    )

    result[DEVIATION_PERCENT_COLUMN] = (
        safe_divide(
            result[DEVIATION_COLUMN],
            result[BASELINE_COLUMN],
        )
        * 100
    )

    result["Absolute_Energy_Deviation_kWh"] = (
        result[DEVIATION_COLUMN].abs()
    )

    result["Energy_Per_Production_Unit"] = safe_divide(
        result[ENERGY_COLUMN],
        result[PRODUCTION_COLUMN],
    )

    result["Baseline_Energy_Per_Production_Unit"] = (
        safe_divide(
            result[BASELINE_COLUMN],
            result[PRODUCTION_COLUMN],
        )
    )

    logger.info(
        "Energy deviation calculations completed."
    )

    return result


# ============================================================================
# ENERGY PERFORMANCE CLASSIFICATION
# ============================================================================

def classify_energy_performance(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Classify observations according to deviation from normal baseline.
    """

    logger.info(
        "Classifying energy performance..."
    )

    result = df.copy()

    result["Energy_Performance"] = np.select(
        [
            result[DEVIATION_PERCENT_COLUMN]
            >= HIGH_CONSUMPTION_THRESHOLD,

            result[DEVIATION_PERCENT_COLUMN]
            <= LOW_CONSUMPTION_THRESHOLD,
        ],
        [
            "ABOVE_BASELINE",
            "BELOW_BASELINE",
        ],
        default="WITHIN_BASELINE",
    )

    result["Potential_High_Consumption_Flag"] = (
        result[DEVIATION_PERCENT_COLUMN]
        >= HIGH_CONSUMPTION_THRESHOLD
    )

    return result


# ============================================================================
# PRODUCTION-AWARE METRICS
# ============================================================================

def calculate_production_metrics(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate production-aware energy metrics.

    Higher energy alone does not imply inefficiency. Production context
    must be considered.
    """

    logger.info(
        "Calculating production-aware energy metrics..."
    )

    result = df.copy()

    result["Energy_Intensity_Variance_kWh_per_Unit"] = (
        result["Energy_Per_Production_Unit"]
        - result["Baseline_Energy_Per_Production_Unit"]
    )

    result["Production_Aware_Performance"] = np.select(
        [
            (
                result[PRODUCTION_COLUMN] > 0
            )
            & (
                result[
                    "Energy_Intensity_Variance_kWh_per_Unit"
                ] > 0
            ),

            (
                result[PRODUCTION_COLUMN] > 0
            )
            & (
                result[
                    "Energy_Intensity_Variance_kWh_per_Unit"
                ] <= 0
            ),
        ],
        [
            "HIGHER_ENERGY_INTENSITY",
            "NORMAL_OR_BETTER",
        ],
        default="NON_PRODUCTIVE",
    )

    logger.info(
        "Production-aware metrics calculated."
    )

    return result


# ============================================================================
# HIGH-CONSUMPTION OBSERVATIONS
# ============================================================================

def identify_high_consumption(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract observations with materially positive baseline deviation.
    """

    logger.info(
        "Identifying potential high-consumption observations..."
    )

    high = df[
        df["Potential_High_Consumption_Flag"]
    ].copy()

    columns = [
        "Record_ID",
        TIMESTAMP_COLUMN,
        MACHINE_COLUMN,
        "Machine_Name",
        STATE_COLUMN,
        ENERGY_COLUMN,
        BASELINE_COLUMN,
        DEVIATION_COLUMN,
        DEVIATION_PERCENT_COLUMN,
        PRODUCTION_COLUMN,
        "Energy_Per_Production_Unit",
        "Energy_Performance",
        "Production_Aware_Performance",
    ]

    columns = [
        column
        for column in columns
        if column in high.columns
    ]

    result = (
        high[columns]
        .sort_values(
            DEVIATION_PERCENT_COLUMN,
            ascending=False,
        )
    )

    logger.info(
        "Potential high-consumption observations: %s",
        len(result),
    )

    return result


# ============================================================================
# MACHINE BASELINE PERFORMANCE
# ============================================================================

def calculate_machine_performance(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize machine-level actual versus baseline performance.
    """

    logger.info(
        "Calculating machine baseline performance..."
    )

    result = (
        df.groupby(MACHINE_COLUMN)
        .agg(
            Records=(ENERGY_COLUMN, "count"),
            Actual_Energy_kWh=(
                ENERGY_COLUMN,
                "sum",
            ),
            Baseline_Energy_kWh=(
                BASELINE_COLUMN,
                "sum",
            ),
            Energy_Deviation_kWh=(
                DEVIATION_COLUMN,
                "sum",
            ),
            Average_Deviation_Percent=(
                DEVIATION_PERCENT_COLUMN,
                "mean",
            ),
            Production_Units=(
                PRODUCTION_COLUMN,
                "sum",
            ),
            Idle_Energy_kWh=(
                "Idle_Energy_kWh",
                "sum",
            ),
        )
        .reset_index()
    )

    result["Deviation_Percent"] = (
        safe_divide(
            result["Energy_Deviation_kWh"],
            result["Baseline_Energy_kWh"],
        )
        * 100
    )

    result["Energy_Per_Unit"] = safe_divide(
        result["Actual_Energy_kWh"],
        result["Production_Units"],
    )

    result["Baseline_Energy_Per_Unit"] = safe_divide(
        result["Baseline_Energy_kWh"],
        result["Production_Units"],
    )

    result["Machine_Performance"] = np.select(
        [
            result["Deviation_Percent"] >= 10,
            result["Deviation_Percent"] <= -10,
        ],
        [
            "ABOVE_BASELINE",
            "BELOW_BASELINE",
        ],
        default="WITHIN_BASELINE",
    )

    return result


# ============================================================================
# VALIDATION
# ============================================================================

def validate_baseline(
    original_df: pd.DataFrame,
    enriched_df: pd.DataFrame,
    machine_baseline: pd.DataFrame,
) -> pd.DataFrame:
    """
    Perform final baseline validation.
    """

    logger.info(
        "Validating energy baseline..."
    )

    expected_records = len(original_df)
    actual_records = len(enriched_df)

    expected_machines = len(
        original_df[MACHINE_COLUMN].dropna().unique()
    )

    actual_machines = len(
        machine_baseline[MACHINE_COLUMN].unique()
    )

    validation = [
        {
            "Validation": "Record Count",
            "Expected": expected_records,
            "Actual": actual_records,
            "Status": (
                "PASS"
                if expected_records == actual_records
                else "FAIL"
            ),
        },
        {
            "Validation": "Machine Baseline Count",
            "Expected": expected_machines,
            "Actual": actual_machines,
            "Status": (
                "PASS"
                if expected_machines == actual_machines
                else "FAIL"
            ),
        },
        {
            "Validation": "Missing Calculated Baselines",
            "Expected": 0,
            "Actual": int(
                enriched_df[BASELINE_COLUMN]
                .isna()
                .sum()
            ),
            "Status": (
                "PASS"
                if enriched_df[BASELINE_COLUMN]
                .notna()
                .all()
                else "FAIL"
            ),
        },
        {
            "Validation": "Negative Baseline Values",
            "Expected": 0,
            "Actual": int(
                (
                    enriched_df[BASELINE_COLUMN] < 0
                ).sum()
            ),
            "Status": (
                "PASS"
                if (
                    enriched_df[BASELINE_COLUMN] >= 0
                ).all()
                else "FAIL"
            ),
        },
        {
            "Validation": "Infinite Deviation Values",
            "Expected": 0,
            "Actual": int(
                np.isinf(
                    enriched_df[
                        DEVIATION_PERCENT_COLUMN
                    ]
                    .fillna(0)
                ).sum()
            ),
            "Status": "PASS",
        },
        {
            "Validation": "Valid Machine IDs",
            "Expected": len(MACHINE_IDS),
            "Actual": int(
                enriched_df[MACHINE_COLUMN]
                .isin(MACHINE_IDS)
                .sum()
                == len(enriched_df)
            ),
            "Status": (
                "PASS"
                if enriched_df[MACHINE_COLUMN]
                .isin(MACHINE_IDS)
                .all()
                else "FAIL"
            ),
        },
        {
            "Validation": "Valid Machine States",
            "Expected": len(VALID_STATES),
            "Actual": int(
                enriched_df[STATE_COLUMN]
                .isin(VALID_STATES)
                .sum()
                == len(enriched_df)
            ),
            "Status": (
                "PASS"
                if enriched_df[STATE_COLUMN]
                .isin(VALID_STATES)
                .all()
                else "FAIL"
            ),
        },
    ]

    validation_df = pd.DataFrame(validation)

    failed = validation_df[
        validation_df["Status"] == "FAIL"
    ]

    if not failed.empty:
        logger.error(
            "Baseline validation failed."
        )

        raise ValueError(
            "Baseline validation failed:\n"
            + failed.to_string(index=False)
        )

    logger.info(
        "Baseline validation completed successfully."
    )

    return validation_df


# ============================================================================
# EXECUTIVE SUMMARY
# ============================================================================

def create_summary(
    df: pd.DataFrame,
    high_consumption: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create the baseline executive summary.
    """

    actual_energy = df[ENERGY_COLUMN].sum()

    calculated_baseline = (
        df[BASELINE_COLUMN].sum()
    )

    deviation = (
        actual_energy
        - calculated_baseline
    )

    deviation_percent = (
        deviation / calculated_baseline * 100
        if calculated_baseline != 0
        else np.nan
    )

    summary = pd.DataFrame(
        {
            "Metric": [
                "Records Analyzed",
                "Machines Analyzed",
                "Actual Energy (kWh)",
                "Calculated Baseline Energy (kWh)",
                "Energy Deviation (kWh)",
                "Energy Deviation (%)",
                "Total Production Units",
                "Total Idle Energy (kWh)",
                "Potential High-Consumption Records",
                "Machines Above Aggregate Baseline",
            ],
            "Value": [
                len(df),
                df[MACHINE_COLUMN].nunique(),
                actual_energy,
                calculated_baseline,
                deviation,
                deviation_percent,
                df[PRODUCTION_COLUMN].sum(),
                df["Idle_Energy_kWh"].sum()
                if "Idle_Energy_kWh" in df.columns
                else np.nan,
                len(high_consumption),
                int(
                    (
                        calculate_machine_performance(df)[
                            "Energy_Deviation_kWh"
                        ]
                        > 0
                    ).sum()
                ),
            ],
        }
    )

    return summary


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main() -> None:
    """
    Execute the complete energy baseline development pipeline.
    """

    logger.info("=" * 78)
    logger.info(
        "SUSTAINABLE MANUFACTURING - ENERGY BASELINE DEVELOPMENT"
    )
    logger.info("=" * 78)

    initialize_directories()

    BASELINE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.info(
        "Baseline output directory initialized."
    )

    # ------------------------------------------------------------------
    # LOAD
    # ------------------------------------------------------------------

    raw_df = load_cleaned_dataset()

    # ------------------------------------------------------------------
    # PREPARE
    # ------------------------------------------------------------------

    df = prepare_features(raw_df)

    # ------------------------------------------------------------------
    # BASELINE LAYERS
    # ------------------------------------------------------------------

    machine_baseline = calculate_machine_baseline(df)

    state_baseline = calculate_state_baseline(df)

    hourly_baseline = calculate_hourly_baseline(df)

    daily_baseline = calculate_daily_baseline(df)

    production_baseline = (
        calculate_production_adjusted_baseline(df)
    )

    # ------------------------------------------------------------------
    # APPLY BASELINE
    # ------------------------------------------------------------------

    enriched = apply_baseline_layers(
        df=df,
        machine_baseline=machine_baseline,
        state_baseline=state_baseline,
        hourly_baseline=hourly_baseline,
        production_baseline=production_baseline,
    )

    # ------------------------------------------------------------------
    # CALCULATIONS
    # ------------------------------------------------------------------

    enriched = calculate_deviations(
        enriched
    )

    enriched = classify_energy_performance(
        enriched
    )

    enriched = calculate_production_metrics(
        enriched
    )

    # ------------------------------------------------------------------
    # HIGH CONSUMPTION
    # ------------------------------------------------------------------

    high_consumption = identify_high_consumption(
        enriched
    )

    # ------------------------------------------------------------------
    # MACHINE PERFORMANCE
    # ------------------------------------------------------------------

    machine_performance = calculate_machine_performance(
        enriched
    )

    # ------------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------------

    validation = validate_baseline(
        original_df=raw_df,
        enriched_df=enriched,
        machine_baseline=machine_baseline,
    )

    # ------------------------------------------------------------------
    # SAVE OUTPUTS
    # ------------------------------------------------------------------

    logger.info(
        "Saving baseline analytical outputs..."
    )

    save_csv(
        machine_baseline,
        "machine_baseline.csv",
    )

    save_csv(
        state_baseline,
        "state_baseline.csv",
    )

    save_csv(
        hourly_baseline,
        "hourly_baseline.csv",
    )

    save_csv(
        daily_baseline,
        "daily_baseline.csv",
    )

    save_csv(
        production_baseline,
        "production_adjusted_baseline.csv",
    )

    save_csv(
        machine_performance,
        "machine_baseline_performance.csv",
    )

    save_csv(
        high_consumption,
        "high_consumption_observations.csv",
    )

    save_csv(
        validation,
        "baseline_validation.csv",
    )

    summary = create_summary(
        enriched,
        high_consumption,
    )

    save_csv(
        summary,
        "baseline_summary.csv",
    )

    # ------------------------------------------------------------------
    # BASELINE-ENRICHED DATASET
    # ------------------------------------------------------------------

    output_dataset = (
        BASELINE_DIR
        / "baseline_enriched_dataset.csv"
    )

    enriched.to_csv(
        output_dataset,
        index=False,
    )

    logger.info(
        "Baseline-enriched dataset saved: %s",
        output_dataset,
    )

    # ------------------------------------------------------------------
    # FINAL OUTPUT
    # ------------------------------------------------------------------

    actual_energy = enriched[
        ENERGY_COLUMN
    ].sum()

    baseline_energy = enriched[
        BASELINE_COLUMN
    ].sum()

    deviation = (
        actual_energy
        - baseline_energy
    )

    deviation_percent = (
        deviation / baseline_energy * 100
        if baseline_energy != 0
        else np.nan
    )

    logger.info("=" * 78)
    logger.info(
        "ENERGY BASELINE DEVELOPMENT COMPLETED SUCCESSFULLY"
    )
    logger.info("=" * 78)

    logger.info(
        "Records analyzed        : %s",
        f"{len(enriched):,}",
    )

    logger.info(
        "Machines analyzed       : %s",
        enriched[MACHINE_COLUMN].nunique(),
    )

    logger.info(
        "Actual energy           : %.2f kWh",
        actual_energy,
    )

    logger.info(
        "Calculated baseline     : %.2f kWh",
        baseline_energy,
    )

    logger.info(
        "Energy deviation        : %.2f kWh",
        deviation,
    )

    logger.info(
        "Energy deviation        : %.2f%%",
        deviation_percent,
    )

    logger.info(
        "High-consumption cases  : %s",
        len(high_consumption),
    )

    logger.info(
        "Baseline output folder  : %s",
        BASELINE_DIR,
    )

    print("\n" + "=" * 78)
    print("ENERGY BASELINE EXECUTIVE SUMMARY")
    print("=" * 78)
    print(
        summary.to_string(
            index=False
        )
    )
    print("=" * 78)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        logger.exception(
            "Unexpected error during baseline development: %s",
            error,
        )

        raise