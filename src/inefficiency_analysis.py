"""
Inefficiency Analysis
Sustainable Manufacturing:
Energy Consumption & Carbon Footprint Optimization Analytics

Purpose:
    Identify potential energy inefficiencies using:
        - Energy baseline deviation
        - Idle-energy consumption
        - Energy intensity
        - Schedule deviations
        - Machine-level performance
        - Production-aware analysis

Input:
    outputs/baseline/baseline_enriched_dataset.csv

Outputs:
    outputs/inefficiency/
        machine_inefficiency_summary.csv
        idle_energy_analysis.csv
        idle_energy_opportunities.csv
        high_consumption_analysis.csv
        energy_intensity_analysis.csv
        schedule_deviation_analysis.csv
        inefficiency_candidates.csv
        inefficiency_validation.csv
        inefficiency_summary.csv
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from library.config import (
    BASELINE_DIR,
    INEFFICIENCY_DIR,
    initialize_directories,
)

from library.constants import (
    MACHINE_IDS,
    VALID_STATES,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_FILE = (
    BASELINE_DIR
    / "baseline_enriched_dataset.csv"
)

ENERGY_COLUMN = "Energy_Consumption_kWh"
BASELINE_COLUMN = "Calculated_Baseline_Energy_kWh"
DEVIATION_COLUMN = "Baseline_Deviation_kWh"
DEVIATION_PERCENT_COLUMN = "Baseline_Deviation_Percent"

PRODUCTION_COLUMN = "Actual_Production_Units"
STATE_COLUMN = "Actual_State"
PLANNED_STATE_COLUMN = "Planned_State"
MACHINE_COLUMN = "Machine_ID"
TIMESTAMP_COLUMN = "Timestamp"

IDLE_ENERGY_COLUMN = "Idle_Energy_kWh"

# Analytical thresholds.
HIGH_CONSUMPTION_THRESHOLD = 10.0
IDLE_ENERGY_PERCENTILE = 75
ENERGY_INTENSITY_PERCENTILE = 75

# Minimum production required for meaningful energy-per-unit analysis.
MIN_PRODUCTION_FOR_INTENSITY = 0


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
# HELPERS
# ============================================================================

def save_csv(
    dataframe: pd.DataFrame,
    filename: str,
) -> Path:
    """
    Save an analytical DataFrame to the centralized output directory.
    """

    output_path = INEFFICIENCY_DIR / filename

    dataframe.to_csv(
        output_path,
        index=False,
    )

    logger.info(
        "CSV saved: %s",
        output_path,
    )

    return output_path


def safe_divide(
    numerator,
    denominator,
):
    """
    Perform safe element-wise division for both Pandas Series
    and scalar numeric denominators.

    The function prevents division-by-zero and infinite values
    while preserving compatibility with Pandas analytical workflows.
    """

    if np.isscalar(denominator):

        denominator_value = float(denominator)

        if (
            denominator_value == 0
            or not np.isfinite(denominator_value)
        ):
            return pd.Series(
                np.nan,
                index=numerator.index,
            )

        result = numerator / denominator_value

    else:

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

def load_baseline_dataset() -> pd.DataFrame:
    """
    Load the baseline-enriched analytical dataset.
    """

    logger.info(
        "Loading baseline-enriched dataset..."
    )

    logger.info(
        "Source: %s",
        INPUT_FILE,
    )

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Baseline dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    logger.info(
        "Dataset loaded: %s rows × %s columns",
        f"{len(df):,}",
        len(df.columns),
    )

    return df


# ============================================================================
# DATA PREPARATION
# ============================================================================

def prepare_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare analytical variables required for inefficiency detection.
    """

    logger.info(
        "Preparing inefficiency-analysis features..."
    )

    required_columns = [
        ENERGY_COLUMN,
        BASELINE_COLUMN,
        PRODUCTION_COLUMN,
        STATE_COLUMN,
        MACHINE_COLUMN,
        TIMESTAMP_COLUMN,
        IDLE_ENERGY_COLUMN,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Required columns missing from baseline dataset: "
            + ", ".join(missing_columns)
        )

    result = df.copy()

    result[TIMESTAMP_COLUMN] = pd.to_datetime(
        result[TIMESTAMP_COLUMN],
        errors="coerce",
    )

    numeric_columns = [
        ENERGY_COLUMN,
        BASELINE_COLUMN,
        PRODUCTION_COLUMN,
        IDLE_ENERGY_COLUMN,
    ]

    for column in numeric_columns:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

    # ------------------------------------------------------------------
    # Compatibility alias
    # ------------------------------------------------------------------
    #
    # This explicitly provides the name expected by downstream
    # analytical modules.
    # ------------------------------------------------------------------

    if "Energy_Per_Production_Unit" not in result.columns:

        result["Energy_Per_Production_Unit"] = safe_divide(
            result[ENERGY_COLUMN],
            result[PRODUCTION_COLUMN],
        )

    # Baseline energy intensity.
    if "Baseline_Energy_Per_Production_Unit" not in result.columns:

        result["Baseline_Energy_Per_Production_Unit"] = safe_divide(
            result[BASELINE_COLUMN],
            result[PRODUCTION_COLUMN],
        )

    # Recalculate deviation if necessary.
    if DEVIATION_COLUMN not in result.columns:

        result[DEVIATION_COLUMN] = (
            result[ENERGY_COLUMN]
            - result[BASELINE_COLUMN]
        )

    if DEVIATION_PERCENT_COLUMN not in result.columns:

        result[DEVIATION_PERCENT_COLUMN] = (
            safe_divide(
                result[DEVIATION_COLUMN],
                result[BASELINE_COLUMN],
            )
            * 100
        )

    # State deviation.
    result["State_Deviation_Flag"] = (
        result[PLANNED_STATE_COLUMN]
        != result[STATE_COLUMN]
    )

    # Schedule deviation type.
    result["Schedule_Deviation_Type"] = (
        result.apply(
            classify_schedule_deviation,
            axis=1,
        )
    )

    logger.info(
        "Analytical features prepared."
    )

    return result


# ============================================================================
# SCHEDULE DEVIATION CLASSIFICATION
# ============================================================================

def classify_schedule_deviation(
    row: pd.Series,
) -> str:
    """
    Classify planned-versus-actual operating-state differences.

    This classification does not automatically label every deviation
    as inefficient.
    """

    planned = row.get(
        PLANNED_STATE_COLUMN
    )

    actual = row.get(
        STATE_COLUMN
    )

    if planned == actual:
        return "NO_DEVIATION"

    if (
        planned == "RUNNING"
        and actual == "IDLE"
    ):
        return "PLANNED_RUNNING_ACTUAL_IDLE"

    if (
        planned == "RUNNING"
        and actual == "OFF"
    ):
        return "UNPLANNED_SHUTDOWN"

    if (
        planned == "RUNNING"
        and actual == "MAINTENANCE"
    ):
        return "UNPLANNED_MAINTENANCE"

    if (
        planned == "IDLE"
        and actual == "RUNNING"
    ):
        return "UNPLANNED_OPERATION"

    if (
        planned == "IDLE"
        and actual == "OFF"
    ):
        return "EARLY_SHUTDOWN"

    if (
        planned == "IDLE"
        and actual == "MAINTENANCE"
    ):
        return "MAINTENANCE_DURING_IDLE"

    return "OTHER_STATE_DEVIATION"


# ============================================================================
# MACHINE-LEVEL ANALYSIS
# ============================================================================

def analyze_machine_inefficiency(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate machine-level energy inefficiency indicators.
    """

    logger.info(
        "Calculating machine-level inefficiency summary..."
    )

    grouped = (
        df.groupby(MACHINE_COLUMN)
        .agg(
            Records=(
                ENERGY_COLUMN,
                "count",
            ),
            Total_Energy_kWh=(
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
            Total_Production_Units=(
                PRODUCTION_COLUMN,
                "sum",
            ),
            Total_Idle_Energy_kWh=(
                IDLE_ENERGY_COLUMN,
                "sum",
            ),
        )
        .reset_index()
    )

    grouped["Energy_Deviation_Percent"] = (
        safe_divide(
            grouped["Energy_Deviation_kWh"],
            grouped["Baseline_Energy_kWh"],
        )
        * 100
    )

    grouped["Energy_Per_Production_Unit"] = safe_divide(
        grouped["Total_Energy_kWh"],
        grouped["Total_Production_Units"],
    )

    grouped["Baseline_Energy_Per_Production_Unit"] = (
        safe_divide(
            grouped["Baseline_Energy_kWh"],
            grouped["Total_Production_Units"],
        )
    )

    grouped["Idle_Energy_Percent"] = (
        safe_divide(
            grouped["Total_Idle_Energy_kWh"],
            grouped["Total_Energy_kWh"],
        )
        * 100
    )

    grouped["Potential_Inefficiency_Flag"] = (
        (
            grouped["Energy_Deviation_Percent"]
            >= HIGH_CONSUMPTION_THRESHOLD
        )
        | (
            grouped["Idle_Energy_Percent"]
            >= grouped["Idle_Energy_Percent"].median()
        )
    )

    logger.info(
        "Machine-level summary generated for %s machines.",
        len(grouped),
    )

    return grouped


# ============================================================================
# IDLE ENERGY ANALYSIS
# ============================================================================

def analyze_idle_energy(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Analyze energy consumed while machines are idle.
    """

    logger.info(
        "Analyzing idle-energy consumption..."
    )

    idle = df[
        df[STATE_COLUMN] == "IDLE"
    ].copy()

    if idle.empty:
        return pd.DataFrame(
            columns=[
                MACHINE_COLUMN,
                "Idle_Observations",
                "Idle_Energy_kWh",
                "Average_Idle_Energy_kWh",
                "Maximum_Idle_Energy_kWh",
                "Idle_Energy_Percent",
            ]
        )

    result = (
        idle.groupby(MACHINE_COLUMN)
        .agg(
            Idle_Observations=(
                ENERGY_COLUMN,
                "count",
            ),
            Idle_Energy_kWh=(
                IDLE_ENERGY_COLUMN,
                "sum",
            ),
            Average_Idle_Energy_kWh=(
                IDLE_ENERGY_COLUMN,
                "mean",
            ),
            Maximum_Idle_Energy_kWh=(
                IDLE_ENERGY_COLUMN,
                "max",
            ),
            Total_Machine_Energy_kWh=(
                ENERGY_COLUMN,
                "sum",
            ),
        )
        .reset_index()
    )

    result["Idle_Energy_Percent"] = (
        safe_divide(
            result["Idle_Energy_kWh"],
            result["Total_Machine_Energy_kWh"],
        )
        * 100
    )

    result["Idle_Energy_Rank"] = (
        result["Idle_Energy_kWh"]
        .rank(
            ascending=False,
            method="dense",
        )
        .astype(int)
    )

    logger.info(
        "Idle-energy analysis generated."
    )

    return result


# ============================================================================
# IDLE OPPORTUNITIES
# ============================================================================

def identify_idle_opportunities(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Identify idle observations that may represent potential energy-saving
    opportunities.

    The analysis does not claim that all idle energy is waste.
    """

    logger.info(
        "Identifying potential idle-energy opportunities..."
    )

    idle = df[
        df[STATE_COLUMN] == "IDLE"
    ].copy()

    if idle.empty:
        return idle

    threshold = idle[
        IDLE_ENERGY_COLUMN
    ].quantile(
        IDLE_ENERGY_PERCENTILE / 100
    )

    idle["Idle_Energy_Threshold_kWh"] = threshold

    idle["Idle_Energy_Opportunity_Flag"] = (
        idle[IDLE_ENERGY_COLUMN]
        >= threshold
    )

    idle["Idle_Opportunity_Score"] = (
        safe_divide(
            idle[IDLE_ENERGY_COLUMN],
            threshold,
        )
        * 100
    )

    idle["Opportunity_Type"] = (
        "IDLE_ENERGY_REVIEW"
    )

    columns = [
        "Record_ID",
        TIMESTAMP_COLUMN,
        MACHINE_COLUMN,
        "Machine_Name",
        PLANNED_STATE_COLUMN,
        STATE_COLUMN,
        ENERGY_COLUMN,
        IDLE_ENERGY_COLUMN,
        BASELINE_COLUMN,
        DEVIATION_COLUMN,
        DEVIATION_PERCENT_COLUMN,
        PRODUCTION_COLUMN,
        "Idle_Energy_Opportunity_Flag",
        "Idle_Opportunity_Score",
        "Opportunity_Type",
    ]

    columns = [
        column
        for column in columns
        if column in idle.columns
    ]

    result = (
        idle[
            idle["Idle_Energy_Opportunity_Flag"]
        ][columns]
        .sort_values(
            "Idle_Opportunity_Score",
            ascending=False,
        )
    )

    logger.info(
        "Potential idle-energy opportunities: %s",
        len(result),
    )

    return result


# ============================================================================
# HIGH-CONSUMPTION ANALYSIS
# ============================================================================

def analyze_high_consumption(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Analyze observations where actual energy materially exceeds baseline.
    """

    logger.info(
        "Analyzing high-consumption observations..."
    )

    high = df[
        df[DEVIATION_PERCENT_COLUMN]
        >= HIGH_CONSUMPTION_THRESHOLD
    ].copy()

    columns = [
        "Record_ID",
        TIMESTAMP_COLUMN,
        MACHINE_COLUMN,
        "Machine_Name",
        STATE_COLUMN,
        PLANNED_STATE_COLUMN,
        ENERGY_COLUMN,
        BASELINE_COLUMN,
        DEVIATION_COLUMN,
        DEVIATION_PERCENT_COLUMN,
        PRODUCTION_COLUMN,
        "Energy_Per_Production_Unit",
        "Baseline_Energy_Per_Production_Unit",
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
        "High-consumption analysis generated: %s records.",
        len(result),
    )

    return result


# ============================================================================
# ENERGY INTENSITY ANALYSIS
# ============================================================================

def analyze_energy_intensity(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Identify machines and observations with relatively high energy
    consumption per production unit.
    """

    logger.info(
        "Analyzing energy intensity..."
    )

    productive = df[
        (
            df[PRODUCTION_COLUMN]
            > MIN_PRODUCTION_FOR_INTENSITY
        )
        & (
            df[STATE_COLUMN]
            == "RUNNING"
        )
    ].copy()

    if productive.empty:
        return pd.DataFrame()

    productive["Energy_Per_Production_Unit"] = safe_divide(
        productive[ENERGY_COLUMN],
        productive[PRODUCTION_COLUMN],
    )

    productive["Baseline_Energy_Per_Production_Unit"] = (
        safe_divide(
            productive[BASELINE_COLUMN],
            productive[PRODUCTION_COLUMN],
        )
    )

    threshold = productive[
        "Energy_Per_Production_Unit"
    ].quantile(
        ENERGY_INTENSITY_PERCENTILE / 100
    )

    productive["Energy_Intensity_Threshold"] = (
        threshold
    )

    productive["High_Energy_Intensity_Flag"] = (
        productive["Energy_Per_Production_Unit"]
        >= threshold
    )

    productive["Energy_Intensity_Variance"] = (
        productive["Energy_Per_Production_Unit"]
        - productive[
            "Baseline_Energy_Per_Production_Unit"
        ]
    )

    columns = [
        "Record_ID",
        TIMESTAMP_COLUMN,
        MACHINE_COLUMN,
        "Machine_Name",
        ENERGY_COLUMN,
        PRODUCTION_COLUMN,
        "Energy_Per_Production_Unit",
        "Baseline_Energy_Per_Production_Unit",
        "Energy_Intensity_Variance",
        "High_Energy_Intensity_Flag",
    ]

    columns = [
        column
        for column in columns
        if column in productive.columns
    ]

    result = (
        productive[columns]
        .sort_values(
            "Energy_Per_Production_Unit",
            ascending=False,
        )
    )

    logger.info(
        "Energy-intensity analysis generated."
    )

    return result


# ============================================================================
# SCHEDULE ANALYSIS
# ============================================================================

def analyze_schedule_deviations(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare planned and actual operating states.
    """

    logger.info(
        "Analyzing schedule deviations..."
    )

    deviations = df[
        df["State_Deviation_Flag"]
    ].copy()

    if deviations.empty:
        return pd.DataFrame(
            columns=[
                MACHINE_COLUMN,
                "Deviation_Type",
                "Records",
                "Energy_kWh",
                "Production_Units",
                "Idle_Energy_kWh",
            ]
        )

    result = (
        deviations.groupby(
            [
                MACHINE_COLUMN,
                "Schedule_Deviation_Type",
            ]
        )
        .agg(
            Records=(
                "Record_ID",
                "count",
            ),
            Energy_kWh=(
                ENERGY_COLUMN,
                "sum",
            ),
            Production_Units=(
                PRODUCTION_COLUMN,
                "sum",
            ),
            Idle_Energy_kWh=(
                IDLE_ENERGY_COLUMN,
                "sum",
            ),
        )
        .reset_index()
        .rename(
            columns={
                "Schedule_Deviation_Type":
                    "Deviation_Type"
            }
        )
    )

    result["Potential_Energy_Impact_kWh"] = (
        result["Energy_kWh"]
    )

    logger.info(
        "Schedule deviation analysis generated."
    )

    return result


# ============================================================================
# CANDIDATE GENERATION
# ============================================================================

def generate_inefficiency_candidates(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine multiple analytical signals into a candidate list.

    A candidate is not automatically a confirmed inefficiency. It represents
    an observation requiring operational investigation.
    """

    logger.info(
        "Generating potential inefficiency candidates..."
    )

    result = df.copy()

    result["Baseline_Signal"] = (
        result[DEVIATION_PERCENT_COLUMN]
        >= HIGH_CONSUMPTION_THRESHOLD
    )

    idle_threshold = np.nan

    idle_rows = result[
        result[STATE_COLUMN] == "IDLE"
    ]

    if not idle_rows.empty:
        idle_threshold = idle_rows[
            IDLE_ENERGY_COLUMN
        ].quantile(
            IDLE_ENERGY_PERCENTILE / 100
        )

    result["Idle_Signal"] = False

    if not pd.isna(idle_threshold):
        result.loc[
            result[STATE_COLUMN] == "IDLE",
            "Idle_Signal",
        ] = (
            result.loc[
                result[STATE_COLUMN] == "IDLE",
                IDLE_ENERGY_COLUMN,
            ]
            >= idle_threshold
        )

    result["Schedule_Signal"] = (
        result["State_Deviation_Flag"]
    )

    result["Intensity_Signal"] = (
        (
            result[PRODUCTION_COLUMN] > 0
        )
        & (
            result[
                "Energy_Per_Production_Unit"
            ]
            >
            result[
                "Baseline_Energy_Per_Production_Unit"
            ]
        )
    )

    result["Inefficiency_Signal_Count"] = (
        result["Baseline_Signal"].astype(int)
        + result["Idle_Signal"].astype(int)
        + result["Schedule_Signal"].astype(int)
        + result["Intensity_Signal"].astype(int)
    )

    # ------------------------------------------------------------------
    # Priority
    # ------------------------------------------------------------------

    result["Priority"] = np.select(
        [
            result[
                "Inefficiency_Signal_Count"
            ] >= 3,

            result[
                "Inefficiency_Signal_Count"
            ] == 2,

            result[
                "Inefficiency_Signal_Count"
            ] == 1,
        ],
        [
            "HIGH",
            "MEDIUM",
            "LOW",
        ],
        default="NONE",
    )

    candidates = result[
        result[
            "Inefficiency_Signal_Count"
        ] > 0
    ].copy()

    # ------------------------------------------------------------------
    # Opportunity score
    # ------------------------------------------------------------------

    baseline_score = (
        result[
            DEVIATION_PERCENT_COLUMN
        ]
        .clip(lower=0)
        .fillna(0)
    )

    intensity_score = (
        result[
            "Energy_Per_Production_Unit"
        ]
        /
        result[
            "Baseline_Energy_Per_Production_Unit"
        ]
    )

    intensity_score = (
        intensity_score
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(1)
    )

    intensity_score = (
        (intensity_score - 1)
        .clip(lower=0)
        * 100
    )

    idle_score = (
        result[IDLE_ENERGY_COLUMN]
        .fillna(0)
    )

    # Normalize idle energy against maximum observed idle energy.
    max_idle = idle_score.max()

    if max_idle > 0:
        idle_score = (
            idle_score / max_idle * 100
        )
    else:
        idle_score = 0

    candidates["Opportunity_Score"] = (
        baseline_score.loc[
            candidates.index
        ] * 0.40
        +
        intensity_score.loc[
            candidates.index
        ] * 0.25
        +
        idle_score.loc[
            candidates.index
        ] * 0.20
        +
        candidates[
            "Inefficiency_Signal_Count"
        ] * 5
    )

    candidates["Investigation_Required"] = (
        True
    )

    columns = [
        "Record_ID",
        TIMESTAMP_COLUMN,
        MACHINE_COLUMN,
        "Machine_Name",
        PLANNED_STATE_COLUMN,
        STATE_COLUMN,
        "Schedule_Deviation_Type",
        ENERGY_COLUMN,
        BASELINE_COLUMN,
        DEVIATION_COLUMN,
        DEVIATION_PERCENT_COLUMN,
        PRODUCTION_COLUMN,
        "Energy_Per_Production_Unit",
        "Baseline_Energy_Per_Production_Unit",
        IDLE_ENERGY_COLUMN,
        "Baseline_Signal",
        "Idle_Signal",
        "Schedule_Signal",
        "Intensity_Signal",
        "Inefficiency_Signal_Count",
        "Opportunity_Score",
        "Priority",
        "Investigation_Required",
    ]

    columns = [
        column
        for column in columns
        if column in candidates.columns
    ]

    candidates = (
        candidates[columns]
        .sort_values(
            "Opportunity_Score",
            ascending=False,
        )
    )

    logger.info(
        "Potential inefficiency candidates: %s",
        len(candidates),
    )

    return candidates


# ============================================================================
# VALIDATION
# ============================================================================

def validate_outputs(
    df: pd.DataFrame,
    machine_summary: pd.DataFrame,
    idle_analysis: pd.DataFrame,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate the core inefficiency-analysis outputs.
    """

    logger.info(
        "Validating inefficiency-analysis outputs..."
    )

    validations = []

    def add_validation(
        name: str,
        expected,
        actual,
        status: bool,
    ) -> None:

        validations.append(
            {
                "Validation": name,
                "Expected": expected,
                "Actual": actual,
                "Status": (
                    "PASS"
                    if status
                    else "FAIL"
                ),
            }
        )

    add_validation(
        "Record Count",
        10000,
        len(df),
        len(df) == 10000,
    )

    add_validation(
        "Machine Count",
        len(MACHINE_IDS),
        df[MACHINE_COLUMN].nunique(),
        df[MACHINE_COLUMN].nunique()
        == len(MACHINE_IDS),
    )

    add_validation(
        "Negative Energy",
        0,
        int(
            (
                df[ENERGY_COLUMN] < 0
            ).sum()
        ),
        (
            df[ENERGY_COLUMN] >= 0
        ).all(),
    )

    add_validation(
        "Negative Idle Energy",
        0,
        int(
            (
                df[IDLE_ENERGY_COLUMN] < 0
            ).sum()
        ),
        (
            df[IDLE_ENERGY_COLUMN] >= 0
        ).all(),
    )

    add_validation(
        "Negative Opportunity Scores",
        0,
        int(
            (
                candidates[
                    "Opportunity_Score"
                ] < 0
            ).sum()
        ),
        (
            candidates[
                "Opportunity_Score"
            ] >= 0
        ).all(),
    )

    add_validation(
        "Machine Summary",
        len(MACHINE_IDS),
        len(machine_summary),
        len(machine_summary)
        == len(MACHINE_IDS),
    )

    add_validation(
        "Idle Analysis Exists",
        True,
        not idle_analysis.empty,
        True,
    )

    add_validation(
        "Valid Machine IDs",
        True,
        df[MACHINE_COLUMN]
        .isin(MACHINE_IDS)
        .all(),
        df[MACHINE_COLUMN]
        .isin(MACHINE_IDS)
        .all(),
    )

    add_validation(
        "Valid Machine States",
        True,
        df[STATE_COLUMN]
        .isin(VALID_STATES)
        .all(),
        df[STATE_COLUMN]
        .isin(VALID_STATES)
        .all(),
    )

    validation = pd.DataFrame(
        validations
    )

    failed = validation[
        validation["Status"] == "FAIL"
    ]

    if not failed.empty:
        raise ValueError(
            "Inefficiency validation failed:\n"
            + failed.to_string(
                index=False
            )
        )

    logger.info(
        "Inefficiency-analysis validation completed."
    )

    return validation


# ============================================================================
# EXECUTIVE SUMMARY
# ============================================================================

def create_summary(
    df: pd.DataFrame,
    candidates: pd.DataFrame,
    machine_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create an executive summary of inefficiency analysis.
    """

    total_energy = df[
        ENERGY_COLUMN
    ].sum()

    total_idle_energy = df[
        IDLE_ENERGY_COLUMN
    ].sum()

    idle_percentage = (
        total_idle_energy
        / total_energy
        * 100
        if total_energy > 0
        else 0
    )

    positive_deviation = df.loc[
        df[DEVIATION_COLUMN] > 0,
        DEVIATION_COLUMN,
    ].sum()

    high_priority = (
        candidates["Priority"]
        .eq("HIGH")
        .sum()
        if not candidates.empty
        else 0
    )

    medium_priority = (
        candidates["Priority"]
        .eq("MEDIUM")
        .sum()
        if not candidates.empty
        else 0
    )

    summary = pd.DataFrame(
        {
            "Metric": [
                "Records Analyzed",
                "Machines Analyzed",
                "Total Energy (kWh)",
                "Total Idle Energy (kWh)",
                "Idle Energy Percentage",
                "Positive Energy Above Baseline (kWh)",
                "High-Consumption Records",
                "Idle-Energy Opportunity Records",
                "Potential Inefficiency Candidates",
                "High-Priority Candidates",
                "Medium-Priority Candidates",
                "Machines Analyzed for Opportunities",
            ],
            "Value": [
                len(df),
                df[MACHINE_COLUMN].nunique(),
                total_energy,
                total_idle_energy,
                idle_percentage,
                positive_deviation,
                int(
                    (
                        df[
                            DEVIATION_PERCENT_COLUMN
                        ]
                        >= HIGH_CONSUMPTION_THRESHOLD
                    ).sum()
                ),
                int(
                    (
                        (
                            df[STATE_COLUMN]
                            == "IDLE"
                        )
                        &
                        (
                            df[IDLE_ENERGY_COLUMN]
                            > 0
                        )
                    ).sum()
                ),
                len(candidates),
                high_priority,
                medium_priority,
                len(machine_summary),
            ],
        }
    )

    return summary


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main() -> None:
    """
    Execute the complete inefficiency-analysis pipeline.
    """

    logger.info("=" * 78)
    logger.info(
        "SUSTAINABLE MANUFACTURING - INEFFICIENCY ANALYSIS"
    )
    logger.info("=" * 78)

    initialize_directories()

    INEFFICIENCY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.info(
        "Inefficiency-analysis directory initialized."
    )

    # ------------------------------------------------------------------
    # LOAD
    # ------------------------------------------------------------------

    raw_df = load_baseline_dataset()

    # ------------------------------------------------------------------
    # PREPARE
    # ------------------------------------------------------------------

    df = prepare_features(
        raw_df
    )

    # ------------------------------------------------------------------
    # MACHINE ANALYSIS
    # ------------------------------------------------------------------

    machine_summary = (
        analyze_machine_inefficiency(
            df
        )
    )

    # ------------------------------------------------------------------
    # IDLE ANALYSIS
    # ------------------------------------------------------------------

    idle_analysis = (
        analyze_idle_energy(
            df
        )
    )

    idle_opportunities = (
        identify_idle_opportunities(
            df
        )
    )

    # ------------------------------------------------------------------
    # HIGH CONSUMPTION
    # ------------------------------------------------------------------

    high_consumption = (
        analyze_high_consumption(
            df
        )
    )

    # ------------------------------------------------------------------
    # ENERGY INTENSITY
    # ------------------------------------------------------------------

    energy_intensity = (
        analyze_energy_intensity(
            df
        )
    )

    # ------------------------------------------------------------------
    # SCHEDULE ANALYSIS
    # ------------------------------------------------------------------

    schedule_analysis = (
        analyze_schedule_deviations(
            df
        )
    )

    # ------------------------------------------------------------------
    # CANDIDATES
    # ------------------------------------------------------------------

    candidates = (
        generate_inefficiency_candidates(
            df
        )
    )

    # ------------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------------

    validation = validate_outputs(
        df=df,
        machine_summary=machine_summary,
        idle_analysis=idle_analysis,
        candidates=candidates,
    )

    # ------------------------------------------------------------------
    # SAVE OUTPUTS
    # ------------------------------------------------------------------

    logger.info(
        "Saving inefficiency-analysis outputs..."
    )

    save_csv(
        machine_summary,
        "machine_inefficiency_summary.csv",
    )

    save_csv(
        idle_analysis,
        "idle_energy_analysis.csv",
    )

    save_csv(
        idle_opportunities,
        "idle_energy_opportunities.csv",
    )

    save_csv(
        high_consumption,
        "high_consumption_analysis.csv",
    )

    save_csv(
        energy_intensity,
        "energy_intensity_analysis.csv",
    )

    save_csv(
        schedule_analysis,
        "schedule_deviation_analysis.csv",
    )

    save_csv(
        candidates,
        "inefficiency_candidates.csv",
    )

    save_csv(
        validation,
        "inefficiency_validation.csv",
    )

    # Save enriched dataset for downstream modules.
    enriched_path = (
        INEFFICIENCY_DIR
        / "inefficiency_enriched_dataset.csv"
    )

    df.to_csv(
        enriched_path,
        index=False,
    )

    logger.info(
        "Enriched analytical dataset saved: %s",
        enriched_path,
    )

    # ------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------

    summary = create_summary(
        df=df,
        candidates=candidates,
        machine_summary=machine_summary,
    )

    save_csv(
        summary,
        "inefficiency_summary.csv",
    )

    # ------------------------------------------------------------------
    # FINAL LOGGING
    # ------------------------------------------------------------------

    total_energy = df[
        ENERGY_COLUMN
    ].sum()

    total_idle_energy = df[
        IDLE_ENERGY_COLUMN
    ].sum()

    idle_percentage = (
        total_idle_energy
        / total_energy
        * 100
    )

    high_consumption_count = (
        df[
            DEVIATION_PERCENT_COLUMN
        ]
        >= HIGH_CONSUMPTION_THRESHOLD
    ).sum()

    high_priority = (
        candidates["Priority"]
        .eq("HIGH")
        .sum()
        if not candidates.empty
        else 0
    )

    medium_priority = (
        candidates["Priority"]
        .eq("MEDIUM")
        .sum()
        if not candidates.empty
        else 0
    )

    logger.info("=" * 78)
    logger.info(
        "INEFFICIENCY ANALYSIS COMPLETED SUCCESSFULLY"
    )
    logger.info("=" * 78)

    logger.info(
        "Records analyzed        : %s",
        f"{len(df):,}",
    )

    logger.info(
        "Machines analyzed       : %s",
        df[MACHINE_COLUMN].nunique(),
    )

    logger.info(
        "Total energy            : %.2f kWh",
        total_energy,
    )

    logger.info(
        "Total idle energy       : %.2f kWh",
        total_idle_energy,
    )

    logger.info(
        "Idle energy percentage  : %.2f%%",
        idle_percentage,
    )

    logger.info(
        "High-consumption cases  : %s",
        high_consumption_count,
    )

    logger.info(
        "Inefficiency candidates : %s",
        len(candidates),
    )

    logger.info(
        "High-priority candidates: %s",
        high_priority,
    )

    logger.info(
        "Medium-priority candidates: %s",
        medium_priority,
    )

    logger.info(
        "Output directory        : %s",
        INEFFICIENCY_DIR,
    )

    print("\n" + "=" * 78)
    print(
        "INEFFICIENCY ANALYSIS EXECUTIVE SUMMARY"
    )
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
            "Unexpected error during inefficiency analysis: %s",
            error,
        )

        raise