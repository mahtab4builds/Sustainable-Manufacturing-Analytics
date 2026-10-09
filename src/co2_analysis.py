"""
Sustainable Manufacturing:
Energy Consumption & Carbon Footprint Optimization Analytics

Module:
    CO2 & Sustainability Analysis

Purpose:
    Analyze energy-related CO2e emissions using the project's documented
    synthetic emission factor and generate sustainability analytics
    for downstream recommendation and dashboard development.

Important:
    The emission factor used in this project is SYNTHETIC and intended
    strictly for educational, analytical, and demonstration purposes.

    CO2 reduction values derived from What-If scenarios represent
    MODELLED POTENTIAL impacts, not actual achieved savings.
"""

# ============================================================================
# IMPORTS
# ============================================================================

from pathlib import Path
import logging
import sys

import numpy as np
import pandas as pd


# ============================================================================
# PROJECT CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_ROOT / "outputs"

BASELINE_DATA_FILE = (
    OUTPUT_DIR
    / "baseline"
    / "baseline_enriched_dataset.csv"
)

WHAT_IF_SUMMARY_FILE = (
    OUTPUT_DIR
    / "what_if"
    / "scenario_summary.csv"
)

WHAT_IF_CO2_FILE = (
    OUTPUT_DIR
    / "what_if"
    / "scenario_co2_impact.csv"
)

RAW_DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "Sustainable_Manufacturing_Energy_Analytics_10000.xlsx"
)

CO2_DIR = (
    OUTPUT_DIR
    / "co2_analysis"
)

MACHINE_CO2_FILE = (
    CO2_DIR
    / "machine_co2_analysis.csv"
)

DAILY_CO2_FILE = (
    CO2_DIR
    / "daily_co2_analysis.csv"
)

MONTHLY_CO2_FILE = (
    CO2_DIR
    / "monthly_co2_analysis.csv"
)

STATE_CO2_FILE = (
    CO2_DIR
    / "state_co2_analysis.csv"
)

CO2_INTENSITY_FILE = (
    CO2_DIR
    / "co2_intensity_analysis.csv"
)

SUSTAINABILITY_KPI_FILE = (
    CO2_DIR
    / "sustainability_kpis.csv"
)

SCENARIO_CO2_FILE = (
    CO2_DIR
    / "scenario_co2_analysis.csv"
)

MACHINE_SUSTAINABILITY_FILE = (
    CO2_DIR
    / "machine_sustainability_ranking.csv"
)

VALIDATION_FILE = (
    CO2_DIR
    / "co2_validation.csv"
)

EXPECTED_RECORDS = 10_000

DEFAULT_SYNTHETIC_EMISSION_FACTOR = 0.70


# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)

logger = logging.getLogger(__name__)


# ============================================================================
# DIRECTORY MANAGEMENT
# ============================================================================

def initialize_directories():
    """Create the CO2 analysis output directory."""

    CO2_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    logger.info(
        "CO2 and sustainability-analysis directory initialized."
    )


# ============================================================================
# DATA LOADING
# ============================================================================

def load_baseline_dataset():
    """Load the validated baseline-enriched dataset."""

    if not BASELINE_DATA_FILE.exists():

        raise FileNotFoundError(
            "Baseline-enriched dataset was not found.\n"
            f"Expected: {BASELINE_DATA_FILE}\n"
            "Run baseline.py before executing co2_analysis.py."
        )

    logger.info(
        "Loading baseline-enriched dataset..."
    )

    logger.info(
        "Source: %s",
        BASELINE_DATA_FILE
    )

    df = pd.read_csv(
        BASELINE_DATA_FILE
    )

    logger.info(
        "Dataset loaded: %s rows × %s columns",
        df.shape[0],
        df.shape[1]
    )

    return df


def load_what_if_summary():
    """Load validated What-If scenario summary."""

    if not WHAT_IF_SUMMARY_FILE.exists():

        raise FileNotFoundError(
            "What-If scenario summary was not found.\n"
            f"Expected: {WHAT_IF_SUMMARY_FILE}\n"
            "Run what_if_analysis.py before executing co2_analysis.py."
        )

    logger.info(
        "Loading What-If scenario summary..."
    )

    summary = pd.read_csv(
        WHAT_IF_SUMMARY_FILE
    )

    logger.info(
        "What-If scenarios loaded: %s",
        len(summary)
    )

    return summary


# ============================================================================
# EMISSION FACTOR
# ============================================================================

def load_emission_factor():
    """
    Load the project's synthetic emission factor.

    The value must come from Project_Parameters when available.
    """

    if not RAW_DATA_FILE.exists():

        logger.warning(
            "Project workbook not found."
        )

        logger.warning(
            "Using fallback synthetic emission factor: %.4f",
            DEFAULT_SYNTHETIC_EMISSION_FACTOR
        )

        return DEFAULT_SYNTHETIC_EMISSION_FACTOR

    try:

        parameters = pd.read_excel(
            RAW_DATA_FILE,
            sheet_name="Project_Parameters"
        )

        parameter_columns = {
            str(column).strip().lower(): column
            for column in parameters.columns
        }

        parameter_column = parameter_columns.get(
            "parameter"
        )

        value_column = parameter_columns.get(
            "value"
        )

        if (
            parameter_column is None
            or value_column is None
        ):

            raise ValueError(
                "Project_Parameters must contain "
                "Parameter and Value columns."
            )

        matches = parameters[
            parameters[
                parameter_column
            ]
            .astype(str)
            .str.contains(
                "emission",
                case=False,
                na=False
            )
        ]

        if matches.empty:

            raise ValueError(
                "Synthetic emission factor was not found."
            )

        value = pd.to_numeric(
            matches.iloc[0][
                value_column
            ],
            errors="coerce"
        )

        if pd.isna(value) or value <= 0:

            raise ValueError(
                "Emission factor is invalid."
            )

        logger.info(
            "Synthetic emission factor loaded: "
            "%.4f kg CO2e/kWh",
            value
        )

        return float(value)

    except Exception as error:

        logger.warning(
            "Could not read project emission factor: %s",
            error
        )

        logger.warning(
            "Using fallback synthetic factor: %.4f",
            DEFAULT_SYNTHETIC_EMISSION_FACTOR
        )

        return DEFAULT_SYNTHETIC_EMISSION_FACTOR


# ============================================================================
# DATA PREPARATION
# ============================================================================

def prepare_co2_features(
    df,
    emission_factor
):
    """Prepare energy and CO2e analytical features."""

    logger.info(
        "Preparing CO2e analytical features..."
    )

    df = df.copy()

    required_columns = [
        "Record_ID",
        "Timestamp",
        "Machine_ID",
        "Machine_Name",
        "Machine_Type",
        "Actual_State",
        "Energy_Consumption_kWh",
        "Baseline_Energy_kWh",
        "Actual_Production_Units",
        "Idle_Energy_kWh"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Required columns missing: "
            + ", ".join(missing_columns)
        )

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        errors="coerce"
    )

    numeric_columns = [
        "Energy_Consumption_kWh",
        "Baseline_Energy_kWh",
        "Actual_Production_Units",
        "Idle_Energy_kWh"
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # ------------------------------------------------------------------------
    # Actual CO2e
    # ------------------------------------------------------------------------

    df["Calculated_CO2e_kg"] = (
        df["Energy_Consumption_kWh"]
        *
        emission_factor
    )

    # ------------------------------------------------------------------------
    # Baseline CO2e
    # ------------------------------------------------------------------------

    df["Baseline_CO2e_kg"] = (
        df["Baseline_Energy_kWh"]
        *
        emission_factor
    )

    # ------------------------------------------------------------------------
    # CO2e deviation
    # ------------------------------------------------------------------------

    df["CO2e_Deviation_kg"] = (
        df["Calculated_CO2e_kg"]
        -
        df["Baseline_CO2e_kg"]
    )

    df["CO2e_Deviation_Percent"] = np.where(
        df["Baseline_CO2e_kg"] > 0,
        (
            df["CO2e_Deviation_kg"]
            /
            df["Baseline_CO2e_kg"]
        )
        *
        100,
        np.nan
    )

    # ------------------------------------------------------------------------
    # Production-adjusted CO2e
    # ------------------------------------------------------------------------

    df["CO2e_Per_Production_Unit"] = np.where(
        df["Actual_Production_Units"] > 0,
        (
            df["Calculated_CO2e_kg"]
            /
            df["Actual_Production_Units"]
        ),
        np.nan
    )

    # ------------------------------------------------------------------------
    # Idle CO2e
    # ------------------------------------------------------------------------

    df["Idle_CO2e_kg"] = (
        df["Idle_Energy_kWh"]
        *
        emission_factor
    )

    # ------------------------------------------------------------------------
    # Date/time dimensions
    # ------------------------------------------------------------------------

    df["Date"] = (
        df["Timestamp"]
        .dt
        .date
    )

    df["Year"] = (
        df["Timestamp"]
        .dt
        .year
    )

    df["Month"] = (
        df["Timestamp"]
        .dt
        .month
    )

    df["Month_Name"] = (
        df["Timestamp"]
        .dt
        .strftime("%B")
    )

    df["Week"] = (
        df["Timestamp"]
        .dt
        .isocalendar()
        .week
        .astype(int)
    )

    logger.info(
        "CO2e analytical features prepared."
    )

    return df


# ============================================================================
# MACHINE-LEVEL CO2 ANALYSIS
# ============================================================================

def analyze_machine_co2(
    df
):
    """Calculate machine-level energy and CO2e metrics."""

    logger.info(
        "Analyzing machine-level CO2e emissions..."
    )

    result = (
        df
        .groupby(
            [
                "Machine_ID",
                "Machine_Name",
                "Machine_Type"
            ],
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
            Baseline_Energy_kWh=(
                "Baseline_Energy_kWh",
                "sum"
            ),
            Production_Units=(
                "Actual_Production_Units",
                "sum"
            ),
            Idle_Energy_kWh=(
                "Idle_Energy_kWh",
                "sum"
            ),
            CO2e_kg=(
                "Calculated_CO2e_kg",
                "sum"
            ),
            Baseline_CO2e_kg=(
                "Baseline_CO2e_kg",
                "sum"
            ),
            Idle_CO2e_kg=(
                "Idle_CO2e_kg",
                "sum"
            )
        )
    )

    result["CO2e_Deviation_kg"] = (
        result["CO2e_kg"]
        -
        result["Baseline_CO2e_kg"]
    )

    result["CO2e_Deviation_Percent"] = np.where(
        result["Baseline_CO2e_kg"] > 0,
        (
            result["CO2e_Deviation_kg"]
            /
            result["Baseline_CO2e_kg"]
        )
        *
        100,
        np.nan
    )

    result["CO2e_Per_Production_Unit"] = np.where(
        result["Production_Units"] > 0,
        (
            result["CO2e_kg"]
            /
            result["Production_Units"]
        ),
        np.nan
    )

    result["Energy_Share_Percent"] = (
        result["Energy_kWh"]
        /
        result["Energy_kWh"].sum()
        *
        100
    )

    result["CO2e_Share_Percent"] = (
        result["CO2e_kg"]
        /
        result["CO2e_kg"].sum()
        *
        100
    )

    result = result.sort_values(
        "CO2e_kg",
        ascending=False
    )

    result.to_csv(
        MACHINE_CO2_FILE,
        index=False
    )

    logger.info(
        "Machine-level CO2e analysis generated."
    )

    return result


# ============================================================================
# DAILY CO2 ANALYSIS
# ============================================================================

def analyze_daily_co2(
    df
):
    """Create daily energy and CO2e analysis."""

    logger.info(
        "Analyzing daily CO2e emissions..."
    )

    result = (
        df
        .groupby(
            "Date",
            as_index=False
        )
        .agg(
            Energy_kWh=(
                "Energy_Consumption_kWh",
                "sum"
            ),
            Baseline_Energy_kWh=(
                "Baseline_Energy_kWh",
                "sum"
            ),
            Production_Units=(
                "Actual_Production_Units",
                "sum"
            ),
            Idle_Energy_kWh=(
                "Idle_Energy_kWh",
                "sum"
            ),
            CO2e_kg=(
                "Calculated_CO2e_kg",
                "sum"
            ),
            Baseline_CO2e_kg=(
                "Baseline_CO2e_kg",
                "sum"
            ),
            Idle_CO2e_kg=(
                "Idle_CO2e_kg",
                "sum"
            )
        )
    )

    result["CO2e_Deviation_kg"] = (
        result["CO2e_kg"]
        -
        result["Baseline_CO2e_kg"]
    )

    result["CO2e_Per_Production_Unit"] = np.where(
        result["Production_Units"] > 0,
        (
            result["CO2e_kg"]
            /
            result["Production_Units"]
        ),
        np.nan
    )

    result["Idle_CO2e_Percent"] = np.where(
        result["CO2e_kg"] > 0,
        (
            result["Idle_CO2e_kg"]
            /
            result["CO2e_kg"]
        )
        *
        100,
        0
    )

    result.to_csv(
        DAILY_CO2_FILE,
        index=False
    )

    logger.info(
        "Daily CO2e analysis generated: %s days.",
        len(result)
    )

    return result


# ============================================================================
# MONTHLY CO2 ANALYSIS
# ============================================================================

def analyze_monthly_co2(
    df
):
    """Create monthly sustainability metrics."""

    logger.info(
        "Analyzing monthly CO2e emissions..."
    )

    result = (
        df
        .groupby(
            [
                "Year",
                "Month",
                "Month_Name"
            ],
            as_index=False
        )
        .agg(
            Energy_kWh=(
                "Energy_Consumption_kWh",
                "sum"
            ),
            Baseline_Energy_kWh=(
                "Baseline_Energy_kWh",
                "sum"
            ),
            Production_Units=(
                "Actual_Production_Units",
                "sum"
            ),
            CO2e_kg=(
                "Calculated_CO2e_kg",
                "sum"
            ),
            Baseline_CO2e_kg=(
                "Baseline_CO2e_kg",
                "sum"
            ),
            Idle_CO2e_kg=(
                "Idle_CO2e_kg",
                "sum"
            )
        )
    )

    result["CO2e_Deviation_kg"] = (
        result["CO2e_kg"]
        -
        result["Baseline_CO2e_kg"]
    )

    result["CO2e_Per_Production_Unit"] = np.where(
        result["Production_Units"] > 0,
        (
            result["CO2e_kg"]
            /
            result["Production_Units"]
        ),
        np.nan
    )

    result.to_csv(
        MONTHLY_CO2_FILE,
        index=False
    )

    logger.info(
        "Monthly CO2e analysis generated."
    )

    return result


# ============================================================================
# MACHINE STATE CO2 ANALYSIS
# ============================================================================

def analyze_state_co2(
    df
):
    """Analyze CO2e emissions by machine operating state."""

    logger.info(
        "Analyzing CO2e by machine operating state..."
    )

    result = (
        df
        .groupby(
            "Actual_State",
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
            CO2e_kg=(
                "Calculated_CO2e_kg",
                "sum"
            ),
            Idle_Energy_kWh=(
                "Idle_Energy_kWh",
                "sum"
            ),
            Idle_CO2e_kg=(
                "Idle_CO2e_kg",
                "sum"
            )
        )
    )

    result["CO2e_Share_Percent"] = (
        result["CO2e_kg"]
        /
        result["CO2e_kg"].sum()
        *
        100
    )

    result["CO2e_Per_Production_Unit"] = np.where(
        result["Production_Units"] > 0,
        (
            result["CO2e_kg"]
            /
            result["Production_Units"]
        ),
        np.nan
    )

    result.to_csv(
        STATE_CO2_FILE,
        index=False
    )

    logger.info(
        "Machine-state CO2e analysis generated."
    )

    return result


# ============================================================================
# CO2 INTENSITY ANALYSIS
# ============================================================================

def analyze_co2_intensity(
    df
):
    """
    Analyze carbon intensity relative to production.

    This is especially important because absolute energy or CO2e should
    not be interpreted without considering production output.
    """

    logger.info(
        "Analyzing production-adjusted CO2 intensity..."
    )

    result = (
        df
        .groupby(
            [
                "Machine_ID",
                "Machine_Name"
            ],
            as_index=False
        )
        .agg(
            Energy_kWh=(
                "Energy_Consumption_kWh",
                "sum"
            ),
            Production_Units=(
                "Actual_Production_Units",
                "sum"
            ),
            CO2e_kg=(
                "Calculated_CO2e_kg",
                "sum"
            ),
            Baseline_CO2e_kg=(
                "Baseline_CO2e_kg",
                "sum"
            )
        )
    )

    result["Energy_Per_Unit_kWh"] = np.where(
        result["Production_Units"] > 0,
        (
            result["Energy_kWh"]
            /
            result["Production_Units"]
        ),
        np.nan
    )

    result["CO2e_Per_Unit_kg"] = np.where(
        result["Production_Units"] > 0,
        (
            result["CO2e_kg"]
            /
            result["Production_Units"]
        ),
        np.nan
    )

    result["Baseline_CO2e_Per_Unit_kg"] = np.where(
        result["Production_Units"] > 0,
        (
            result["Baseline_CO2e_kg"]
            /
            result["Production_Units"]
        ),
        np.nan
    )

    result["CO2e_Intensity_Deviation_Percent"] = np.where(
        result["Baseline_CO2e_Per_Unit_kg"] > 0,
        (
            (
                result["CO2e_Per_Unit_kg"]
                -
                result["Baseline_CO2e_Per_Unit_kg"]
            )
            /
            result["Baseline_CO2e_Per_Unit_kg"]
        )
        *
        100,
        np.nan
    )

    result = result.sort_values(
        "CO2e_Per_Unit_kg",
        ascending=False
    )

    result.to_csv(
        CO2_INTENSITY_FILE,
        index=False
    )

    logger.info(
        "CO2 intensity analysis generated."
    )

    return result


# ============================================================================
# SUSTAINABILITY KPI GENERATION
# ============================================================================

def create_sustainability_kpis(
    df,
    machine_co2,
    daily_co2,
    scenario_summary,
    emission_factor
):
    """Create executive-level sustainability KPIs."""

    logger.info(
        "Creating sustainability KPI dataset..."
    )

    total_energy = (
        df[
            "Energy_Consumption_kWh"
        ]
        .sum()
    )

    total_baseline_energy = (
        df[
            "Baseline_Energy_kWh"
        ]
        .sum()
    )

    total_production = (
        df[
            "Actual_Production_Units"
        ]
        .sum()
    )

    total_co2 = (
        df[
            "Calculated_CO2e_kg"
        ]
        .sum()
    )

    baseline_co2 = (
        df[
            "Baseline_CO2e_kg"
        ]
        .sum()
    )

    total_idle_energy = (
        df[
            "Idle_Energy_kWh"
        ]
        .sum()
    )

    total_idle_co2 = (
        df[
            "Idle_CO2e_kg"
        ]
        .sum()
    )

    energy_deviation = (
        total_energy
        -
        total_baseline_energy
    )

    co2_deviation = (
        total_co2
        -
        baseline_co2
    )

    energy_per_unit = (
        total_energy
        /
        total_production
        if total_production > 0
        else np.nan
    )

    co2_per_unit = (
        total_co2
        /
        total_production
        if total_production > 0
        else np.nan
    )

    # ------------------------------------------------------------------------
    # Combined scenario
    # ------------------------------------------------------------------------

    combined = scenario_summary[
        scenario_summary["Scenario"]
        ==
        "COMBINED_IMPROVEMENT"
    ]

    if combined.empty:

        potential_energy_reduction = 0.0

        potential_co2_reduction = 0.0

        potential_reduction_percent = 0.0

    else:

        potential_energy_reduction = (
            combined.iloc[0][
                "Potential_Energy_Reduction_kWh"
            ]
        )

        potential_co2_reduction = (
            combined.iloc[0][
                "Potential_CO2_Reduction_kg"
            ]
        )

        potential_reduction_percent = (
            combined.iloc[0][
                "Potential_Energy_Reduction_Percent"
            ]
        )

    metrics = [

        (
            "Total Energy Consumption",
            total_energy,
            "kWh",
            "Observed"
        ),

        (
            "Baseline Energy",
            total_baseline_energy,
            "kWh",
            "Modelled Baseline"
        ),

        (
            "Energy Deviation",
            energy_deviation,
            "kWh",
            "Observed vs Baseline"
        ),

        (
            "Total Production",
            total_production,
            "Units",
            "Observed"
        ),

        (
            "Energy per Production Unit",
            energy_per_unit,
            "kWh/unit",
            "Observed"
        ),

        (
            "Total CO2e",
            total_co2,
            "kg CO2e",
            "Calculated"
        ),

        (
            "Baseline CO2e",
            baseline_co2,
            "kg CO2e",
            "Modelled Baseline"
        ),

        (
            "CO2e Deviation",
            co2_deviation,
            "kg CO2e",
            "Observed vs Baseline"
        ),

        (
            "CO2e per Production Unit",
            co2_per_unit,
            "kg CO2e/unit",
            "Observed"
        ),

        (
            "Idle Energy",
            total_idle_energy,
            "kWh",
            "Observed"
        ),

        (
            "Idle CO2e",
            total_idle_co2,
            "kg CO2e",
            "Calculated"
        ),

        (
            "Potential Combined Energy Reduction",
            potential_energy_reduction,
            "kWh",
            "Modelled Potential"
        ),

        (
            "Potential Combined Energy Reduction",
            potential_reduction_percent,
            "%",
            "Modelled Potential"
        ),

        (
            "Potential Combined CO2 Reduction",
            potential_co2_reduction,
            "kg CO2e",
            "Modelled Potential"
        ),

        (
            "Synthetic Emission Factor",
            emission_factor,
            "kg CO2e/kWh",
            "Synthetic / Demonstration Only"
        )
    ]

    result = pd.DataFrame(
        metrics,
        columns=[
            "KPI",
            "Value",
            "Unit",
            "Data_Status"
        ]
    )

    result.to_csv(
        SUSTAINABILITY_KPI_FILE,
        index=False
    )

    logger.info(
        "Sustainability KPI dataset generated."
    )

    return result


# ============================================================================
# SCENARIO CO2 ANALYSIS
# ============================================================================

def create_scenario_co2_analysis(
    scenario_summary
):
    """Prepare What-If scenario CO2e impact for dashboards."""

    logger.info(
        "Preparing scenario CO2e impact analysis..."
    )

    required_columns = [
        "Scenario",
        "Actual_CO2e_kg",
        "Scenario_CO2e_kg",
        "Potential_CO2_Reduction_kg",
        "Potential_Energy_Reduction_kWh",
        "Potential_Energy_Reduction_Percent",
        "Production_Output_Units",
        "Production_Maintained"
    ]

    missing = [
        column
        for column in required_columns
        if column not in scenario_summary.columns
    ]

    if missing:

        raise ValueError(
            "Scenario summary missing columns: "
            + ", ".join(missing)
        )

    result = scenario_summary[
        required_columns
    ].copy()

    result["Impact_Type"] = (
        "MODELLED POTENTIAL"
    )

    result["Interpretation"] = (
        "Potential scenario impact; "
        "not actual achieved savings."
    )

    result.to_csv(
        SCENARIO_CO2_FILE,
        index=False
    )

    logger.info(
        "Scenario CO2e analysis generated."
    )

    return result


# ============================================================================
# MACHINE SUSTAINABILITY RANKING
# ============================================================================

def create_machine_sustainability_ranking(
    machine_co2
):
    """
    Rank machines using multiple sustainability indicators.

    Lower carbon intensity is better.
    Higher idle CO2e percentage indicates greater potential attention.
    """

    logger.info(
        "Creating machine sustainability ranking..."
    )

    result = machine_co2.copy()

    total_co2 = (
        result[
            "CO2e_kg"
        ]
        .sum()
    )

    result["Idle_CO2e_Percent"] = np.where(
        result["CO2e_kg"] > 0,
        (
            result["Idle_CO2e_kg"]
            /
            result["CO2e_kg"]
        )
        *
        100,
        0
    )

    result["CO2e_Share_Percent"] = np.where(
        total_co2 > 0,
        (
            result["CO2e_kg"]
            /
            total_co2
        )
        *
        100,
        0
    )

    # ------------------------------------------------------------------------
    # Normalized indicators
    # ------------------------------------------------------------------------

    def normalize(series):

        minimum = series.min()

        maximum = series.max()

        if maximum == minimum:

            return pd.Series(
                0.0,
                index=series.index
            )

        return (
            series
            -
            minimum
        ) / (
            maximum
            -
            minimum
        )

    result["Carbon_Intensity_Score"] = normalize(
        result["CO2e_Per_Production_Unit"]
        .fillna(
            result["CO2e_Per_Production_Unit"]
            .median()
        )
    )

    result["Idle_Impact_Score"] = normalize(
        result["Idle_CO2e_Percent"]
    )

    result["CO2_Share_Score"] = normalize(
        result["CO2e_Share_Percent"]
    )

    # ------------------------------------------------------------------------
    # Composite attention score
    # ------------------------------------------------------------------------

    result["Sustainability_Attention_Score"] = (
        (
            result["Carbon_Intensity_Score"]
            *
            0.50
        )
        +
        (
            result["Idle_Impact_Score"]
            *
            0.30
        )
        +
        (
            result["CO2_Share_Score"]
            *
            0.20
        )
    ) * 100

    result["Sustainability_Priority"] = pd.cut(
        result[
            "Sustainability_Attention_Score"
        ],
        bins=[
            -np.inf,
            33.33,
            66.67,
            np.inf
        ],
        labels=[
            "LOW",
            "MEDIUM",
            "HIGH"
        ]
    )

    result = result.sort_values(
        "Sustainability_Attention_Score",
        ascending=False
    )

    result.to_csv(
        MACHINE_SUSTAINABILITY_FILE,
        index=False
    )

    logger.info(
        "Machine sustainability ranking generated."
    )

    return result


# ============================================================================
# VALIDATION
# ============================================================================

def validate_co2_analysis(
    df,
    machine_co2,
    daily_co2,
    scenario_summary,
    emission_factor
):
    """Perform final CO2 analysis validation."""

    logger.info(
        "Running CO2 and sustainability validation..."
    )

    validations = []

    # ------------------------------------------------------------------------
    # Record count
    # ------------------------------------------------------------------------

    validations.append({
        "Validation":
            "Record Count",

        "Expected":
            EXPECTED_RECORDS,

        "Actual":
            len(df),

        "Status":
            "PASS"
            if len(df) == EXPECTED_RECORDS
            else "FAIL"
    })

    # ------------------------------------------------------------------------
    # Machine count
    # ------------------------------------------------------------------------

    machine_count = (
        df[
            "Machine_ID"
        ]
        .nunique()
    )

    validations.append({
        "Validation":
            "Machine Count",

        "Expected":
            5,

        "Actual":
            machine_count,

        "Status":
            "PASS"
            if machine_count == 5
            else "FAIL"
    })

    # ------------------------------------------------------------------------
    # Negative energy
    # ------------------------------------------------------------------------

    negative_energy = int(
        (
            df[
                "Energy_Consumption_kWh"
            ]
            < 0
        ).sum()
    )

    validations.append({
        "Validation":
            "Negative Energy Values",

        "Expected":
            0,

        "Actual":
            negative_energy,

        "Status":
            "PASS"
            if negative_energy == 0
            else "FAIL"
    })

    # ------------------------------------------------------------------------
    # Negative CO2
    # ------------------------------------------------------------------------

    negative_co2 = int(
        (
            df[
                "Calculated_CO2e_kg"
            ]
            < 0
        ).sum()
    )

    validations.append({
        "Validation":
            "Negative CO2e Values",

        "Expected":
            0,

        "Actual":
            negative_co2,

        "Status":
            "PASS"
            if negative_co2 == 0
            else "FAIL"
    })

    # ------------------------------------------------------------------------
    # CO2 formula validation
    # ------------------------------------------------------------------------

    expected_co2 = (
        df[
            "Energy_Consumption_kWh"
        ]
        *
        emission_factor
    )

    calculation_errors = int(
        (
            ~np.isclose(
                df[
                    "Calculated_CO2e_kg"
                ],
                expected_co2,
                atol=0.01
            )
        ).sum()
    )

    validations.append({
        "Validation":
            "CO2e Formula",

        "Expected":
            0,

        "Actual":
            calculation_errors,

        "Status":
            "PASS"
            if calculation_errors == 0
            else "FAIL"
    })

    # ------------------------------------------------------------------------
    # Machine aggregation
    # ------------------------------------------------------------------------

    machine_energy = (
        machine_co2[
            "Energy_kWh"
        ]
        .sum()
    )

    total_energy = (
        df[
            "Energy_Consumption_kWh"
        ]
        .sum()
    )

    machine_aggregation_valid = np.isclose(
        machine_energy,
        total_energy,
        atol=0.01
    )

    validations.append({
        "Validation":
            "Machine Energy Aggregation",

        "Expected":
            round(total_energy, 2),

        "Actual":
            round(machine_energy, 2),

        "Status":
            "PASS"
            if machine_aggregation_valid
            else "FAIL"
    })

    # ------------------------------------------------------------------------
    # Daily aggregation
    # ------------------------------------------------------------------------

    daily_energy = (
        daily_co2[
            "Energy_kWh"
        ]
        .sum()
    )

    daily_aggregation_valid = np.isclose(
        daily_energy,
        total_energy,
        atol=0.01
    )

    validations.append({
        "Validation":
            "Daily Energy Aggregation",

        "Expected":
            round(total_energy, 2),

        "Actual":
            round(daily_energy, 2),

        "Status":
            "PASS"
            if daily_aggregation_valid
            else "FAIL"
    })

    # ------------------------------------------------------------------------
    # Scenario count
    # ------------------------------------------------------------------------

    scenario_count = (
        len(scenario_summary)
    )

    validations.append({
        "Validation":
            "Scenario Count",

        "Expected":
            5,

        "Actual":
            scenario_count,

        "Status":
            "PASS"
            if scenario_count == 5
            else "FAIL"
    })

    # ------------------------------------------------------------------------
    # Scenario production maintained
    # ------------------------------------------------------------------------

    production_failures = int(
        (
            scenario_summary[
                "Production_Maintained"
            ]
            != True
        ).sum()
    )

    validations.append({
        "Validation":
            "Scenario Production Maintained",

        "Expected":
            0,

        "Actual":
            production_failures,

        "Status":
            "PASS"
            if production_failures == 0
            else "FAIL"
    })

    # ------------------------------------------------------------------------
    # Emission factor
    # ------------------------------------------------------------------------

    validations.append({
        "Validation":
            "Emission Factor",

        "Expected":
            "> 0",

        "Actual":
            round(
                emission_factor,
                4
            ),

        "Status":
            "PASS"
            if emission_factor > 0
            else "FAIL"
    })

    # ------------------------------------------------------------------------
    # Final validation table
    # ------------------------------------------------------------------------

    validation = pd.DataFrame(
        validations
    )

    validation.to_csv(
        VALIDATION_FILE,
        index=False
    )

    return validation


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Execute the complete CO2 and sustainability analysis."""

    logger.info("=" * 78)

    logger.info(
        "SUSTAINABLE MANUFACTURING - CO2 & SUSTAINABILITY ANALYSIS"
    )

    logger.info("=" * 78)

    try:

        # --------------------------------------------------------------------
        # Initialize
        # --------------------------------------------------------------------

        initialize_directories()

        # --------------------------------------------------------------------
        # Load datasets
        # --------------------------------------------------------------------

        df = load_baseline_dataset()

        scenario_summary = (
            load_what_if_summary()
        )

        # --------------------------------------------------------------------
        # Load emission factor
        # --------------------------------------------------------------------

        emission_factor = (
            load_emission_factor()
        )

        # --------------------------------------------------------------------
        # Prepare analytical features
        # --------------------------------------------------------------------

        df = prepare_co2_features(
            df,
            emission_factor
        )

        # --------------------------------------------------------------------
        # Machine-level analysis
        # --------------------------------------------------------------------

        machine_co2 = (
            analyze_machine_co2(
                df
            )
        )

        # --------------------------------------------------------------------
        # Daily analysis
        # --------------------------------------------------------------------

        daily_co2 = (
            analyze_daily_co2(
                df
            )
        )

        # --------------------------------------------------------------------
        # Monthly analysis
        # --------------------------------------------------------------------

        monthly_co2 = (
            analyze_monthly_co2(
                df
            )
        )

        # --------------------------------------------------------------------
        # State analysis
        # --------------------------------------------------------------------

        state_co2 = (
            analyze_state_co2(
                df
            )
        )

        # --------------------------------------------------------------------
        # Carbon intensity
        # --------------------------------------------------------------------

        intensity = (
            analyze_co2_intensity(
                df
            )
        )

        # --------------------------------------------------------------------
        # Sustainability KPIs
        # --------------------------------------------------------------------

        kpis = (
            create_sustainability_kpis(
                df,
                machine_co2,
                daily_co2,
                scenario_summary,
                emission_factor
            )
        )

        # --------------------------------------------------------------------
        # Scenario CO2 impact
        # --------------------------------------------------------------------

        scenario_co2 = (
            create_scenario_co2_analysis(
                scenario_summary
            )
        )

        # --------------------------------------------------------------------
        # Machine sustainability ranking
        # --------------------------------------------------------------------

        ranking = (
            create_machine_sustainability_ranking(
                machine_co2
            )
        )

        # --------------------------------------------------------------------
        # Validation
        # --------------------------------------------------------------------

        validation = (
            validate_co2_analysis(
                df,
                machine_co2,
                daily_co2,
                scenario_summary,
                emission_factor
            )
        )

        validation_failures = int(
            (
                validation[
                    "Status"
                ]
                ==
                "FAIL"
            ).sum()
        )

        # --------------------------------------------------------------------
        # Summary calculations
        # --------------------------------------------------------------------

        total_energy = (
            df[
                "Energy_Consumption_kWh"
            ]
            .sum()
        )

        total_co2 = (
            df[
                "Calculated_CO2e_kg"
            ]
            .sum()
        )

        total_production = (
            df[
                "Actual_Production_Units"
            ]
            .sum()
        )

        total_idle_co2 = (
            df[
                "Idle_CO2e_kg"
            ]
            .sum()
        )

        combined = scenario_summary[
            scenario_summary[
                "Scenario"
            ]
            ==
            "COMBINED_IMPROVEMENT"
        ]

        if not combined.empty:

            potential_co2_reduction = (
                combined.iloc[0][
                    "Potential_CO2_Reduction_kg"
                ]
            )

        else:

            potential_co2_reduction = 0.0

        # --------------------------------------------------------------------
        # Final log
        # --------------------------------------------------------------------

        logger.info("=" * 78)

        if validation_failures == 0:

            logger.info(
                "CO2 & SUSTAINABILITY ANALYSIS COMPLETED SUCCESSFULLY"
            )

        else:

            logger.warning(
                "CO2 ANALYSIS COMPLETED WITH %s VALIDATION FAILURES",
                validation_failures
            )

        logger.info("=" * 78)

        logger.info(
            "Records analyzed        : %s",
            len(df)
        )

        logger.info(
            "Machines analyzed       : %s",
            df["Machine_ID"].nunique()
        )

        logger.info(
            "Total energy            : %.2f kWh",
            total_energy
        )

        logger.info(
            "Total production        : %.2f units",
            total_production
        )

        logger.info(
            "Total CO2e              : %.2f kg",
            total_co2
        )

        logger.info(
            "Idle CO2e               : %.2f kg",
            total_idle_co2
        )

        logger.info(
            "Potential combined CO2e reduction : %.2f kg",
            potential_co2_reduction
        )

        logger.info(
            "Synthetic emission factor: %.4f kg CO2e/kWh",
            emission_factor
        )

        logger.info(
            "Output directory        : %s",
            CO2_DIR
        )

        # --------------------------------------------------------------------
        # Console KPI summary
        # --------------------------------------------------------------------

        print()

        print("=" * 78)

        print(
            "SUSTAINABILITY KPI SUMMARY"
        )

        print("=" * 78)

        print(
            kpis.to_string(
                index=False
            )
        )

        print()

        print(
            "MACHINE SUSTAINABILITY RANKING"
        )

        print("=" * 78)

        ranking_columns = [
            "Machine_ID",
            "Machine_Name",
            "CO2e_kg",
            "CO2e_Per_Production_Unit",
            "Idle_CO2e_Percent",
            "Sustainability_Attention_Score",
            "Sustainability_Priority"
        ]

        print(
            ranking[
                ranking_columns
            ].to_string(
                index=False
            )
        )

        print()

        print(
            "SCENARIO CO2e IMPACT"
        )

        print("=" * 78)

        print(
            scenario_co2.to_string(
                index=False
            )
        )

        print()

        print(
            "CO2 ANALYSIS VALIDATION"
        )

        print("=" * 78)

        print(
            validation.to_string(
                index=False
            )
        )

        print()

        print(
            "IMPORTANT:"
        )

        print(
            "The emission factor is SYNTHETIC / "
            "FOR DEMONSTRATION ONLY."
        )

        print(
            "Potential CO2 reductions are MODELLED "
            "SCENARIO ESTIMATES."
        )

        print(
            "They are NOT actual achieved emissions reductions."
        )

        print()

        print(
            f"CO2 outputs: {CO2_DIR}"
        )

        print("=" * 78)

        # --------------------------------------------------------------------
        # Exit if validation fails
        # --------------------------------------------------------------------

        if validation_failures > 0:

            sys.exit(1)

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
            "Unexpected error during CO2 analysis: %s",
            error
        )

        sys.exit(1)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()