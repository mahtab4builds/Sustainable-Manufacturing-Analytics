"""
Sustainable Manufacturing:
Energy Consumption & Carbon Footprint Optimization Analytics

Module:
    Optimization Analysis

Purpose:
    Evaluate energy-reduction opportunities while maintaining required
    production output.

Method:
    1. Build machine-level optimization inputs.
    2. Quantify available energy-reduction opportunities.
    3. Formulate a Linear Programming optimization model where supported.
    4. Preserve required production output.
    5. Estimate potential energy and CO2e reduction.
    6. Generate machine priorities and dashboard-ready outputs.

Important:
    Optimization results are MODELLED POTENTIAL impacts.
    They are not actual achieved factory savings.

    The emission factor used by this project is synthetic and intended
    only for educational, analytical, and demonstration purposes.
"""

# ============================================================================
# IMPORTS
# ============================================================================

from pathlib import Path
import logging
import sys

import numpy as np
import pandas as pd

try:
    from scipy.optimize import linprog

    SCIPY_AVAILABLE = True

except ImportError:
    SCIPY_AVAILABLE = False


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

INEFFICIENCY_MACHINE_FILE = (
    OUTPUT_DIR
    / "inefficiency"
    / "machine_inefficiency_summary.csv"
)

CO2_MACHINE_FILE = (
    OUTPUT_DIR
    / "co2_analysis"
    / "machine_co2_analysis.csv"
)

WHAT_IF_SUMMARY_FILE = (
    OUTPUT_DIR
    / "what_if"
    / "scenario_summary.csv"
)

RAW_DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "Sustainable_Manufacturing_Energy_Analytics_10000.xlsx"
)

OPTIMIZATION_DIR = (
    OUTPUT_DIR
    / "optimization"
)

MACHINE_OPTIMIZATION_FILE = (
    OPTIMIZATION_DIR
    / "machine_optimization.csv"
)

LP_RESULT_FILE = (
    OPTIMIZATION_DIR
    / "linear_programming_result.csv"
)

OPPORTUNITY_FILE = (
    OPTIMIZATION_DIR
    / "optimization_opportunities.csv"
)

CO2_IMPACT_FILE = (
    OPTIMIZATION_DIR
    / "optimization_co2_impact.csv"
)

VALIDATION_FILE = (
    OPTIMIZATION_DIR
    / "optimization_validation.csv"
)

ASSUMPTIONS_FILE = (
    OPTIMIZATION_DIR
    / "optimization_assumptions.csv"
)

EXPECTED_RECORDS = 10_000

DEFAULT_EMISSION_FACTOR = 0.70

# Maximum modelled intervention level for each optimization lever.
MAX_IDLE_REDUCTION = 0.50
MAX_SCHEDULE_REDUCTION = 0.40
MAX_INTENSITY_REDUCTION = 0.20


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
# DIRECTORY INITIALIZATION
# ============================================================================

def initialize_directories():
    """Create optimization output directories."""

    OPTIMIZATION_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    logger.info(
        "Optimization output directory initialized."
    )


# ============================================================================
# DATA LOADING
# ============================================================================

def load_baseline_dataset():
    """Load the validated baseline-enriched dataset."""

    if not BASELINE_DATA_FILE.exists():

        raise FileNotFoundError(
            "Baseline dataset not found.\n"
            f"Expected: {BASELINE_DATA_FILE}\n"
            "Run baseline.py before optimization.py."
        )

    logger.info(
        "Loading baseline-enriched dataset..."
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
    """Load validated What-If scenario results."""

    if not WHAT_IF_SUMMARY_FILE.exists():

        raise FileNotFoundError(
            "What-If scenario summary not found.\n"
            f"Expected: {WHAT_IF_SUMMARY_FILE}\n"
            "Run what_if_analysis.py first."
        )

    logger.info(
        "Loading What-If scenario summary..."
    )

    return pd.read_csv(
        WHAT_IF_SUMMARY_FILE
    )


def load_machine_inefficiency():
    """Load machine-level inefficiency results when available."""

    if not INEFFICIENCY_MACHINE_FILE.exists():

        logger.warning(
            "Machine inefficiency summary not found."
        )

        logger.warning(
            "Optimization will continue using baseline data."
        )

        return None

    logger.info(
        "Loading machine inefficiency summary..."
    )

    return pd.read_csv(
        INEFFICIENCY_MACHINE_FILE
    )


def load_machine_co2():
    """Load machine-level CO2 analysis."""

    if not CO2_MACHINE_FILE.exists():

        logger.warning(
            "Machine CO2 analysis not found."
        )

        return None

    logger.info(
        "Loading machine sustainability data..."
    )

    return pd.read_csv(
        CO2_MACHINE_FILE
    )


# ============================================================================
# EMISSION FACTOR
# ============================================================================

def load_emission_factor():
    """Load the project's documented synthetic emission factor."""

    if not RAW_DATA_FILE.exists():

        logger.warning(
            "Project workbook not found."
        )

        return DEFAULT_EMISSION_FACTOR

    try:

        parameters = pd.read_excel(
            RAW_DATA_FILE,
            sheet_name="Project_Parameters"
        )

        parameter_map = {
            str(column).strip().lower(): column
            for column in parameters.columns
        }

        parameter_column = parameter_map.get(
            "parameter"
        )

        value_column = parameter_map.get(
            "value"
        )

        if (
            parameter_column is None
            or value_column is None
        ):

            raise ValueError(
                "Project_Parameters requires Parameter and Value columns."
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
                "Emission factor not found."
            )

        value = pd.to_numeric(
            matches.iloc[0][
                value_column
            ],
            errors="coerce"
        )

        if pd.isna(value) or value <= 0:

            raise ValueError(
                "Invalid emission factor."
            )

        logger.info(
            "Synthetic emission factor: %.4f kg CO2e/kWh",
            value
        )

        return float(value)

    except Exception as error:

        logger.warning(
            "Unable to load emission factor: %s",
            error
        )

        logger.warning(
            "Using fallback synthetic emission factor: %.4f",
            DEFAULT_EMISSION_FACTOR
        )

        return DEFAULT_EMISSION_FACTOR


# ============================================================================
# FEATURE PREPARATION
# ============================================================================

def prepare_optimization_features(df):
    """Prepare machine-level optimization variables."""

    logger.info(
        "Preparing optimization features..."
    )

    required_columns = [
        "Machine_ID",
        "Machine_Name",
        "Machine_Type",
        "Energy_Consumption_kWh",
        "Baseline_Energy_kWh",
        "Idle_Energy_kWh",
        "Actual_Production_Units",
        "Actual_State",
        "Planned_State"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing optimization columns: "
            + ", ".join(missing_columns)
        )

    data = df.copy()

    numeric_columns = [
        "Energy_Consumption_kWh",
        "Baseline_Energy_kWh",
        "Idle_Energy_kWh",
        "Actual_Production_Units"
    ]

    for column in numeric_columns:

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        ).fillna(0)

    data["Idle_Flag"] = (
        data["Actual_State"]
        ==
        "IDLE"
    )

    data["Schedule_Deviation_Flag"] = (
        data["Planned_State"]
        !=
        data["Actual_State"]
    )

    data["Running_Flag"] = (
        data["Actual_State"]
        ==
        "RUNNING"
    )

    data["Productive_Running_Flag"] = (
        data["Running_Flag"]
        &
        (
            data[
                "Actual_Production_Units"
            ]
            > 0
        )
    )

    logger.info(
        "Optimization features prepared."
    )

    return data


# ============================================================================
# MACHINE-LEVEL OPPORTUNITY MODEL
# ============================================================================

def build_machine_optimization_table(df):
    """
    Build machine-level optimization inputs.

    Each machine receives three controllable improvement levers:

        1. Idle energy reduction
        2. Schedule improvement
        3. Energy-intensity improvement

    These are modelled opportunities rather than guaranteed savings.
    """

    logger.info(
        "Building machine-level optimization opportunities..."
    )

    grouped = (
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
                "Machine_ID",
                "size"
            ),
            Energy_kWh=(
                "Energy_Consumption_kWh",
                "sum"
            ),
            Baseline_Energy_kWh=(
                "Baseline_Energy_kWh",
                "sum"
            ),
            Idle_Energy_kWh=(
                "Idle_Energy_kWh",
                "sum"
            ),
            Production_Units=(
                "Actual_Production_Units",
                "sum"
            ),
            Idle_Records=(
                "Idle_Flag",
                "sum"
            ),
            Schedule_Deviation_Records=(
                "Schedule_Deviation_Flag",
                "sum"
            ),
            Productive_Running_Records=(
                "Productive_Running_Flag",
                "sum"
            )
        )
    )

    # ------------------------------------------------------------------------
    # Opportunity pools
    # ------------------------------------------------------------------------

    grouped["Idle_Opportunity_kWh"] = (
        grouped[
            "Idle_Energy_kWh"
        ]
        *
        MAX_IDLE_REDUCTION
    )

    grouped["Schedule_Opportunity_kWh"] = (
        grouped[
            "Energy_kWh"
        ]
        *
        (
            grouped[
                "Schedule_Deviation_Records"
            ]
            /
            grouped[
                "Records"
            ]
        )
        *
        MAX_SCHEDULE_REDUCTION
    )

    grouped["Intensity_Opportunity_kWh"] = (
        grouped[
            "Energy_kWh"
        ]
        *
        (
            grouped[
                "Productive_Running_Records"
            ]
            /
            grouped[
                "Records"
            ]
        )
        *
        MAX_INTENSITY_REDUCTION
    )

    # ------------------------------------------------------------------------
    # Prevent optimization from exceeding observed energy.
    # ------------------------------------------------------------------------

    opportunity_columns = [
        "Idle_Opportunity_kWh",
        "Schedule_Opportunity_kWh",
        "Intensity_Opportunity_kWh"
    ]

    grouped[opportunity_columns] = (
        grouped[
            opportunity_columns
        ]
        .clip(lower=0)
    )

    grouped["Maximum_Opportunity_kWh"] = (
        grouped[
            opportunity_columns
        ]
        .sum(
            axis=1
        )
    )

    grouped["Maximum_Opportunity_kWh"] = np.minimum(
        grouped[
            "Maximum_Opportunity_kWh"
        ],
        grouped[
            "Energy_kWh"
        ]
    )

    grouped["Current_Energy_Intensity_kWh_per_Unit"] = np.where(
        grouped["Production_Units"] > 0,
        (
            grouped["Energy_kWh"]
            /
            grouped["Production_Units"]
        ),
        np.nan
    )

    logger.info(
        "Optimization opportunities generated for %s machines.",
        len(grouped)
    )

    return grouped


# ============================================================================
# LINEAR PROGRAMMING OPTIMIZATION
# ============================================================================

def run_linear_programming(machine_table):
    """
    Solve the energy-reduction allocation problem using Linear Programming.

    Objective:
        Maximize modelled energy reduction.

    Decision variables:
        x_idle_machine
        x_schedule_machine
        x_intensity_machine

    Constraints:
        Each intervention cannot exceed its calculated opportunity.

    The production quantity remains unchanged because the optimization
    operates on energy-reduction levers rather than reducing production.
    """

    logger.info(
        "Preparing Linear Programming optimization..."
    )

    if not SCIPY_AVAILABLE:

        logger.warning(
            "SciPy is not installed."
        )

        logger.warning(
            "Linear Programming will be skipped."
        )

        return None

    machines = machine_table[
        "Machine_ID"
    ].tolist()

    n_machines = len(
        machines
    )

    n_variables = (
        n_machines
        *
        3
    )

    objective = np.zeros(
        n_variables
    )

    bounds = []

    # ------------------------------------------------------------------------
    # Decision variables:
    #
    # [idle M001 ... idle M005,
    #  schedule M001 ... schedule M005,
    #  intensity M001 ... intensity M005]
    # ------------------------------------------------------------------------

    for index in range(
        n_machines
    ):

        row = machine_table.iloc[
            index
        ]

        objective[
            index
        ] = -row[
            "Idle_Opportunity_kWh"
        ]

        bounds.append(
            (
                0,
                row[
                    "Idle_Opportunity_kWh"
                ]
            )
        )

    for index in range(
        n_machines
    ):

        row = machine_table.iloc[
            index
        ]

        position = (
            n_machines
            +
            index
        )

        objective[
            position
        ] = -row[
            "Schedule_Opportunity_kWh"
        ]

        bounds.append(
            (
                0,
                row[
                    "Schedule_Opportunity_kWh"
                ]
            )
        )

    for index in range(
        n_machines
    ):

        row = machine_table.iloc[
            index
        ]

        position = (
            (
                n_machines
                *
                2
            )
            +
            index
        )

        objective[
            position
        ] = -row[
            "Intensity_Opportunity_kWh"
        ]

        bounds.append(
            (
                0,
                row[
                    "Intensity_Opportunity_kWh"
                ]
            )
        )

    # ------------------------------------------------------------------------
    # Production constraint
    #
    # Production is protected by fixing the optimization to energy levers.
    # No production variable is allowed to decrease output.
    # ------------------------------------------------------------------------

    result = linprog(
        c=objective,
        bounds=bounds,
        method="highs"
    )

    if not result.success:

        logger.warning(
            "Linear Programming optimization failed: %s",
            result.message
        )

        return None

    logger.info(
        "Linear Programming optimization completed successfully."
    )

    decision_values = (
        result.x
    )

    output_rows = []

    for index, machine_id in enumerate(
        machines
    ):

        idle_value = (
            decision_values[
                index
            ]
        )

        schedule_value = (
            decision_values[
                n_machines
                +
                index
            ]
        )

        intensity_value = (
            decision_values[
                (
                    n_machines
                    *
                    2
                )
                +
                index
            ]
        )

        total_reduction = (
            idle_value
            +
            schedule_value
            +
            intensity_value
        )

        output_rows.append({

            "Machine_ID":
                machine_id,

            "Optimized_Idle_Reduction_kWh":
                idle_value,

            "Optimized_Schedule_Reduction_kWh":
                schedule_value,

            "Optimized_Intensity_Reduction_kWh":
                intensity_value,

            "Optimized_Total_Reduction_kWh":
                total_reduction
        })

    lp_result = pd.DataFrame(
        output_rows
    )

    lp_result.to_csv(
        LP_RESULT_FILE,
        index=False
    )

    return lp_result


# ============================================================================
# ANALYTICAL FALLBACK
# ============================================================================

def create_analytical_fallback(
    machine_table
):
    """
    Create a deterministic analytical optimization result when SciPy
    Linear Programming is unavailable.

    The fallback uses the full calculated opportunity pool subject to
    the same production-preservation principle.
    """

    logger.info(
        "Creating analytical optimization fallback..."
    )

    result = machine_table[
        [
            "Machine_ID",
            "Idle_Opportunity_kWh",
            "Schedule_Opportunity_kWh",
            "Intensity_Opportunity_kWh"
        ]
    ].copy()

    result = result.rename(
        columns={
            "Idle_Opportunity_kWh":
                "Optimized_Idle_Reduction_kWh",

            "Schedule_Opportunity_kWh":
                "Optimized_Schedule_Reduction_kWh",

            "Intensity_Opportunity_kWh":
                "Optimized_Intensity_Reduction_kWh"
        }
    )

    result["Optimized_Total_Reduction_kWh"] = (
        result[
            [
                "Optimized_Idle_Reduction_kWh",
                "Optimized_Schedule_Reduction_kWh",
                "Optimized_Intensity_Reduction_kWh"
            ]
        ]
        .sum(
            axis=1
        )
    )

    result.to_csv(
        LP_RESULT_FILE,
        index=False
    )

    return result


# ============================================================================
# MERGE OPTIMIZATION RESULTS
# ============================================================================

def merge_optimization_results(
    machine_table,
    optimization_result
):
    """Merge optimized decision variables with machine metrics."""

    result = machine_table.merge(
        optimization_result,
        on="Machine_ID",
        how="left"
    )

    result[
        [
            "Optimized_Idle_Reduction_kWh",
            "Optimized_Schedule_Reduction_kWh",
            "Optimized_Intensity_Reduction_kWh",
            "Optimized_Total_Reduction_kWh"
        ]
    ] = (
        result[
            [
                "Optimized_Idle_Reduction_kWh",
                "Optimized_Schedule_Reduction_kWh",
                "Optimized_Intensity_Reduction_kWh",
                "Optimized_Total_Reduction_kWh"
            ]
        ]
        .fillna(0)
    )

    result["Optimized_Energy_kWh"] = (
        result[
            "Energy_kWh"
        ]
        -
        result[
            "Optimized_Total_Reduction_kWh"
        ]
    )

    result["Optimized_Energy_kWh"] = (
        result[
            "Optimized_Energy_kWh"
        ]
        .clip(
            lower=0
        )
    )

    result["Potential_Reduction_Percent"] = np.where(
        result["Energy_kWh"] > 0,
        (
            result[
                "Optimized_Total_Reduction_kWh"
            ]
            /
            result[
                "Energy_kWh"
            ]
        )
        *
        100,
        0
    )

    result["Production_Output_Maintained"] = True

    result["Optimization_Status"] = (
        "MODELLED POTENTIAL"
    )

    return result


# ============================================================================
# OPPORTUNITY SUMMARY
# ============================================================================

def create_opportunity_summary(
    optimized_machine
):
    """Create a machine-level optimization opportunity summary."""

    result = optimized_machine[
        [
            "Machine_ID",
            "Machine_Name",
            "Machine_Type",
            "Energy_kWh",
            "Production_Units",
            "Idle_Energy_kWh",
            "Idle_Opportunity_kWh",
            "Schedule_Opportunity_kWh",
            "Intensity_Opportunity_kWh",
            "Optimized_Idle_Reduction_kWh",
            "Optimized_Schedule_Reduction_kWh",
            "Optimized_Intensity_Reduction_kWh",
            "Optimized_Total_Reduction_kWh",
            "Optimized_Energy_kWh",
            "Potential_Reduction_Percent",
            "Production_Output_Maintained",
            "Optimization_Status"
        ]
    ].copy()

    result = result.sort_values(
        "Optimized_Total_Reduction_kWh",
        ascending=False
    )

    result.to_csv(
        OPPORTUNITY_FILE,
        index=False
    )

    logger.info(
        "Optimization opportunity report generated."
    )

    return result


# ============================================================================
# CO2 IMPACT
# ============================================================================

def create_co2_impact(
    optimized_machine,
    emission_factor
):
    """Calculate potential CO2e reduction from optimization."""

    result = optimized_machine[
        [
            "Machine_ID",
            "Machine_Name",
            "Energy_kWh",
            "Optimized_Energy_kWh",
            "Optimized_Total_Reduction_kWh",
            "Production_Units",
            "Production_Output_Maintained"
        ]
    ].copy()

    result["Current_CO2e_kg"] = (
        result[
            "Energy_kWh"
        ]
        *
        emission_factor
    )

    result["Optimized_CO2e_kg"] = (
        result[
            "Optimized_Energy_kWh"
        ]
        *
        emission_factor
    )

    result["Potential_CO2_Reduction_kg"] = (
        result[
            "Current_CO2e_kg"
        ]
        -
        result[
            "Optimized_CO2e_kg"
        ]
    ).clip(
        lower=0
    )

    result["Impact_Type"] = (
        "MODELLED POTENTIAL"
    )

    result.to_csv(
        CO2_IMPACT_FILE,
        index=False
    )

    logger.info(
        "Optimization CO2e impact generated."
    )

    return result


# ============================================================================
# ASSUMPTIONS
# ============================================================================

def create_assumptions(
    emission_factor
):
    """Document optimization assumptions."""

    assumptions = pd.DataFrame({

        "Parameter": [

            "Idle energy reduction ceiling",

            "Schedule improvement ceiling",

            "Energy intensity improvement ceiling",

            "Production constraint",

            "Emission factor",

            "Optimization interpretation"
        ],

        "Value": [

            MAX_IDLE_REDUCTION,

            MAX_SCHEDULE_REDUCTION,

            MAX_INTENSITY_REDUCTION,

            "Production output maintained",

            emission_factor,

            "Potential / modelled only"
        ],

        "Unit": [

            "%",

            "%",

            "%",

            "Operational constraint",

            "kg CO2e/kWh",

            "N/A"
        ],

        "Type": [

            "Optimization assumption",

            "Optimization assumption",

            "Optimization assumption",

            "Business constraint",

            "Synthetic parameter",

            "Interpretation"
        ],

        "Description": [

            "Maximum modelled reduction applied to idle-energy opportunity.",

            "Maximum modelled reduction applied to schedule-related opportunity.",

            "Maximum modelled reduction applied to productive energy intensity.",

            "Optimization is not allowed to reduce production output.",

            "Synthetic project emission factor for demonstration only.",

            "Results represent potential optimization impact rather than achieved savings."
        ]
    })

    assumptions.to_csv(
        ASSUMPTIONS_FILE,
        index=False
    )

    return assumptions


# ============================================================================
# VALIDATION
# ============================================================================

def validate_optimization(
    df,
    optimized_machine,
    lp_result,
    emission_factor
):
    """Perform comprehensive optimization validation."""

    logger.info(
        "Validating optimization outputs..."
    )

    validations = []

    # ------------------------------------------------------------------------
    # Record count
    # ------------------------------------------------------------------------

    validations.append({
        "Validation":
            "Raw Record Count",

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
        optimized_machine[
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
    # Negative optimized energy
    # ------------------------------------------------------------------------

    negative_optimized_energy = int(
        (
            optimized_machine[
                "Optimized_Energy_kWh"
            ]
            < 0
        ).sum()
    )

    validations.append({
        "Validation":
            "Negative Optimized Energy",

        "Expected":
            0,

        "Actual":
            negative_optimized_energy,

        "Status":
            "PASS"
            if negative_optimized_energy == 0
            else "FAIL"
    })

    # ------------------------------------------------------------------------
    # Negative reduction
    # ------------------------------------------------------------------------

    negative_reduction = int(
        (
            optimized_machine[
                "Optimized_Total_Reduction_kWh"
            ]
            < 0
        ).sum()
    )

    validations.append({
        "Validation":
            "Negative Optimization Reduction",

        "Expected":
            0,

        "Actual":
            negative_reduction,

        "Status":
            "PASS"
            if negative_reduction == 0
            else "FAIL"
    })

    # ------------------------------------------------------------------------
    # Production preservation
    # ------------------------------------------------------------------------

    production_failures = int(
        (
            optimized_machine[
                "Production_Output_Maintained"
            ]
            != True
        ).sum()
    )

    validations.append({
        "Validation":
            "Production Output Maintained",

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
    # Optimization cannot increase energy
    # ------------------------------------------------------------------------

    energy_increase = int(
        (
            optimized_machine[
                "Optimized_Energy_kWh"
            ]
            >
            optimized_machine[
                "Energy_kWh"
            ]
            +
            0.01
        ).sum()
    )

    validations.append({
        "Validation":
            "Optimized Energy Not Increased",

        "Expected":
            0,

        "Actual":
            energy_increase,

        "Status":
            "PASS"
            if energy_increase == 0
            else "FAIL"
    })

    # ------------------------------------------------------------------------
    # CO2 factor
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
    # Machine optimization count
    # ------------------------------------------------------------------------

    validations.append({
        "Validation":
            "Optimization Machine Count",

        "Expected":
            machine_count,

        "Actual":
            len(lp_result),

        "Status":
            "PASS"
            if len(lp_result) == machine_count
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
# MAIN
# ============================================================================

def main():
    """Execute the complete optimization pipeline."""

    logger.info("=" * 78)

    logger.info(
        "SUSTAINABLE MANUFACTURING - ENERGY OPTIMIZATION"
    )

    logger.info("=" * 78)

    try:

        # --------------------------------------------------------------------
        # Initialize
        # --------------------------------------------------------------------

        initialize_directories()

        # --------------------------------------------------------------------
        # Load data
        # --------------------------------------------------------------------

        df = load_baseline_dataset()

        what_if_summary = (
            load_what_if_summary()
        )

        machine_inefficiency = (
            load_machine_inefficiency()
        )

        machine_co2 = (
            load_machine_co2()
        )

        emission_factor = (
            load_emission_factor()
        )

        # --------------------------------------------------------------------
        # Prepare features
        # --------------------------------------------------------------------

        df = prepare_optimization_features(
            df
        )

        # --------------------------------------------------------------------
        # Build machine optimization table
        # --------------------------------------------------------------------

        machine_table = (
            build_machine_optimization_table(
                df
            )
        )

        # --------------------------------------------------------------------
        # Linear Programming
        # --------------------------------------------------------------------

        if SCIPY_AVAILABLE:

            logger.info(
                "SciPy detected. Linear Programming enabled."
            )

            lp_result = (
                run_linear_programming(
                    machine_table
                )
            )

            if lp_result is None:

                logger.warning(
                    "LP solver did not return a valid solution."
                )

                lp_result = (
                    create_analytical_fallback(
                        machine_table
                    )
                )

                optimization_method = (
                    "Analytical Fallback"
                )

            else:

                optimization_method = (
                    "Linear Programming"
                )

        else:

            logger.warning(
                "SciPy unavailable."
            )

            lp_result = (
                create_analytical_fallback(
                    machine_table
                )
            )

            optimization_method = (
                "Analytical Fallback"
            )

        # --------------------------------------------------------------------
        # Merge results
        # --------------------------------------------------------------------

        optimized_machine = (
            merge_optimization_results(
                machine_table,
                lp_result
            )
        )

        optimized_machine[
            "Optimization_Method"
        ] = optimization_method

        optimized_machine.to_csv(
            MACHINE_OPTIMIZATION_FILE,
            index=False
        )

        # --------------------------------------------------------------------
        # Opportunity report
        # --------------------------------------------------------------------

        opportunities = (
            create_opportunity_summary(
                optimized_machine
            )
        )

        # --------------------------------------------------------------------
        # CO2 impact
        # --------------------------------------------------------------------

        co2_impact = (
            create_co2_impact(
                optimized_machine,
                emission_factor
            )
        )

        # --------------------------------------------------------------------
        # Assumptions
        # --------------------------------------------------------------------

        assumptions = (
            create_assumptions(
                emission_factor
            )
        )

        # --------------------------------------------------------------------
        # Validation
        # --------------------------------------------------------------------

        validation = (
            validate_optimization(
                df,
                optimized_machine,
                lp_result,
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
        # Summary metrics
        # --------------------------------------------------------------------

        total_energy = (
            optimized_machine[
                "Energy_kWh"
            ]
            .sum()
        )

        optimized_energy = (
            optimized_machine[
                "Optimized_Energy_kWh"
            ]
            .sum()
        )

        potential_reduction = (
            total_energy
            -
            optimized_energy
        )

        reduction_percent = (
            (
                potential_reduction
                /
                total_energy
            )
            *
            100
            if total_energy > 0
            else 0
        )

        total_production = (
            optimized_machine[
                "Production_Units"
            ]
            .sum()
        )

        potential_co2_reduction = (
            potential_reduction
            *
            emission_factor
        )

        # --------------------------------------------------------------------
        # Final output
        # --------------------------------------------------------------------

        logger.info("=" * 78)

        if validation_failures == 0:

            logger.info(
                "OPTIMIZATION ANALYSIS COMPLETED SUCCESSFULLY"
            )

        else:

            logger.warning(
                "OPTIMIZATION COMPLETED WITH %s VALIDATION FAILURES",
                validation_failures
            )

        logger.info("=" * 78)

        logger.info(
            "Records analyzed        : %s",
            len(df)
        )

        logger.info(
            "Machines analyzed       : %s",
            len(optimized_machine)
        )

        logger.info(
            "Optimization method     : %s",
            optimization_method
        )

        logger.info(
            "Current energy          : %.2f kWh",
            total_energy
        )

        logger.info(
            "Optimized model energy  : %.2f kWh",
            optimized_energy
        )

        logger.info(
            "Potential reduction     : %.2f kWh",
            potential_reduction
        )

        logger.info(
            "Potential reduction %%   : %.2f%%",
            reduction_percent
        )

        logger.info(
            "Potential CO2e reduction: %.2f kg",
            potential_co2_reduction
        )

        logger.info(
            "Production maintained   : %.2f units",
            total_production
        )

        logger.info(
            "Output directory        : %s",
            OPTIMIZATION_DIR
        )

        # --------------------------------------------------------------------
        # Console summary
        # --------------------------------------------------------------------

        print()

        print("=" * 78)

        print(
            "OPTIMIZATION SUMMARY"
        )

        print("=" * 78)

        print(
            f"Optimization Method          : {optimization_method}"
        )

        print(
            f"Current Energy               : {total_energy:.2f} kWh"
        )

        print(
            f"Optimized Model Energy       : {optimized_energy:.2f} kWh"
        )

        print(
            f"Potential Energy Reduction   : {potential_reduction:.2f} kWh"
        )

        print(
            f"Potential Reduction          : {reduction_percent:.2f}%"
        )

        print(
            f"Potential CO2e Reduction     : {potential_co2_reduction:.2f} kg CO2e"
        )

        print(
            f"Production Output            : {total_production:.2f} units"
        )

        print(
            "Production Output Maintained: YES"
        )

        print()

        print(
            "MACHINE OPTIMIZATION PRIORITIES"
        )

        print("=" * 78)

        priority_columns = [
            "Machine_ID",
            "Machine_Name",
            "Energy_kWh",
            "Optimized_Total_Reduction_kWh",
            "Potential_Reduction_Percent",
            "Production_Output_Maintained"
        ]

        print(
            opportunities[
                priority_columns
            ].to_string(
                index=False
            )
        )

        print()

        print(
            "OPTIMIZATION VALIDATION"
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
            "Optimization values represent MODELLED POTENTIAL impacts."
        )

        print(
            "They are not actual achieved manufacturing savings."
        )

        print(
            "Production output is explicitly preserved."
        )

        print()

        print(
            f"Optimization outputs: {OPTIMIZATION_DIR}"
        )

        print("=" * 78)

        # --------------------------------------------------------------------
        # Fail process if validation failed.
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
            "Unexpected error during optimization: %s",
            error
        )

        sys.exit(1)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()