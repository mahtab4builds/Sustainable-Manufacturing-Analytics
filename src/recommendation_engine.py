"""
Sustainable Manufacturing:
Energy Consumption & Carbon Footprint Optimization Analytics

Module:
    Recommendation Engine

Purpose:
    Convert analytical findings from baseline, inefficiency, What-If,
    CO2 and optimization analysis into actionable operational
    recommendations.

Important:
    Recommendations are generated from the synthetic project dataset.
    They represent analytical decision-support suggestions and should
    be validated against actual plant conditions before implementation.
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

BASELINE_FILE = (
    OUTPUT_DIR
    / "baseline"
    / "baseline_enriched_dataset.csv"
)

INEFFICIENCY_DIR = (
    OUTPUT_DIR
    / "inefficiency"
)

OPTIMIZATION_DIR = (
    OUTPUT_DIR
    / "optimization"
)

WHAT_IF_DIR = (
    OUTPUT_DIR
    / "what_if"
)

CO2_DIR = (
    OUTPUT_DIR
    / "co2_analysis"
)

RECOMMENDATION_DIR = (
    OUTPUT_DIR
    / "recommendations"
)

RECOMMENDATION_FILE = (
    RECOMMENDATION_DIR
    / "recommendation_register.csv"
)

MACHINE_ACTION_FILE = (
    RECOMMENDATION_DIR
    / "machine_action_plan.csv"
)

SUMMARY_FILE = (
    RECOMMENDATION_DIR
    / "recommendation_summary.csv"
)

VALIDATION_FILE = (
    RECOMMENDATION_DIR
    / "recommendation_validation.csv"
)

ASSUMPTIONS_FILE = (
    RECOMMENDATION_DIR
    / "recommendation_assumptions.csv"
)

EXPECTED_RECORDS = 10_000


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
    """Create recommendation-engine output directories."""

    RECOMMENDATION_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    logger.info(
        "Recommendation-engine directory initialized."
    )


# ============================================================================
# DATA LOADING
# ============================================================================

def load_csv(path, description):
    """Load a CSV file with a clear error message."""

    if not path.exists():

        raise FileNotFoundError(
            f"{description} not found.\n"
            f"Expected file: {path}"
        )

    logger.info(
        "Loading %s...",
        description
    )

    data = pd.read_csv(
        path
    )

    logger.info(
        "%s loaded: %s rows × %s columns",
        description,
        data.shape[0],
        data.shape[1]
    )

    return data


def load_baseline_data():
    """Load baseline-enriched analytical observations."""

    return load_csv(
        BASELINE_FILE,
        "Baseline-enriched dataset"
    )


def load_inefficiency_summary():
    """Load machine-level inefficiency analysis."""

    candidates = [
        INEFFICIENCY_DIR
        / "machine_inefficiency_summary.csv",

        INEFFICIENCY_DIR
        / "machine_inefficiency_analysis.csv"
    ]

    for path in candidates:

        if path.exists():

            return load_csv(
                path,
                "Machine inefficiency summary"
            )

    logger.warning(
        "Machine inefficiency summary not found."
    )

    return None


def load_optimization_data():
    """Load machine-level optimization results."""

    path = (
        OPTIMIZATION_DIR
        / "machine_optimization.csv"
    )

    return load_csv(
        path,
        "Machine optimization results"
    )


def load_optimization_opportunities():
    """Load optimization opportunity report."""

    path = (
        OPTIMIZATION_DIR
        / "optimization_opportunities.csv"
    )

    return load_csv(
        path,
        "Optimization opportunity report"
    )


def load_co2_data():
    """Load machine-level sustainability results."""

    candidates = [
        CO2_DIR
        / "machine_sustainability_ranking.csv",

        CO2_DIR
        / "machine_co2_analysis.csv"
    ]

    for path in candidates:

        if path.exists():

            return load_csv(
                path,
                "Machine sustainability data"
            )

    logger.warning(
        "Machine sustainability data not found."
    )

    return None


def load_what_if_summary():
    """Load What-If scenario results."""

    path = (
        WHAT_IF_DIR
        / "scenario_summary.csv"
    )

    if not path.exists():

        logger.warning(
            "What-If scenario summary not found."
        )

        return None

    return load_csv(
        path,
        "What-If scenario summary"
    )


# ============================================================================
# COLUMN RESOLUTION
# ============================================================================

def resolve_column(
    dataframe,
    possible_names,
    required=False
):
    """
    Find the first available column from a list of possible names.

    This makes the recommendation engine robust to minor naming
    differences between analytical modules.
    """

    normalized = {
        str(column).strip().lower(): column
        for column in dataframe.columns
    }

    for name in possible_names:

        key = str(
            name
        ).strip().lower()

        if key in normalized:

            return normalized[
                key
            ]

    if required:

        raise KeyError(
            "Required column not found. "
            f"Expected one of: {possible_names}"
        )

    return None


# ============================================================================
# MACHINE METRIC PREPARATION
# ============================================================================

def prepare_machine_metrics(
    baseline,
    optimization,
    opportunities,
    co2_data=None,
    inefficiency=None
):
    """
    Create a unified machine-level analytical table.

    The table becomes the central input for recommendation generation.
    """

    logger.info(
        "Preparing unified machine analytical metrics..."
    )

    required_baseline_columns = [
        "Machine_ID",
        "Machine_Name",
        "Energy_Consumption_kWh",
        "Baseline_Energy_kWh",
        "Idle_Energy_kWh",
        "Actual_Production_Units",
        "Actual_State",
        "Planned_State"
    ]

    missing = [
        column
        for column in required_baseline_columns
        if column not in baseline.columns
    ]

    if missing:

        raise ValueError(
            "Baseline dataset is missing columns: "
            + ", ".join(missing)
        )

    # ------------------------------------------------------------------------
    # Machine-level aggregation
    # ------------------------------------------------------------------------

    machine = (
        baseline
        .groupby(
            [
                "Machine_ID",
                "Machine_Name"
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
            Idle_Energy_kWh=(
                "Idle_Energy_kWh",
                "sum"
            ),
            Production_Units=(
                "Actual_Production_Units",
                "sum"
            )
        )
    )

    # ------------------------------------------------------------------------
    # Idle records
    # ------------------------------------------------------------------------

    idle_counts = (
        baseline[
            baseline[
                "Actual_State"
            ]
            ==
            "IDLE"
        ]
        .groupby(
            "Machine_ID"
        )
        .size()
        .rename(
            "Idle_Records"
        )
    )

    machine = machine.merge(
        idle_counts,
        on="Machine_ID",
        how="left"
    )

    machine["Idle_Records"] = (
        machine[
            "Idle_Records"
        ]
        .fillna(0)
    )

    machine["Idle_Percentage"] = (
        machine[
            "Idle_Records"
        ]
        /
        machine[
            "Records"
        ]
        *
        100
    )

    # ------------------------------------------------------------------------
    # Schedule deviations
    # ------------------------------------------------------------------------

    deviation_counts = (
        baseline[
            baseline[
                "Planned_State"
            ]
            !=
            baseline[
                "Actual_State"
            ]
        ]
        .groupby(
            "Machine_ID"
        )
        .size()
        .rename(
            "Schedule_Deviation_Records"
        )
    )

    machine = machine.merge(
        deviation_counts,
        on="Machine_ID",
        how="left"
    )

    machine[
        "Schedule_Deviation_Records"
    ] = (
        machine[
            "Schedule_Deviation_Records"
        ]
        .fillna(0)
    )

    machine[
        "Schedule_Deviation_Percentage"
    ] = (
        machine[
            "Schedule_Deviation_Records"
        ]
        /
        machine[
            "Records"
        ]
        *
        100
    )

    # ------------------------------------------------------------------------
    # Energy intensity
    # ------------------------------------------------------------------------

    machine[
        "Energy_Per_Production_Unit"
    ] = np.where(
        machine[
            "Production_Units"
        ]
        > 0,

        machine[
            "Energy_kWh"
        ]
        /
        machine[
            "Production_Units"
        ],

        np.nan
    )

    # ------------------------------------------------------------------------
    # Baseline deviation
    # ------------------------------------------------------------------------

    machine[
        "Positive_Baseline_Deviation_kWh"
    ] = (
        machine[
            "Energy_kWh"
        ]
        -
        machine[
            "Baseline_Energy_kWh"
        ]
    ).clip(
        lower=0
    )

    # ------------------------------------------------------------------------
    # Optimization merge
    # ------------------------------------------------------------------------

    optimization_columns = [
        "Machine_ID",
        "Optimized_Total_Reduction_kWh",
        "Optimized_Energy_kWh",
        "Potential_Reduction_Percent",
        "Production_Output_Maintained"
    ]

    available_optimization_columns = [
        column
        for column in optimization_columns
        if column in optimization.columns
    ]

    machine = machine.merge(
        optimization[
            available_optimization_columns
        ],
        on="Machine_ID",
        how="left"
    )

    # ------------------------------------------------------------------------
    # Opportunity merge
    # ------------------------------------------------------------------------

    opportunity_columns = [
        "Machine_ID",
        "Idle_Opportunity_kWh",
        "Schedule_Opportunity_kWh",
        "Intensity_Opportunity_kWh"
    ]

    available_opportunity_columns = [
        column
        for column in opportunity_columns
        if column in opportunities.columns
    ]

    machine = machine.merge(
        opportunities[
            available_opportunity_columns
        ],
        on="Machine_ID",
        how="left"
    )

    # ------------------------------------------------------------------------
    # CO2 merge
    # ------------------------------------------------------------------------

    if co2_data is not None:

        co2_machine_id = resolve_column(
            co2_data,
            [
                "Machine_ID"
            ]
        )

        co2_columns = [
            "CO2e_kg",
            "CO2e_Per_Production_Unit",
            "Sustainability_Attention_Score",
            "Sustainability_Priority"
        ]

        available_co2_columns = [
            column
            for column in co2_columns
            if column in co2_data.columns
        ]

        if (
            co2_machine_id is not None
            and available_co2_columns
        ):

            merge_columns = [
                co2_machine_id
            ] + available_co2_columns

            co2_subset = (
                co2_data[
                    merge_columns
                ]
                .drop_duplicates(
                    subset=[
                        co2_machine_id
                    ]
                )
                .rename(
                    columns={
                        co2_machine_id:
                            "Machine_ID"
                    }
                )
            )

            machine = machine.merge(
                co2_subset,
                on="Machine_ID",
                how="left"
            )

    # ------------------------------------------------------------------------
    # Fill optional metrics
    # ------------------------------------------------------------------------

    numeric_defaults = [
        "Optimized_Total_Reduction_kWh",
        "Optimized_Energy_kWh",
        "Potential_Reduction_Percent",
        "Idle_Opportunity_kWh",
        "Schedule_Opportunity_kWh",
        "Intensity_Opportunity_kWh",
        "CO2e_kg",
        "CO2e_Per_Production_Unit",
        "Sustainability_Attention_Score"
    ]

    for column in numeric_defaults:

        if column in machine.columns:

            machine[column] = pd.to_numeric(
                machine[column],
                errors="coerce"
            ).fillna(0)

    logger.info(
        "Unified machine metrics prepared for %s machines.",
        len(machine)
    )

    return machine


# ============================================================================
# RECOMMENDATION HELPERS
# ============================================================================

def create_recommendation(
    machine_id,
    machine_name,
    issue,
    evidence,
    root_cause_indicator,
    action,
    energy_impact,
    co2_impact,
    priority,
    confidence,
    constraint,
    source
):
    """Create one standardized recommendation record."""

    return {

        "Machine_ID":
            machine_id,

        "Machine_Name":
            machine_name,

        "Issue":
            issue,

        "Evidence":
            evidence,

        "Root_Cause_Indicator":
            root_cause_indicator,

        "Recommended_Action":
            action,

        "Potential_Energy_Impact_kWh":
            round(
                max(
                    float(
                        energy_impact
                    ),
                    0
                ),
                2
            ),

        "Potential_CO2e_Impact_kg":
            round(
                max(
                    float(
                        co2_impact
                    ),
                    0
                ),
                2
            ),

        "Priority":
            priority,

        "Confidence":
            confidence,

        "Operational_Constraint":
            constraint,

        "Evidence_Source":
            source,

        "Recommendation_Status":
            "ANALYTICAL RECOMMENDATION",

        "Implementation_Status":
            "To be validated operationally"
    }


# ============================================================================
# PRIORITY CLASSIFICATION
# ============================================================================

def classify_priority(
    score
):
    """Convert numerical recommendation score into a priority."""

    if score >= 70:

        return "HIGH"

    if score >= 40:

        return "MEDIUM"

    return "LOW"


# ============================================================================
# RECOMMENDATION GENERATION
# ============================================================================

def generate_recommendations(
    machine
):
    """
    Generate machine-specific recommendations using actual analytical
    evidence.

    Multiple recommendations may be generated for the same machine
    when multiple independent opportunities exist.
    """

    logger.info(
        "Generating evidence-based recommendations..."
    )

    recommendations = []

    for _, row in machine.iterrows():

        machine_id = row[
            "Machine_ID"
        ]

        machine_name = row[
            "Machine_Name"
        ]

        energy = float(
            row[
                "Energy_kWh"
            ]
        )

        idle_energy = float(
            row[
                "Idle_Energy_kWh"
            ]
        )

        idle_pct = float(
            row[
                "Idle_Percentage"
            ]
        )

        schedule_pct = float(
            row[
                "Schedule_Deviation_Percentage"
            ]
        )

        energy_intensity = float(
            row[
                "Energy_Per_Production_Unit"
            ]
        )

        optimized_reduction = float(
            row.get(
                "Optimized_Total_Reduction_kWh",
                0
            )
        )

        optimized_pct = float(
            row.get(
                "Potential_Reduction_Percent",
                0
            )
        )

        idle_opportunity = float(
            row.get(
                "Idle_Opportunity_kWh",
                0
            )
        )

        schedule_opportunity = float(
            row.get(
                "Schedule_Opportunity_kWh",
                0
            )
        )

        intensity_opportunity = float(
            row.get(
                "Intensity_Opportunity_kWh",
                0
            )
        )

        co2e_per_unit = float(
            row.get(
                "CO2e_Per_Production_Unit",
                0
            )
        )

        # --------------------------------------------------------------------
        # 1. Idle-energy opportunity
        # --------------------------------------------------------------------

        if (
            idle_energy > 0
            and idle_pct >= 10
            and idle_opportunity > 0
        ):

            score = min(
                100,
                (
                    idle_pct
                    * 2
                    +
                    (
                        idle_opportunity
                        /
                        max(
                            energy,
                            1
                        )
                    )
                    *
                    100
                )
            )

            priority = classify_priority(
                score
            )

            recommendations.append(
                create_recommendation(

                    machine_id,

                    machine_name,

                    "Idle energy consumption",

                    (
                        f"{idle_pct:.2f}% of observations were idle "
                        f"and idle energy totaled "
                        f"{idle_energy:.2f} kWh."
                    ),

                    "Non-zero energy consumption during idle operation",

                    (
                        "Review idle-state operating procedures and "
                        "investigate whether non-essential auxiliary "
                        "loads can be reduced during prolonged idle periods."
                    ),

                    idle_opportunity,

                    idle_opportunity * 0.70,

                    priority,

                    "HIGH",

                    (
                        "Production requirements and machine restart "
                        "constraints must be validated."
                    ),

                    "Baseline + Inefficiency Analysis"
                )
            )

        # --------------------------------------------------------------------
        # 2. Schedule adherence
        # --------------------------------------------------------------------

        if (
            schedule_pct >= 10
            and schedule_opportunity > 0
        ):

            score = min(
                100,
                schedule_pct
                * 2
            )

            priority = classify_priority(
                score
            )

            recommendations.append(
                create_recommendation(

                    machine_id,

                    machine_name,

                    "Production schedule deviation",

                    (
                        f"{schedule_pct:.2f}% of observations "
                        f"showed a difference between planned and "
                        f"actual machine state."
                    ),

                    "Mismatch between planned and actual operating state",

                    (
                        "Review startup, shutdown and break schedules. "
                        "Investigate avoidable operation outside planned "
                        "production windows."
                    ),

                    schedule_opportunity,

                    schedule_opportunity * 0.70,

                    priority,

                    "MEDIUM",

                    (
                        "Schedule changes must not interfere with "
                        "production demand or maintenance requirements."
                    ),

                    "Baseline + Schedule Analysis"
                )
            )

        # --------------------------------------------------------------------
        # 3. Energy intensity
        # --------------------------------------------------------------------

        if (
            energy_intensity > 0
            and intensity_opportunity > 0
        ):

            score = min(
                100,
                optimized_pct
                +
                (
                    co2e_per_unit
                    /
                    max(
                        energy_intensity,
                        0.01
                    )
                    *
                    10
                )
            )

            priority = classify_priority(
                score
            )

            recommendations.append(
                create_recommendation(

                    machine_id,

                    machine_name,

                    "Energy intensity improvement opportunity",

                    (
                        f"Observed energy intensity was "
                        f"{energy_intensity:.3f} kWh/unit."
                    ),

                    "Energy consumption remains significant relative to production output",

                    (
                        "Investigate operating parameters, machine "
                        "utilization and process efficiency before "
                        "considering equipment-level interventions."
                    ),

                    intensity_opportunity,

                    intensity_opportunity * 0.70,

                    priority,

                    "MEDIUM",

                    (
                        "Production quality and throughput must remain "
                        "within operational requirements."
                    ),

                    "Baseline + Optimization Analysis"
                )
            )

        # --------------------------------------------------------------------
        # 4. High overall optimization opportunity
        # --------------------------------------------------------------------

        if (
            optimized_reduction > 0
            and optimized_pct >= 20
        ):

            recommendations.append(
                create_recommendation(

                    machine_id,

                    machine_name,

                    "High optimization opportunity",

                    (
                        f"The optimization model identifies "
                        f"{optimized_reduction:.2f} kWh of potential "
                        f"energy reduction, equivalent to "
                        f"{optimized_pct:.2f}% of machine energy."
                    ),

                    "Combined idle, schedule and energy-intensity opportunities",

                    (
                        "Prioritize this machine for a detailed "
                        "operational review combining idle reduction, "
                        "schedule adherence and energy-intensity measures."
                    ),

                    optimized_reduction,

                    optimized_reduction * 0.70,

                    "HIGH",

                    "HIGH",

                    (
                        "Modelled opportunity must be validated against "
                        "actual machine operating conditions."
                    ),

                    "Optimization Analysis"
                )
            )

    logger.info(
        "Recommendations generated: %s",
        len(recommendations)
    )

    return pd.DataFrame(
        recommendations
    )


# ============================================================================
# MACHINE ACTION PLAN
# ============================================================================

def create_machine_action_plan(
    machine,
    recommendations
):
    """
    Create one consolidated action plan per machine.

    This table is designed for Power BI and management review.
    """

    logger.info(
        "Creating machine action plan..."
    )

    priority_map = {
        "HIGH": 3,
        "MEDIUM": 2,
        "LOW": 1
    }

    if recommendations.empty:

        action_plan = machine.copy()

        action_plan[
            "Recommendation_Count"
        ] = 0

        action_plan[
            "Highest_Priority"
        ] = "NONE"

        return action_plan

    recommendation_summary = (
        recommendations
        .assign(
            Priority_Score=lambda x:
                x[
                    "Priority"
                ]
                .map(
                    priority_map
                )
        )
        .groupby(
            [
                "Machine_ID"
            ],
            as_index=False
        )
        .agg(
            Recommendation_Count=(
                "Issue",
                "count"
            ),
            Highest_Priority_Score=(
                "Priority_Score",
                "max"
            ),
            Total_Potential_Energy_Impact_kWh=(
                "Potential_Energy_Impact_kWh",
                "sum"
            ),
            Total_Potential_CO2e_Impact_kg=(
                "Potential_CO2e_Impact_kg",
                "sum"
            )
        )
    )

    recommendation_summary[
        "Highest_Priority"
    ] = (
        recommendation_summary[
            "Highest_Priority_Score"
        ]
        .map(
            {
                3: "HIGH",
                2: "MEDIUM",
                1: "LOW"
            }
        )
        .fillna(
            "LOW"
        )
    )

    action_plan = machine.merge(
        recommendation_summary[
            [
                "Machine_ID",
                "Recommendation_Count",
                "Highest_Priority",
                "Total_Potential_Energy_Impact_kWh",
                "Total_Potential_CO2e_Impact_kg"
            ]
        ],
        on="Machine_ID",
        how="left"
    )

    action_plan[
        "Recommendation_Count"
    ] = (
        action_plan[
            "Recommendation_Count"
        ]
        .fillna(0)
        .astype(int)
    )

    action_plan[
        "Highest_Priority"
    ] = (
        action_plan[
            "Highest_Priority"
        ]
        .fillna(
            "LOW"
        )
    )

    return action_plan


# ============================================================================
# SUMMARY
# ============================================================================

def create_summary(
    baseline,
    machine,
    recommendations
):
    """Create executive recommendation summary."""

    total_energy = (
        baseline[
            "Energy_Consumption_kWh"
        ]
        .sum()
    )

    total_production = (
        baseline[
            "Actual_Production_Units"
        ]
        .sum()
    )

    total_idle_energy = (
        baseline[
            "Idle_Energy_kWh"
        ]
        .sum()
    )

    total_potential_energy = (
        recommendations[
            "Potential_Energy_Impact_kWh"
        ]
        .sum()
        if not recommendations.empty
        else 0
    )

    total_potential_co2 = (
        recommendations[
            "Potential_CO2e_Impact_kg"
        ]
        .sum()
        if not recommendations.empty
        else 0
    )

    high_count = (
        (
            recommendations[
                "Priority"
            ]
            ==
            "HIGH"
        )
        .sum()
        if not recommendations.empty
        else 0
    )

    medium_count = (
        (
            recommendations[
                "Priority"
            ]
            ==
            "MEDIUM"
        )
        .sum()
        if not recommendations.empty
        else 0
    )

    low_count = (
        (
            recommendations[
                "Priority"
            ]
            ==
            "LOW"
        )
        .sum()
        if not recommendations.empty
        else 0
    )

    summary = pd.DataFrame({

        "Metric": [

            "Records Analyzed",

            "Machines Analyzed",

            "Total Energy",

            "Total Production",

            "Total Idle Energy",

            "Recommendation Count",

            "High Priority Recommendations",

            "Medium Priority Recommendations",

            "Low Priority Recommendations",

            "Potential Energy Impact",

            "Potential CO2e Impact"
        ],

        "Value": [

            len(
                baseline
            ),

            machine[
                "Machine_ID"
            ].nunique(),

            total_energy,

            total_production,

            total_idle_energy,

            len(
                recommendations
            ),

            high_count,

            medium_count,

            low_count,

            total_potential_energy,

            total_potential_co2
        ],

        "Unit": [

            "Records",

            "Machines",

            "kWh",

            "Units",

            "kWh",

            "Recommendations",

            "Recommendations",

            "Recommendations",

            "Recommendations",

            "kWh",

            "kg CO2e"
        ],

        "Data_Status": [

            "Observed",

            "Observed",

            "Observed",

            "Observed",

            "Observed",

            "Analytical",

            "Analytical",

            "Analytical",

            "Analytical",

            "Modelled Potential",

            "Modelled Potential"
        ]
    })

    summary.to_csv(
        SUMMARY_FILE,
        index=False
    )

    return summary


# ============================================================================
# ASSUMPTIONS
# ============================================================================

def create_assumptions():
    """Document recommendation-engine assumptions."""

    assumptions = pd.DataFrame({

        "Assumption": [

            "Recommendation source",

            "Production constraint",

            "Energy impact interpretation",

            "CO2e impact interpretation",

            "Operational validation",

            "Dataset type"
        ],

        "Value": [

            (
                "Generated from baseline, inefficiency, "
                "What-If, CO2 and optimization outputs."
            ),

            "Production output must be maintained.",

            "Potential / modelled impact.",

            "Potential / modelled impact.",

            (
                "All recommendations require validation "
                "against actual operating conditions."
            ),

            "Synthetic analytical dataset."
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

def validate_recommendations(
    baseline,
    machine,
    recommendations,
    action_plan
):
    """Validate recommendation-engine outputs."""

    logger.info(
        "Running recommendation-engine validation..."
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
            len(baseline),

        "Status":
            (
                "PASS"
                if len(baseline)
                ==
                EXPECTED_RECORDS
                else "FAIL"
            )
    })

    # ------------------------------------------------------------------------
    # Machine count
    # ------------------------------------------------------------------------

    machine_count = (
        machine[
            "Machine_ID"
        ].nunique()
    )

    validations.append({

        "Validation":
            "Machine Count",

        "Expected":
            5,

        "Actual":
            machine_count,

        "Status":
            (
                "PASS"
                if machine_count == 5
                else "FAIL"
            )
    })

    # ------------------------------------------------------------------------
    # Recommendation machine IDs
    # ------------------------------------------------------------------------

    invalid_machine_recommendations = 0

    if not recommendations.empty:

        invalid_machine_recommendations = (
            ~recommendations[
                "Machine_ID"
            ].isin(
                machine[
                    "Machine_ID"
                ]
            )
        ).sum()

    validations.append({

        "Validation":
            "Invalid Recommendation Machine IDs",

        "Expected":
            0,

        "Actual":
            int(
                invalid_machine_recommendations
            ),

        "Status":
            (
                "PASS"
                if invalid_machine_recommendations == 0
                else "FAIL"
            )
    })

    # ------------------------------------------------------------------------
    # Negative energy impact
    # ------------------------------------------------------------------------

    negative_energy = 0

    if not recommendations.empty:

        negative_energy = (
            recommendations[
                "Potential_Energy_Impact_kWh"
            ]
            < 0
        ).sum()

    validations.append({

        "Validation":
            "Negative Energy Impact",

        "Expected":
            0,

        "Actual":
            int(
                negative_energy
            ),

        "Status":
            (
                "PASS"
                if negative_energy == 0
                else "FAIL"
            )
    })

    # ------------------------------------------------------------------------
    # Negative CO2 impact
    # ------------------------------------------------------------------------

    negative_co2 = 0

    if not recommendations.empty:

        negative_co2 = (
            recommendations[
                "Potential_CO2e_Impact_kg"
            ]
            < 0
        ).sum()

    validations.append({

        "Validation":
            "Negative CO2e Impact",

        "Expected":
            0,

        "Actual":
            int(
                negative_co2
            ),

        "Status":
            (
                "PASS"
                if negative_co2 == 0
                else "FAIL"
            )
    })

    # ------------------------------------------------------------------------
    # Valid priorities
    # ------------------------------------------------------------------------

    invalid_priorities = 0

    if not recommendations.empty:

        valid_priorities = {
            "HIGH",
            "MEDIUM",
            "LOW"
        }

        invalid_priorities = (
            ~recommendations[
                "Priority"
            ].isin(
                valid_priorities
            )
        ).sum()

    validations.append({

        "Validation":
            "Invalid Recommendation Priorities",

        "Expected":
            0,

        "Actual":
            int(
                invalid_priorities
            ),

        "Status":
            (
                "PASS"
                if invalid_priorities == 0
                else "FAIL"
            )
    })

    # ------------------------------------------------------------------------
    # Action plan machine count
    # ------------------------------------------------------------------------

    validations.append({

        "Validation":
            "Action Plan Machine Count",

        "Expected":
            machine_count,

        "Actual":
            action_plan[
                "Machine_ID"
            ].nunique(),

        "Status":
            (
                "PASS"
                if (
                    action_plan[
                        "Machine_ID"
                    ].nunique()
                    ==
                    machine_count
                )
                else "FAIL"
            )
    })

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
    """Execute the complete recommendation-engine pipeline."""

    logger.info("=" * 78)

    logger.info(
        "SUSTAINABLE MANUFACTURING - RECOMMENDATION ENGINE"
    )

    logger.info("=" * 78)

    try:

        # --------------------------------------------------------------------
        # Initialization
        # --------------------------------------------------------------------

        initialize_directories()

        # --------------------------------------------------------------------
        # Load analytical outputs
        # --------------------------------------------------------------------

        baseline = (
            load_baseline_data()
        )

        optimization = (
            load_optimization_data()
        )

        opportunities = (
            load_optimization_opportunities()
        )

        co2_data = (
            load_co2_data()
        )

        inefficiency = (
            load_inefficiency_summary()
        )

        what_if = (
            load_what_if_summary()
        )

        # --------------------------------------------------------------------
        # Prepare machine metrics
        # --------------------------------------------------------------------

        machine = (
            prepare_machine_metrics(
                baseline,
                optimization,
                opportunities,
                co2_data,
                inefficiency
            )
        )

        # --------------------------------------------------------------------
        # Generate recommendations
        # --------------------------------------------------------------------

        recommendations = (
            generate_recommendations(
                machine
            )
        )

        # --------------------------------------------------------------------
        # Save recommendation register
        # --------------------------------------------------------------------

        recommendations.to_csv(
            RECOMMENDATION_FILE,
            index=False
        )

        logger.info(
            "Recommendation register saved."
        )

        # --------------------------------------------------------------------
        # Machine action plan
        # --------------------------------------------------------------------

        action_plan = (
            create_machine_action_plan(
                machine,
                recommendations
            )
        )

        action_plan.to_csv(
            MACHINE_ACTION_FILE,
            index=False
        )

        logger.info(
            "Machine action plan generated."
        )

        # --------------------------------------------------------------------
        # Executive summary
        # --------------------------------------------------------------------

        summary = (
            create_summary(
                baseline,
                machine,
                recommendations
            )
        )

        # --------------------------------------------------------------------
        # Assumptions
        # --------------------------------------------------------------------

        create_assumptions()

        # --------------------------------------------------------------------
        # Validation
        # --------------------------------------------------------------------

        validation = (
            validate_recommendations(
                baseline,
                machine,
                recommendations,
                action_plan
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
        # Key metrics
        # --------------------------------------------------------------------

        total_energy = (
            baseline[
                "Energy_Consumption_kWh"
            ]
            .sum()
        )

        total_idle_energy = (
            baseline[
                "Idle_Energy_kWh"
            ]
            .sum()
        )

        total_potential_energy = (
            recommendations[
                "Potential_Energy_Impact_kWh"
            ]
            .sum()
            if not recommendations.empty
            else 0
        )

        total_potential_co2 = (
            recommendations[
                "Potential_CO2e_Impact_kg"
            ]
            .sum()
            if not recommendations.empty
            else 0
        )

        high_priority = (
            (
                recommendations[
                    "Priority"
                ]
                ==
                "HIGH"
            )
            .sum()
            if not recommendations.empty
            else 0
        )

        # --------------------------------------------------------------------
        # Completion
        # --------------------------------------------------------------------

        logger.info("=" * 78)

        if validation_failures == 0:

            logger.info(
                "RECOMMENDATION ENGINE COMPLETED SUCCESSFULLY"
            )

        else:

            logger.warning(
                "RECOMMENDATION ENGINE COMPLETED WITH %s VALIDATION FAILURES",
                validation_failures
            )

        logger.info("=" * 78)

        logger.info(
            "Records analyzed          : %s",
            len(baseline)
        )

        logger.info(
            "Machines analyzed         : %s",
            machine[
                "Machine_ID"
            ].nunique()
        )

        logger.info(
            "Total energy              : %.2f kWh",
            total_energy
        )

        logger.info(
            "Total idle energy         : %.2f kWh",
            total_idle_energy
        )

        logger.info(
            "Recommendations generated : %s",
            len(recommendations)
        )

        logger.info(
            "High-priority actions     : %s",
            high_priority
        )

        logger.info(
            "Potential energy impact  : %.2f kWh",
            total_potential_energy
        )

        logger.info(
            "Potential CO2e impact    : %.2f kg",
            total_potential_co2
        )

        logger.info(
            "Output directory          : %s",
            RECOMMENDATION_DIR
        )

        # --------------------------------------------------------------------
        # Console output
        # --------------------------------------------------------------------

        print()

        print("=" * 78)

        print(
            "RECOMMENDATION ENGINE SUMMARY"
        )

        print("=" * 78)

        print(
            summary.to_string(
                index=False
            )
        )

        print()

        print(
            "MACHINE ACTION PLAN"
        )

        print("=" * 78)

        action_columns = [
            "Machine_ID",
            "Machine_Name",
            "Energy_kWh",
            "Idle_Energy_kWh",
            "Idle_Percentage",
            "Schedule_Deviation_Percentage",
            "Optimized_Total_Reduction_kWh",
            "Potential_Reduction_Percent",
            "Recommendation_Count",
            "Highest_Priority"
        ]

        available_action_columns = [
            column
            for column in action_columns
            if column in action_plan.columns
        ]

        print(
            action_plan[
                available_action_columns
            ]
            .sort_values(
                "Highest_Priority"
            )
            .to_string(
                index=False
            )
        )

        print()

        print(
            "RECOMMENDATION REGISTER"
        )

        print("=" * 78)

        if recommendations.empty:

            print(
                "No recommendation candidates were generated."
            )

        else:

            display_columns = [
                "Machine_ID",
                "Machine_Name",
                "Issue",
                "Recommended_Action",
                "Potential_Energy_Impact_kWh",
                "Potential_CO2e_Impact_kg",
                "Priority",
                "Confidence"
            ]

            print(
                recommendations[
                    display_columns
                ]
                .sort_values(
                    [
                        "Priority",
                        "Potential_Energy_Impact_kWh"
                    ],
                    ascending=[
                        True,
                        False
                    ]
                )
                .to_string(
                    index=False
                )
            )

        print()

        print(
            "RECOMMENDATION VALIDATION"
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
            "Recommendations are analytical decision-support outputs."
        )

        print(
            "Potential impacts are MODELLED and must be validated operationally."
        )

        print(
            "The dataset is synthetic and does not represent a real factory."
        )

        print()

        print(
            f"Recommendation outputs: {RECOMMENDATION_DIR}"
        )

        print("=" * 78)

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

    except KeyError as error:

        logger.error(
            "Required analytical column missing: %s",
            error
        )

        sys.exit(1)

    except Exception as error:

        logger.exception(
            "Unexpected error during recommendation analysis: %s",
            error
        )

        sys.exit(1)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()