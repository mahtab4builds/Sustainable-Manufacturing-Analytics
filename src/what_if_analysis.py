"""
Sustainable Manufacturing Analytics
What-If Scenario Analysis

This module evaluates modelled energy and CO2e impacts under
alternative operational improvement scenarios.

All results represent potential/modelled impacts and must not
be interpreted as actual achieved manufacturing savings.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from library.config import (
    PROJECT_NAME,
    BASELINE_DIR,
    WHAT_IF_DIR,
    EMISSION_FACTOR,
    IDLE_ENERGY_REDUCTION,
    SCHEDULE_ENERGY_REDUCTION,
    ENERGY_INTENSITY_IMPROVEMENT,
    initialize_directories,
)

from library.constants import MACHINE_IDS
from library.io import (
    load_csv,
    save_csv,
    save_dataframe,
    ensure_directory,
)
from library.logger import get_logger
from library.plotting import (
    create_standard_figure,
    apply_standard_axis_labels,
    save_chart,
)
from library.validation import (
    validate_required_columns,
    validate_record_count,
    validate_no_negative_values,
)


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

ENERGY_COLUMN = "Energy_Consumption_kWh"
IDLE_ENERGY_COLUMN = "Idle_Energy_kWh"
PRODUCTION_COLUMN = "Actual_Production_Units"
MACHINE_COLUMN = "Machine_ID"
PLANNED_STATE_COLUMN = "Planned_State"
ACTUAL_STATE_COLUMN = "Actual_State"
TIMESTAMP_COLUMN = "Timestamp"

REQUIRED_COLUMNS = [
    ENERGY_COLUMN,
    IDLE_ENERGY_COLUMN,
    PRODUCTION_COLUMN,
    MACHINE_COLUMN,
    PLANNED_STATE_COLUMN,
    ACTUAL_STATE_COLUMN,
    TIMESTAMP_COLUMN,
]


# ---------------------------------------------------------------------------
# DIRECTORY MANAGEMENT
# ---------------------------------------------------------------------------

def initialize_what_if_directories():
    """
    Initialize directories required by the What-If analysis.
    """

    initialize_directories()

    ensure_directory(WHAT_IF_DIR)

    ensure_directory(
        WHAT_IF_DIR / "charts"
    )

    ensure_directory(
        WHAT_IF_DIR / "scenarios"
    )

    logger.info(
        "What-If analysis directories initialized."
    )


# ---------------------------------------------------------------------------
# DATA PREPARATION
# ---------------------------------------------------------------------------

def load_baseline_dataset():
    """
    Load the baseline-enriched analytical dataset.
    """

    source = (
        BASELINE_DIR
        / "baseline_enriched_dataset.csv"
    )

    logger.info(
        "Loading baseline-enriched dataset..."
    )

    logger.info(
        "Source: %s",
        source
    )

    dataframe = load_csv(source)

    validate_record_count(
        dataframe,
        10000
    )

    validate_required_columns(
        dataframe,
        REQUIRED_COLUMNS
    )

    logger.info(
        "Dataset loaded: %d rows × %d columns",
        dataframe.shape[0],
        dataframe.shape[1]
    )

    return dataframe


def prepare_features(dataframe):
    """
    Prepare analytical features required by the scenario engine.
    """

    df = dataframe.copy()

    df[TIMESTAMP_COLUMN] = pd.to_datetime(
        df[TIMESTAMP_COLUMN],
        errors="coerce"
    )

    numeric_columns = [
        ENERGY_COLUMN,
        IDLE_ENERGY_COLUMN,
        PRODUCTION_COLUMN,
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        ).fillna(0.0)

    df["Schedule_Deviation_Flag"] = (
        df[PLANNED_STATE_COLUMN]
        != df[ACTUAL_STATE_COLUMN]
    )

    df["Productive_Energy_kWh"] = (
        df[ENERGY_COLUMN]
        - df[IDLE_ENERGY_COLUMN]
    )

    df["Productive_Energy_kWh"] = (
        df["Productive_Energy_kWh"]
        .clip(lower=0)
    )

    df["Energy_per_Unit_kWh"] = np.where(
        df[PRODUCTION_COLUMN] > 0,
        df[ENERGY_COLUMN]
        / df[PRODUCTION_COLUMN],
        0.0
    )

    logger.info(
        "What-If analytical features prepared."
    )

    return df


# ---------------------------------------------------------------------------
# SCENARIO GENERATION
# ---------------------------------------------------------------------------

def create_current_operations_scenario(df):
    """
    Current operations scenario.

    This scenario represents the observed operating condition
    and therefore applies no improvement.
    """

    scenario = df.copy()

    scenario["Scenario"] = (
        "CURRENT_OPERATIONS"
    )

    scenario["Scenario_Energy_kWh"] = (
        scenario[ENERGY_COLUMN]
    )

    return scenario


def create_idle_reduction_scenario(df):
    """
    Reduce idle energy consumption by the configured percentage.
    """

    scenario = df.copy()

    scenario["Scenario"] = (
        "REDUCE_IDLE_ENERGY"
    )

    idle_reduction = (
        scenario[IDLE_ENERGY_COLUMN]
        * IDLE_ENERGY_REDUCTION
    )

    scenario["Scenario_Energy_kWh"] = (
        scenario[ENERGY_COLUMN]
        - idle_reduction
    )

    return scenario


def create_schedule_improvement_scenario(df):
    """
    Reduce energy associated with schedule deviations.
    """

    scenario = df.copy()

    scenario["Scenario"] = (
        "IMPROVE_SCHEDULE_ADHERENCE"
    )

    schedule_reduction = np.where(
        scenario["Schedule_Deviation_Flag"],
        scenario[ENERGY_COLUMN]
        * SCHEDULE_ENERGY_REDUCTION,
        0.0
    )

    scenario["Scenario_Energy_kWh"] = (
        scenario[ENERGY_COLUMN]
        - schedule_reduction
    )

    return scenario


def create_energy_intensity_scenario(df):
    """
    Improve productive energy intensity while maintaining
    the observed production output.
    """

    scenario = df.copy()

    scenario["Scenario"] = (
        "IMPROVE_ENERGY_INTENSITY"
    )

    productive_reduction = (
        scenario["Productive_Energy_kWh"]
        * ENERGY_INTENSITY_IMPROVEMENT
    )

    scenario["Scenario_Energy_kWh"] = (
        scenario[ENERGY_COLUMN]
        - productive_reduction
    )

    return scenario


def create_combined_scenario(df):
    """
    Apply all configured improvement assumptions together.

    The combined scenario is calculated independently from
    the current observed energy to avoid double-counting
    reductions.
    """

    scenario = df.copy()

    scenario["Scenario"] = (
        "COMBINED_IMPROVEMENT"
    )

    idle_reduction = (
        scenario[IDLE_ENERGY_COLUMN]
        * IDLE_ENERGY_REDUCTION
    )

    schedule_reduction = np.where(
        scenario["Schedule_Deviation_Flag"],
        (
            scenario[ENERGY_COLUMN]
            - idle_reduction
        )
        * SCHEDULE_ENERGY_REDUCTION,
        0.0
    )

    productive_energy = (
        scenario[ENERGY_COLUMN]
        - idle_reduction
        - schedule_reduction
    )

    productive_energy = (
        productive_energy
        .clip(lower=0)
    )

    productive_reduction = (
        productive_energy
        * ENERGY_INTENSITY_IMPROVEMENT
    )

    scenario["Scenario_Energy_kWh"] = (
        scenario[ENERGY_COLUMN]
        - idle_reduction
        - schedule_reduction
        - productive_reduction
    )

    return scenario


def generate_scenarios(df):
    """
    Generate all project What-If scenarios.
    """

    logger.info(
        "Generating current-operations scenario..."
    )

    current = (
        create_current_operations_scenario(df)
    )

    logger.info(
        "Generating idle-energy reduction scenario..."
    )

    idle = (
        create_idle_reduction_scenario(df)
    )

    logger.info(
        "Generating schedule-improvement scenario..."
    )

    schedule = (
        create_schedule_improvement_scenario(df)
    )

    logger.info(
        "Generating energy-intensity improvement scenario..."
    )

    intensity = (
        create_energy_intensity_scenario(df)
    )

    logger.info(
        "Generating combined improvement scenario..."
    )

    combined = (
        create_combined_scenario(df)
    )

    scenarios = pd.concat(
        [
            current,
            idle,
            schedule,
            intensity,
            combined,
        ],
        ignore_index=True
    )

    scenarios["Scenario_Energy_kWh"] = (
        scenarios["Scenario_Energy_kWh"]
        .clip(lower=0)
    )

    scenarios["Scenario_CO2e_kg"] = (
        scenarios["Scenario_Energy_kWh"]
        * EMISSION_FACTOR
    )

    scenarios["Actual_CO2e_kg"] = (
        scenarios[ENERGY_COLUMN]
        * EMISSION_FACTOR
    )

    scenarios["Potential_Energy_Reduction_kWh"] = (
        scenarios[ENERGY_COLUMN]
        - scenarios["Scenario_Energy_kWh"]
    )

    scenarios[
        "Potential_Energy_Reduction_Percent"
    ] = np.where(
        scenarios[ENERGY_COLUMN] > 0,
        (
            scenarios[
                "Potential_Energy_Reduction_kWh"
            ]
            / scenarios[ENERGY_COLUMN]
        )
        * 100,
        0.0
    )

    scenarios["Potential_CO2_Reduction_kg"] = (
        scenarios["Actual_CO2e_kg"]
        - scenarios["Scenario_CO2e_kg"]
    )

    scenarios["Production_Maintained"] = True

    logger.info(
        "Scenario detail dataset created: %d rows.",
        len(scenarios)
    )

    return scenarios


# ---------------------------------------------------------------------------
# SCENARIO SUMMARY
# ---------------------------------------------------------------------------

def create_scenario_summary(scenarios):
    """
    Aggregate detailed scenario results into a management-level summary.
    """

    summary = (
        scenarios
        .groupby("Scenario", as_index=False)
        .agg(
            Actual_Energy_kWh=(
                ENERGY_COLUMN,
                "sum"
            ),
            Scenario_Energy_kWh=(
                "Scenario_Energy_kWh",
                "sum"
            ),
            Actual_CO2e_kg=(
                "Actual_CO2e_kg",
                "sum"
            ),
            Scenario_CO2e_kg=(
                "Scenario_CO2e_kg",
                "sum"
            ),
            Production_Output_Units=(
                PRODUCTION_COLUMN,
                "sum"
            ),
            Production_Maintained=(
                "Production_Maintained",
                "all"
            ),
        )
    )

    summary[
        "Potential_Energy_Reduction_kWh"
    ] = (
        summary["Actual_Energy_kWh"]
        - summary["Scenario_Energy_kWh"]
    )

    summary[
        "Potential_Energy_Reduction_Percent"
    ] = np.where(
        summary["Actual_Energy_kWh"] > 0,
        (
            summary[
                "Potential_Energy_Reduction_kWh"
            ]
            / summary["Actual_Energy_kWh"]
        )
        * 100,
        0.0
    )

    summary[
        "Potential_CO2_Reduction_kg"
    ] = (
        summary["Actual_CO2e_kg"]
        - summary["Scenario_CO2e_kg"]
    )

    return summary


# ---------------------------------------------------------------------------
# SCENARIO ASSUMPTIONS
# ---------------------------------------------------------------------------

def create_assumption_table():
    """
    Create a transparent record of all scenario assumptions.
    """

    return pd.DataFrame(
        [
            {
                "Assumption": (
                    "Idle energy reduction"
                ),
                "Value": (
                    IDLE_ENERGY_REDUCTION
                ),
                "Unit": "Percentage",
            },
            {
                "Assumption": (
                    "Schedule deviation energy reduction"
                ),
                "Value": (
                    SCHEDULE_ENERGY_REDUCTION
                ),
                "Unit": "Percentage",
            },
            {
                "Assumption": (
                    "Productive energy-intensity improvement"
                ),
                "Value": (
                    ENERGY_INTENSITY_IMPROVEMENT
                ),
                "Unit": "Percentage",
            },
            {
                "Assumption": (
                    "Production output"
                ),
                "Value": (
                    "Maintained at observed level"
                ),
                "Unit": "Units",
            },
            {
                "Assumption": (
                    "Synthetic emission factor"
                ),
                "Value": (
                    EMISSION_FACTOR
                ),
                "Unit": "kg CO2e/kWh",
            },
            {
                "Assumption": (
                    "Scenario interpretation"
                ),
                "Value": (
                    "Potential / modelled only"
                ),
                "Unit": "N/A",
            },
        ]
    )


# ---------------------------------------------------------------------------
# CHARTS
# ---------------------------------------------------------------------------

def create_energy_comparison_chart(summary):
    """
    Create scenario energy comparison chart.
    """

    figure, axis = create_standard_figure(
        width=11,
        height=6
    )

    axis.bar(
        summary["Scenario"],
        summary["Scenario_Energy_kWh"]
    )

    axis.tick_params(
        axis="x",
        rotation=25
    )

    apply_standard_axis_labels(
        axis,
        title="Energy Consumption by What-If Scenario",
        xlabel="Scenario",
        ylabel="Energy Consumption (kWh)"
    )

    save_chart(
        figure,
        WHAT_IF_DIR
        / "charts"
        / "01_scenario_energy_comparison.png"
    )


def create_reduction_chart(summary):
    """
    Create potential energy-reduction comparison chart.
    """

    figure, axis = create_standard_figure(
        width=11,
        height=6
    )

    axis.bar(
        summary["Scenario"],
        summary[
            "Potential_Energy_Reduction_Percent"
        ]
    )

    axis.tick_params(
        axis="x",
        rotation=25
    )

    apply_standard_axis_labels(
        axis,
        title="Potential Energy Reduction by Scenario",
        xlabel="Scenario",
        ylabel="Potential Reduction (%)"
    )

    save_chart(
        figure,
        WHAT_IF_DIR
        / "charts"
        / "02_potential_energy_reduction.png"
    )


def create_co2_reduction_chart(summary):
    """
    Create potential CO2e reduction comparison chart.
    """

    figure, axis = create_standard_figure(
        width=11,
        height=6
    )

    axis.bar(
        summary["Scenario"],
        summary[
            "Potential_CO2_Reduction_kg"
        ]
    )

    axis.tick_params(
        axis="x",
        rotation=25
    )

    apply_standard_axis_labels(
        axis,
        title="Potential CO2e Reduction by Scenario",
        xlabel="Scenario",
        ylabel="Potential CO2e Reduction (kg)"
    )

    save_chart(
        figure,
        WHAT_IF_DIR
        / "charts"
        / "03_potential_co2_reduction.png"
    )


# ---------------------------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------------------------

def validate_scenario_outputs(
    dataframe,
    scenarios,
    summary
):
    """
    Perform final validation of What-If results.
    """

    logger.info(
        "Validating What-If scenario outputs..."
    )

    validation_rows = []

    validation_rows.append(
        {
            "Validation": "Raw Record Count",
            "Expected": 10000,
            "Actual": len(dataframe),
            "Status": (
                "PASS"
                if len(dataframe) == 10000
                else "FAIL"
            ),
        }
    )

    expected_scenarios = 5

    validation_rows.append(
        {
            "Validation": "Scenario Count",
            "Expected": expected_scenarios,
            "Actual": scenarios[
                "Scenario"
            ].nunique(),
            "Status": (
                "PASS"
                if scenarios[
                    "Scenario"
                ].nunique()
                == expected_scenarios
                else "FAIL"
            ),
        }
    )

    expected_rows = (
        len(dataframe)
        * expected_scenarios
    )

    validation_rows.append(
        {
            "Validation": "Scenario Row Count",
            "Expected": expected_rows,
            "Actual": len(scenarios),
            "Status": (
                "PASS"
                if len(scenarios)
                == expected_rows
                else "FAIL"
            ),
        }
    )

    negative_energy = int(
        (scenarios["Scenario_Energy_kWh"] < 0)
        .sum()
    )

    validation_rows.append(
        {
            "Validation": "Negative Scenario Energy",
            "Expected": 0,
            "Actual": negative_energy,
            "Status": (
                "PASS"
                if negative_energy == 0
                else "FAIL"
            ),
        }
    )

    negative_reduction = int(
        (
            scenarios[
                "Potential_Energy_Reduction_kWh"
            ]
            < 0
        ).sum()
    )

    validation_rows.append(
        {
            "Validation": "Negative Energy Reductions",
            "Expected": 0,
            "Actual": negative_reduction,
            "Status": (
                "PASS"
                if negative_reduction == 0
                else "FAIL"
            ),
        }
    )

    production_not_maintained = int(
        (
            summary[
                "Production_Maintained"
            ] == False
        ).sum()
    )

    validation_rows.append(
        {
            "Validation": "Production Maintained",
            "Expected": 0,
            "Actual": production_not_maintained,
            "Status": (
                "PASS"
                if production_not_maintained == 0
                else "FAIL"
            ),
        }
    )

    co2_mismatch = np.isclose(
        scenarios["Scenario_CO2e_kg"],
        scenarios["Scenario_Energy_kWh"]
        * EMISSION_FACTOR,
        atol=0.01
    )

    validation_rows.append(
        {
            "Validation": "CO2e Calculation",
            "Expected": 0,
            "Actual": int(
                (~co2_mismatch).sum()
            ),
            "Status": (
                "PASS"
                if co2_mismatch.all()
                else "FAIL"
            ),
        }
    )

    current = summary[
        summary["Scenario"]
        == "CURRENT_OPERATIONS"
    ]

    current_consistent = False

    if not current.empty:

        current_consistent = np.isclose(
            current[
                "Actual_Energy_kWh"
            ].iloc[0],
            current[
                "Scenario_Energy_kWh"
            ].iloc[0],
            atol=0.01
        )

    validation_rows.append(
        {
            "Validation": (
                "Current Scenario Consistency"
            ),
            "Expected": True,
            "Actual": current_consistent,
            "Status": (
                "PASS"
                if current_consistent
                else "FAIL"
            ),
        }
    )

    validation_rows.append(
        {
            "Validation": "Emission Factor Valid",
            "Expected": "> 0",
            "Actual": EMISSION_FACTOR,
            "Status": (
                "PASS"
                if EMISSION_FACTOR > 0
                else "FAIL"
            ),
        }
    )

    validation = pd.DataFrame(
        validation_rows
    )

    if (
        validation["Status"] == "FAIL"
    ).any():

        raise ValueError(
            "What-If validation failed."
        )

    return validation


# ---------------------------------------------------------------------------
# EXECUTIVE SUMMARY
# ---------------------------------------------------------------------------

def create_executive_summary(
    dataframe,
    summary
):
    """
    Create a compact executive summary for the analysis.
    """

    combined = summary[
        summary["Scenario"]
        == "COMBINED_IMPROVEMENT"
    ]

    combined_energy = (
        combined[
            "Potential_Energy_Reduction_kWh"
        ].iloc[0]
        if not combined.empty
        else 0.0
    )

    combined_percent = (
        combined[
            "Potential_Energy_Reduction_Percent"
        ].iloc[0]
        if not combined.empty
        else 0.0
    )

    combined_co2 = (
        combined[
            "Potential_CO2_Reduction_kg"
        ].iloc[0]
        if not combined.empty
        else 0.0
    )

    actual_energy = (
        dataframe[
            ENERGY_COLUMN
        ].sum()
    )

    actual_co2 = (
        actual_energy
        * EMISSION_FACTOR
    )

    return pd.DataFrame(
        [
            {
                "Metric": "Records Analyzed",
                "Value": len(dataframe),
            },
            {
                "Metric": "Machines Analyzed",
                "Value": dataframe[
                    MACHINE_COLUMN
                ].nunique(),
            },
            {
                "Metric": "Actual Energy (kWh)",
                "Value": actual_energy,
            },
            {
                "Metric": "Actual CO2e (kg)",
                "Value": actual_co2,
            },
            {
                "Metric": (
                    "Combined Potential Energy Reduction (kWh)"
                ),
                "Value": combined_energy,
            },
            {
                "Metric": (
                    "Combined Potential Energy Reduction (%)"
                ),
                "Value": combined_percent,
            },
            {
                "Metric": (
                    "Combined Potential CO2e Reduction (kg)"
                ),
                "Value": combined_co2,
            },
            {
                "Metric": "Production Output (Units)",
                "Value": dataframe[
                    PRODUCTION_COLUMN
                ].sum(),
            },
        ]
    )


# ---------------------------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------------------------

def main():
    """
    Execute the complete What-If scenario workflow.
    """

    logger.info("=" * 78)
    logger.info(
        "SUSTAINABLE MANUFACTURING - WHAT-IF SCENARIO ANALYSIS"
    )
    logger.info("=" * 78)

    try:

        initialize_what_if_directories()

        df = load_baseline_dataset()

        df = prepare_features(df)

        scenarios = generate_scenarios(
            df
        )

        summary = create_scenario_summary(
            scenarios
        )

        assumptions = (
            create_assumption_table()
        )

        validation = (
            validate_scenario_outputs(
                df,
                scenarios,
                summary
            )
        )

        executive_summary = (
            create_executive_summary(
                df,
                summary
            )
        )

        # ---------------------------------------------------------------
        # SAVE OUTPUTS
        # ---------------------------------------------------------------

        logger.info(
            "Saving What-If analytical outputs..."
        )

        save_csv(
            scenarios,
            WHAT_IF_DIR
            / "scenario_detail.csv"
        )

        save_csv(
            summary,
            WHAT_IF_DIR
            / "scenario_summary.csv"
        )

        save_csv(
            assumptions,
            WHAT_IF_DIR
            / "scenario_assumptions.csv"
        )

        save_csv(
            validation,
            WHAT_IF_DIR
            / "what_if_validation.csv"
        )

        save_csv(
            executive_summary,
            WHAT_IF_DIR
            / "what_if_summary.csv"
        )

        # ---------------------------------------------------------------
        # CHARTS
        # ---------------------------------------------------------------

        logger.info(
            "Generating What-If charts..."
        )

        create_energy_comparison_chart(
            summary
        )

        create_reduction_chart(
            summary
        )

        create_co2_reduction_chart(
            summary
        )

        # ---------------------------------------------------------------
        # FINAL OUTPUT
        # ---------------------------------------------------------------

        logger.info("=" * 78)
        logger.info(
            "WHAT-IF SCENARIO ANALYSIS COMPLETED SUCCESSFULLY"
        )
        logger.info("=" * 78)

        logger.info(
            "Records analyzed        : %d",
            len(df)
        )

        logger.info(
            "Scenarios generated     : %d",
            scenarios[
                "Scenario"
            ].nunique()
        )

        logger.info(
            "Actual energy           : %.2f kWh",
            df[ENERGY_COLUMN].sum()
        )

        logger.info(
            "Emission factor         : %.4f kg CO2e/kWh",
            EMISSION_FACTOR
        )

        combined = summary[
            summary["Scenario"]
            == "COMBINED_IMPROVEMENT"
        ]

        if not combined.empty:

            logger.info(
                "Combined potential reduction: %.2f kWh",
                combined[
                    "Potential_Energy_Reduction_kWh"
                ].iloc[0]
            )

            logger.info(
                "Combined potential reduction: %.2f%%",
                combined[
                    "Potential_Energy_Reduction_Percent"
                ].iloc[0]
            )

            logger.info(
                "Potential CO2e reduction: %.2f kg",
                combined[
                    "Potential_CO2_Reduction_kg"
                ].iloc[0]
            )

        logger.info(
            "Output directory        : %s",
            WHAT_IF_DIR
        )

        print()
        print("=" * 78)
        print("WHAT-IF SCENARIO EXECUTIVE SUMMARY")
        print("=" * 78)
        print(
            summary.to_string(
                index=False
            )
        )

        print()
        print("=" * 78)
        print("SCENARIO ASSUMPTIONS")
        print("=" * 78)
        print(
            assumptions.to_string(
                index=False
            )
        )

        print()
        print("=" * 78)
        print("WHAT-IF VALIDATION")
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
            "All scenario reductions are MODELLED POTENTIAL impacts."
        )
        print(
            "They are NOT actual achieved manufacturing savings."
        )

    except Exception as error:

        logger.exception(
            "Unexpected error during What-If analysis: %s",
            error
        )

        raise


if __name__ == "__main__":
    main()