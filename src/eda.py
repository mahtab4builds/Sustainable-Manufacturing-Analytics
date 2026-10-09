"""
Sustainable Manufacturing Analytics
Module: Exploratory Data Analysis (EDA)

Purpose:
    Perform structured exploratory data analysis on the synthetic
    manufacturing energy dataset.

Architecture:
    Shared project utilities are provided by src/library.
    This module contains only EDA-specific analytical logic.

Important:
    The dataset is synthetic and intended for educational,
    analytical, demonstration, and dashboard-development purposes.
"""

# ============================================================================
# IMPORTS
# ============================================================================

import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from library.data_source import load_data

from library.config import (
    DATA_FILE,
    OUTPUT_DIR,
    CHARTS_DIR,
)
from library.constants import (
    MACHINE_IDS,
    VALID_STATES,
)
from library.logger import get_logger
from library.io import (
    ensure_directory,
    load_excel,
    save_csv,
)
from library.validation import (
    validate_required_columns,
    validate_record_count,
)
from library.calculations import (
    calculate_energy_per_unit,
    calculate_percentage,
)
from library.plotting import (
    save_chart,
    create_standard_figure,
    apply_standard_axis_labels,
)


# ============================================================================
# LOGGER
# ============================================================================

logger = get_logger(__name__)


# ============================================================================
# PROJECT CONFIGURATION
# ============================================================================

DATA_SHEET = "Manufacturing_Energy_Data"

EXPECTED_RECORDS = 10_000
EXPECTED_MACHINES = 5

SUMMARY_FILE = OUTPUT_DIR / "EDA_summary.csv"
MACHINE_SUMMARY_FILE = OUTPUT_DIR / "machine_analysis.csv"
IDLE_SUMMARY_FILE = OUTPUT_DIR / "idle_energy_analysis.csv"
SCHEDULE_SUMMARY_FILE = OUTPUT_DIR / "schedule_deviation_analysis.csv"


# ============================================================================
# VISUALIZATION CONFIGURATION
# ============================================================================

sns.set_theme(
    style="whitegrid",
    context="notebook"
)

plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 11


# ============================================================================
# REQUIRED DATASET COLUMNS
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
    "Known_Condition_Flag",
]


# ============================================================================
# DIRECTORY INITIALIZATION
# ============================================================================

def initialize_eda_directories():
    """Create directories required specifically by the EDA module."""

    ensure_directory(OUTPUT_DIR)
    ensure_directory(CHARTS_DIR)

    logger.info(
        "EDA directories initialized."
    )


# ============================================================================
# DATA LOADING
# ============================================================================

def load_dataset():
    return load_data()


# ============================================================================
# DATA TYPE PREPARATION
# ============================================================================

def prepare_data_types(df):
    """Prepare columns for analytical processing."""

    logger.info(
        "Preparing data types..."
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

    numeric_columns = [
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
        "Potential_CO2_Reduction_kg",
    ]

    existing_numeric_columns = [
        column
        for column in numeric_columns
        if column in df.columns
    ]

    df[existing_numeric_columns] = (
        df[existing_numeric_columns]
        .apply(
            pd.to_numeric,
            errors="coerce"
        )
    )

    logger.info(
        "Data types prepared successfully."
    )

    return df


# ============================================================================
# STRUCTURAL VALIDATION
# ============================================================================

def validate_structure(df):
    """Validate the basic structure of the manufacturing dataset."""

    logger.info(
        "Running structural validation..."
    )

    validate_required_columns(
        df,
        REQUIRED_COLUMNS
    )

    validate_record_count(
        df,
        EXPECTED_RECORDS
    )

    duplicate_rows = int(
        df.duplicated().sum()
    )

    duplicate_ids = int(
        df["Record_ID"].duplicated().sum()
    )

    if duplicate_rows:
        raise ValueError(
            f"Duplicate rows detected: {duplicate_rows}"
        )

    if duplicate_ids:
        raise ValueError(
            f"Duplicate Record_ID values detected: {duplicate_ids}"
        )

    machine_count = df["Machine_ID"].nunique()

    if machine_count != EXPECTED_MACHINES:
        raise ValueError(
            f"Expected {EXPECTED_MACHINES} machines, "
            f"found {machine_count}."
        )

    invalid_timestamps = int(
        df["Timestamp"].isna().sum()
    )

    if invalid_timestamps:
        raise ValueError(
            f"Invalid timestamps detected: "
            f"{invalid_timestamps}"
        )

    invalid_machine_ids = set(
        df["Machine_ID"].dropna().unique()
    ) - set(MACHINE_IDS)

    if invalid_machine_ids:
        raise ValueError(
            f"Invalid machine IDs detected: "
            f"{sorted(invalid_machine_ids)}"
        )

    invalid_states = set(
        df["Actual_State"].dropna().unique()
    ) - set(VALID_STATES)

    if invalid_states:
        raise ValueError(
            f"Invalid actual states detected: "
            f"{sorted(invalid_states)}"
        )

    logger.info(
        "Structural validation passed."
    )

    logger.info(
        "Duplicate rows: %s | Duplicate Record IDs: %s",
        duplicate_rows,
        duplicate_ids
    )


# ============================================================================
# DATA QUALITY ANALYSIS
# ============================================================================

def analyze_data_quality(df):
    """Generate and save a complete data-quality summary."""

    logger.info(
        "Analyzing data quality..."
    )

    quality = pd.DataFrame({
        "Column": df.columns,
        "Data_Type": [
            str(dtype)
            for dtype in df.dtypes
        ],
        "Missing_Count": [
            int(df[column].isna().sum())
            for column in df.columns
        ],
        "Missing_Percentage": [
            round(
                df[column].isna().mean() * 100,
                3
            )
            for column in df.columns
        ],
        "Unique_Values": [
            int(df[column].nunique())
            for column in df.columns
        ],
    })

    save_csv(
        quality,
        OUTPUT_DIR / "data_quality_summary.csv"
    )

    total_missing = int(
        df.isna().sum().sum()
    )

    logger.info(
        "Total missing values detected: %s",
        total_missing
    )

    return quality


# ============================================================================
# MACHINE PERFORMANCE ANALYSIS
# ============================================================================

def analyze_machine_performance(df):
    """Calculate machine-level energy and production metrics."""

    logger.info(
        "Analyzing machine performance..."
    )

    summary = (
        df.groupby(
            [
                "Machine_ID",
                "Machine_Name",
                "Machine_Type",
            ],
            as_index=False
        )
        .agg(
            Total_Energy_kWh=(
                "Energy_Consumption_kWh",
                "sum"
            ),
            Total_Production_Units=(
                "Actual_Production_Units",
                "sum"
            ),
            Total_Idle_Energy_kWh=(
                "Idle_Energy_kWh",
                "sum"
            ),
            Running_Hours=(
                "Running_Hours",
                "sum"
            ),
            Idle_Hours=(
                "Idle_Hours",
                "sum"
            ),
            CO2e_kg=(
                "CO2e_kg",
                "sum"
            ),
        )
    )

    summary["Energy_Per_Unit_kWh"] = (
        calculate_energy_per_unit(
            summary["Total_Energy_kWh"],
            summary["Total_Production_Units"]
        )
    )

    summary["Idle_Energy_Per_Hour_kWh"] = (
        calculate_energy_per_unit(
            summary["Total_Idle_Energy_kWh"],
            summary["Idle_Hours"]
        )
    )

    summary = summary.sort_values(
        "Total_Energy_kWh",
        ascending=False
    )

    save_csv(
        summary,
        MACHINE_SUMMARY_FILE
    )

    return summary


# ============================================================================
# MACHINE ENERGY CHART
# ============================================================================

def plot_machine_energy(machine_summary):
    """Generate machine-level energy consumption chart."""

    logger.info(
        "Generating machine energy chart..."
    )

    plot_data = machine_summary.sort_values(
        "Total_Energy_kWh"
    )

    figure, axis = create_standard_figure(
        width=10,
        height=6
    )

    axis.barh(
        plot_data["Machine_ID"].astype(str),
        plot_data["Total_Energy_kWh"]
    )

    apply_standard_axis_labels(
        axis,
        title="Total Energy Consumption by Machine",
        xlabel="Energy Consumption (kWh)",
        ylabel="Machine"
    )

    save_chart(
        figure,
        CHARTS_DIR / "01_energy_by_machine.png"
    )

    logger.info(
        "Chart saved: 01_energy_by_machine.png"
    )


# ============================================================================
# MACHINE PRODUCTION CHART
# ============================================================================

def plot_machine_production(machine_summary):
    """Generate machine-level production chart."""

    logger.info(
        "Generating machine production chart..."
    )

    plot_data = machine_summary.sort_values(
        "Total_Production_Units"
    )

    figure, axis = create_standard_figure(
        width=10,
        height=6
    )

    axis.barh(
        plot_data["Machine_ID"].astype(str),
        plot_data["Total_Production_Units"]
    )

    apply_standard_axis_labels(
        axis,
        title="Total Production by Machine",
        xlabel="Production Units",
        ylabel="Machine"
    )

    save_chart(
        figure,
        CHARTS_DIR / "02_production_by_machine.png"
    )

    logger.info(
        "Chart saved: 02_production_by_machine.png"
    )


# ============================================================================
# ENERGY INTENSITY
# ============================================================================

def analyze_energy_intensity(machine_summary):
    """Rank machines by energy consumed per production unit."""

    logger.info(
        "Analyzing energy intensity..."
    )

    result = machine_summary[
        [
            "Machine_ID",
            "Machine_Name",
            "Total_Energy_kWh",
            "Total_Production_Units",
            "Energy_Per_Unit_kWh",
        ]
    ].copy()

    result = result.sort_values(
        "Energy_Per_Unit_kWh",
        ascending=False
    )

    save_csv(
        result,
        OUTPUT_DIR / "energy_intensity_analysis.csv"
    )

    return result


def plot_energy_intensity(intensity_summary):
    """Generate machine energy-intensity chart."""

    logger.info(
        "Generating energy intensity chart..."
    )

    plot_data = intensity_summary.sort_values(
        "Energy_Per_Unit_kWh"
    )

    figure, axis = create_standard_figure(
        width=10,
        height=6
    )

    axis.barh(
        plot_data["Machine_ID"].astype(str),
        plot_data["Energy_Per_Unit_kWh"]
    )

    apply_standard_axis_labels(
        axis,
        title="Energy Intensity by Machine",
        xlabel="Energy per Production Unit (kWh/unit)",
        ylabel="Machine"
    )

    save_chart(
        figure,
        CHARTS_DIR / "03_energy_intensity.png"
    )

    logger.info(
        "Chart saved: 03_energy_intensity.png"
    )


# ============================================================================
# MACHINE STATE ANALYSIS
# ============================================================================

def analyze_machine_states(df):
    """Calculate machine operating-state distribution."""

    logger.info(
        "Analyzing machine operating states..."
    )

    summary = (
        df["Actual_State"]
        .value_counts()
        .rename_axis("Actual_State")
        .reset_index(name="Observations")
    )

    summary["Percentage"] = (
        summary["Observations"]
        / len(df)
        * 100
    ).round(2)

    save_csv(
        summary,
        OUTPUT_DIR / "machine_state_analysis.csv"
    )

    return summary


def plot_machine_states(df):
    """Generate machine-state distribution chart."""

    logger.info(
        "Generating machine-state chart..."
    )

    state_counts = (
        df["Actual_State"]
        .value_counts()
        .reindex(
            VALID_STATES,
            fill_value=0
        )
    )

    figure, axis = create_standard_figure(
        width=9,
        height=6
    )

    axis.bar(
        state_counts.index,
        state_counts.values
    )

    apply_standard_axis_labels(
        axis,
        title="Machine Operating State Distribution",
        xlabel="Actual State",
        ylabel="Observations"
    )

    save_chart(
        figure,
        CHARTS_DIR / "04_machine_state_distribution.png"
    )

    logger.info(
        "Chart saved: 04_machine_state_distribution.png"
    )


# ============================================================================
# IDLE ENERGY ANALYSIS
# ============================================================================

def analyze_idle_energy(df):
    """Calculate machine-level idle-energy metrics."""

    logger.info(
        "Analyzing idle energy consumption..."
    )

    machine_total_energy = (
        df.groupby("Machine_ID")
        ["Energy_Consumption_kWh"]
        .sum()
    )

    summary = (
        df.groupby(
            "Machine_ID",
            as_index=False
        )
        .agg(
            Idle_Hours=(
                "Idle_Hours",
                "sum"
            ),
            Idle_Energy_kWh=(
                "Idle_Energy_kWh",
                "sum"
            ),
        )
    )

    summary["Idle_Energy_Per_Hour_kWh"] = (
        summary["Idle_Energy_kWh"]
        / summary["Idle_Hours"].replace(
            0,
            np.nan
        )
    )

    summary["Total_Energy_kWh"] = (
        summary["Machine_ID"]
        .map(machine_total_energy)
    )

    summary["Idle_Energy_Percentage"] = (
        summary["Idle_Energy_kWh"]
        / summary["Total_Energy_kWh"]
        * 100
    )

    summary = summary.sort_values(
        "Idle_Energy_kWh",
        ascending=False
    )

    save_csv(
        summary,
        IDLE_SUMMARY_FILE
    )

    return summary


def plot_idle_energy(idle_summary):
    """Generate idle-energy comparison chart."""

    logger.info(
        "Generating idle-energy chart..."
    )

    plot_data = idle_summary.sort_values(
        "Idle_Energy_kWh"
    )

    figure, axis = create_standard_figure(
        width=10,
        height=6
    )

    axis.barh(
        plot_data["Machine_ID"].astype(str),
        plot_data["Idle_Energy_kWh"]
    )

    apply_standard_axis_labels(
        axis,
        title="Idle Energy Consumption by Machine",
        xlabel="Idle Energy (kWh)",
        ylabel="Machine"
    )

    save_chart(
        figure,
        CHARTS_DIR / "05_idle_energy.png"
    )

    logger.info(
        "Chart saved: 05_idle_energy.png"
    )


# ============================================================================
# DAILY BASELINE ANALYSIS
# ============================================================================

def create_daily_baseline_analysis(df):
    """Compare daily actual energy with baseline energy."""

    logger.info(
        "Analyzing actual versus baseline energy..."
    )

    daily = (
        df.groupby(
            "Date",
            as_index=False
        )
        .agg(
            Actual_Energy_kWh=(
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
        )
    )

    daily["Energy_Deviation_kWh"] = (
        daily["Actual_Energy_kWh"]
        - daily["Baseline_Energy_kWh"]
    )

    daily["Energy_Deviation_Percent"] = np.where(
        daily["Baseline_Energy_kWh"] > 0,
        (
            daily["Energy_Deviation_kWh"]
            / daily["Baseline_Energy_kWh"]
            * 100
        ),
        np.nan
    )

    save_csv(
        daily,
        OUTPUT_DIR / "daily_baseline_analysis.csv"
    )

    return daily


def plot_actual_vs_baseline(daily_baseline):
    """Generate actual-versus-baseline energy chart."""

    logger.info(
        "Generating baseline comparison chart..."
    )

    figure, axis = create_standard_figure(
        width=14,
        height=6
    )

    axis.plot(
        daily_baseline["Date"],
        daily_baseline["Actual_Energy_kWh"],
        label="Actual Energy",
        linewidth=2
    )

    axis.plot(
        daily_baseline["Date"],
        daily_baseline["Baseline_Energy_kWh"],
        label="Baseline Energy",
        linewidth=2
    )

    apply_standard_axis_labels(
        axis,
        title="Actual vs Baseline Energy Consumption",
        xlabel="Date",
        ylabel="Energy (kWh)"
    )

    axis.legend()
    axis.tick_params(axis="x", rotation=45)

    save_chart(
        figure,
        CHARTS_DIR / "06_actual_vs_baseline.png"
    )

    logger.info(
        "Chart saved: 06_actual_vs_baseline.png"
    )


def plot_energy_deviation(daily_baseline):
    """Generate daily baseline-deviation chart."""

    logger.info(
        "Generating energy deviation chart..."
    )

    figure, axis = create_standard_figure(
        width=14,
        height=5
    )

    axis.plot(
        daily_baseline["Date"],
        daily_baseline["Energy_Deviation_kWh"],
        linewidth=1.8
    )

    axis.axhline(
        0,
        linestyle="--",
        linewidth=1
    )

    apply_standard_axis_labels(
        axis,
        title="Daily Energy Deviation from Baseline",
        xlabel="Date",
        ylabel="Energy Deviation (kWh)"
    )

    axis.tick_params(axis="x", rotation=45)

    save_chart(
        figure,
        CHARTS_DIR / "07_energy_deviation.png"
    )

    logger.info(
        "Chart saved: 07_energy_deviation.png"
    )


# ============================================================================
# HOURLY LOAD PROFILE
# ============================================================================

def analyze_hourly_load_profile(df):
    """Calculate hourly energy load statistics."""

    logger.info(
        "Analyzing hourly energy load profile..."
    )

    hourly = (
        df.assign(
            Hour=df["Timestamp"].dt.hour
        )
        .groupby("Hour")["Energy_Consumption_kWh"]
        .agg(
            Average_Energy_kWh="mean",
            Maximum_Energy_kWh="max",
            Minimum_Energy_kWh="min"
        )
        .reset_index()
        .sort_values("Hour")
    )

    save_csv(
        hourly,
        OUTPUT_DIR / "hourly_load_profile.csv"
    )

    return hourly


def plot_hourly_load_profile(hourly_profile):
    """Generate average hourly energy load profile."""

    logger.info(
        "Generating hourly load profile chart..."
    )

    figure, axis = create_standard_figure(
        width=13,
        height=6
    )

    axis.plot(
        hourly_profile["Hour"],
        hourly_profile["Average_Energy_kWh"],
        marker="o",
        linewidth=2,
        label="Average Energy"
    )

    apply_standard_axis_labels(
        axis,
        title="Average Hourly Energy Load Profile",
        xlabel="Hour of Day",
        ylabel="Average Energy Consumption (kWh)"
    )

    axis.set_xticks(
        range(24)
    )

    axis.legend()

    save_chart(
        figure,
        CHARTS_DIR / "08_hourly_load_profile.png"
    )

    logger.info(
        "Chart saved: 08_hourly_load_profile.png"
    )


# ============================================================================
# SHIFT ANALYSIS
# ============================================================================

def create_shift_column(df):
    """Assign each observation to a manufacturing shift."""

    df = df.copy()

    hour = df["Timestamp"].dt.hour

    df["Shift"] = np.select(
        [
            hour.between(6, 13),
            hour.between(14, 21),
        ],
        [
            "Shift A",
            "Shift B",
        ],
        default="Shift C"
    )

    return df


def analyze_shifts(df):
    """Calculate energy and production metrics by shift."""

    logger.info(
        "Analyzing manufacturing shifts..."
    )

    summary = (
        df.groupby(
            "Shift",
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
            Idle_Energy_kWh=(
                "Idle_Energy_kWh",
                "sum"
            ),
        )
    )

    summary["Energy_Per_Unit_kWh"] = (
        calculate_energy_per_unit(
            summary["Energy_kWh"],
            summary["Production_Units"]
        )
    )

    save_csv(
        summary,
        OUTPUT_DIR / "shift_analysis.csv"
    )

    return summary


def plot_shift_energy(shift_summary):
    """Generate shift-level energy chart."""

    logger.info(
        "Generating shift energy chart..."
    )

    figure, axis = create_standard_figure(
        width=8,
        height=5
    )

    axis.bar(
        shift_summary["Shift"],
        shift_summary["Energy_kWh"]
    )

    apply_standard_axis_labels(
        axis,
        title="Energy Consumption by Manufacturing Shift",
        xlabel="Shift",
        ylabel="Energy Consumption (kWh)"
    )

    save_chart(
        figure,
        CHARTS_DIR / "09_energy_by_shift.png"
    )

    logger.info(
        "Chart saved: 09_energy_by_shift.png"
    )


# ============================================================================
# PRODUCTION-ENERGY RELATIONSHIP
# ============================================================================

def calculate_production_energy_correlation(df):
    """Calculate Pearson correlation between production and energy."""

    return float(
        df[
            [
                "Actual_Production_Units",
                "Energy_Consumption_kWh",
            ]
        ]
        .corr()
        .loc[
            "Actual_Production_Units",
            "Energy_Consumption_kWh",
        ]
    )


def plot_production_energy_relationship(df):
    """Generate production-versus-energy scatter plot."""

    logger.info(
        "Analyzing production-energy relationship..."
    )

    sample_size = min(
        3000,
        len(df)
    )

    sample = df.sample(
        n=sample_size,
        random_state=42
    )

    figure, axis = create_standard_figure(
        width=10,
        height=7
    )

    sns.scatterplot(
        data=sample,
        x="Actual_Production_Units",
        y="Energy_Consumption_kWh",
        hue="Actual_State",
        alpha=0.55,
        ax=axis
    )

    apply_standard_axis_labels(
        axis,
        title="Production vs Energy Consumption",
        xlabel="Actual Production Units",
        ylabel="Energy Consumption (kWh)"
    )

    axis.legend(
        title="Machine State"
    )

    save_chart(
        figure,
        CHARTS_DIR / "10_production_vs_energy.png"
    )

    logger.info(
        "Chart saved: 10_production_vs_energy.png"
    )


# ============================================================================
# CO2e ANALYSIS
# ============================================================================

def analyze_co2(df):
    """Calculate machine-level CO2e emissions."""

    logger.info(
        "Analyzing CO2e emissions..."
    )

    summary = (
        df.groupby(
            "Machine_ID",
            as_index=False
        )
        .agg(
            Energy_kWh=(
                "Energy_Consumption_kWh",
                "sum"
            ),
            CO2e_kg=(
                "CO2e_kg",
                "sum"
            ),
            Baseline_CO2e_kg=(
                "CO2e_Baseline_kg",
                "sum"
            ),
            Potential_CO2_Reduction_kg=(
                "Potential_CO2_Reduction_kg",
                "sum"
            ),
        )
        .sort_values(
            "CO2e_kg",
            ascending=False
        )
    )

    save_csv(
        summary,
        OUTPUT_DIR / "co2_machine_analysis.csv"
    )

    return summary


def plot_co2_by_machine(co2_summary):
    """Generate machine-level CO2e emissions chart."""

    logger.info(
        "Generating CO2e chart..."
    )

    plot_data = co2_summary.sort_values(
        "CO2e_kg"
    )

    figure, axis = create_standard_figure(
        width=10,
        height=6
    )

    axis.barh(
        plot_data["Machine_ID"].astype(str),
        plot_data["CO2e_kg"]
    )

    apply_standard_axis_labels(
        axis,
        title="CO₂e Emissions by Machine",
        xlabel="CO₂e (kg)",
        ylabel="Machine"
    )

    save_chart(
        figure,
        CHARTS_DIR / "11_co2_by_machine.png"
    )

    logger.info(
        "Chart saved: 11_co2_by_machine.png"
    )


# ============================================================================
# HIGH-CONSUMPTION EVENTS
# ============================================================================

def identify_high_consumption_events(
    df,
    deviation_threshold=20.0
):
    """
    Identify running-state observations whose energy
    consumption exceeds baseline by the defined threshold.

    These are investigation candidates and are not
    automatically classified as confirmed inefficiencies.
    """

    logger.info(
        "Identifying potential high-consumption events..."
    )

    events = df[
        (
            df["Energy_Deviation_Percent"]
            > deviation_threshold
        )
        &
        (
            df["Actual_State"]
            == "RUNNING"
        )
    ].copy()

    events = events.sort_values(
        "Energy_Deviation_Percent",
        ascending=False
    )

    save_csv(
        events,
        OUTPUT_DIR
        / "potential_high_consumption_events.csv"
    )

    logger.info(
        "Potential high-consumption events identified: %s",
        len(events)
    )

    return events


# ============================================================================
# SCHEDULE DEVIATION ANALYSIS
# ============================================================================

def analyze_schedule_deviations(df):
    """Analyze planned versus actual machine operation."""

    logger.info(
        "Analyzing planned versus actual operation..."
    )

    deviations = df[
        df["Planned_State"]
        !=
        df["Actual_State"]
    ].copy()

    summary = (
        deviations.groupby(
            "Machine_ID",
            as_index=False
        )
        .agg(
            Deviations=(
                "Record_ID",
                "count"
            ),
            Energy_kWh=(
                "Energy_Consumption_kWh",
                "sum"
            ),
            Idle_Energy_kWh=(
                "Idle_Energy_kWh",
                "sum"
            ),
        )
        .sort_values(
            "Deviations",
            ascending=False
        )
    )

    save_csv(
        summary,
        SCHEDULE_SUMMARY_FILE
    )

    logger.info(
        "Schedule deviation observations: %s",
        len(deviations)
    )

    return summary


# ============================================================================
# EXECUTIVE SUMMARY
# ============================================================================

def create_eda_summary(
    df,
    correlation,
    high_consumption_count,
    schedule_deviation_count
):
    """Create the EDA executive summary."""

    logger.info(
        "Creating EDA executive summary..."
    )

    summary = pd.DataFrame({
        "Metric": [
            "Total Records",
            "Number of Machines",
            "Date Range Start",
            "Date Range End",
            "Total Energy (kWh)",
            "Total Production Units",
            "Total Idle Energy (kWh)",
            "Total CO2e (kg)",
            "Potential High-Consumption Records",
            "Schedule Deviation Records",
            "Production-Energy Correlation",
        ],
        "Value": [
            len(df),
            df["Machine_ID"].nunique(),
            df["Timestamp"].min(),
            df["Timestamp"].max(),
            round(
                df["Energy_Consumption_kWh"].sum(),
                2
            ),
            round(
                df["Actual_Production_Units"].sum(),
                2
            ),
            round(
                df["Idle_Energy_kWh"].sum(),
                2
            ),
            round(
                df["CO2e_kg"].sum(),
                2
            ),
            high_consumption_count,
            schedule_deviation_count,
            round(
                correlation,
                4
            ),
        ],
    })

    save_csv(
        summary,
        SUMMARY_FILE
    )

    return summary


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Execute the complete EDA pipeline."""

    logger.info("=" * 72)
    logger.info(
        "SUSTAINABLE MANUFACTURING - PYTHON EDA"
    )
    logger.info("=" * 72)

    try:

        # Step 1: Initialize directories.
        initialize_eda_directories()

        # Step 2: Load source data.
        df = load_dataset()

        # Step 3: Prepare analytical data types.
        df = prepare_data_types(df)

        # Step 4: Validate structure.
        validate_structure(df)

        # Step 5: Assess data quality.
        analyze_data_quality(df)

        # Step 6: Machine performance.
        machine_summary = (
            analyze_machine_performance(df)
        )

        plot_machine_energy(
            machine_summary
        )

        plot_machine_production(
            machine_summary
        )

        # Step 7: Energy intensity.
        intensity_summary = (
            analyze_energy_intensity(
                machine_summary
            )
        )

        plot_energy_intensity(
            intensity_summary
        )

        # Step 8: Machine states.
        analyze_machine_states(df)
        plot_machine_states(df)

        # Step 9: Idle energy.
        idle_summary = (
            analyze_idle_energy(df)
        )

        plot_idle_energy(
            idle_summary
        )

        # Step 10: Baseline comparison.
        daily_baseline = (
            create_daily_baseline_analysis(df)
        )

        plot_actual_vs_baseline(
            daily_baseline
        )

        plot_energy_deviation(
            daily_baseline
        )

        # Step 11: Hourly load profile.
        hourly_profile = (
            analyze_hourly_load_profile(df)
        )

        plot_hourly_load_profile(
            hourly_profile
        )

        # Step 12: Shift analysis.
        df = create_shift_column(df)

        shift_summary = (
            analyze_shifts(df)
        )

        plot_shift_energy(
            shift_summary
        )

        # Step 13: Production-energy relationship.
        correlation = (
            calculate_production_energy_correlation(df)
        )

        plot_production_energy_relationship(df)

        # Step 14: CO2e analysis.
        co2_summary = analyze_co2(df)

        plot_co2_by_machine(
            co2_summary
        )

        # Step 15: High-consumption detection.
        high_consumption = (
            identify_high_consumption_events(df)
        )

        # Step 16: Schedule analysis.
        schedule_summary = (
            analyze_schedule_deviations(df)
        )

        schedule_deviation_count = int(
            schedule_summary["Deviations"].sum()
        )

        # Step 17: Executive summary.
        summary = create_eda_summary(
            df=df,
            correlation=correlation,
            high_consumption_count=len(
                high_consumption
            ),
            schedule_deviation_count=(
                schedule_deviation_count
            ),
        )

        # Step 18: Final report.
        logger.info("=" * 72)
        logger.info(
            "EDA COMPLETED SUCCESSFULLY"
        )
        logger.info("=" * 72)

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
            df["Energy_Consumption_kWh"].sum()
        )

        logger.info(
            "Total production        : %.2f units",
            df["Actual_Production_Units"].sum()
        )

        logger.info(
            "Total idle energy       : %.2f kWh",
            df["Idle_Energy_kWh"].sum()
        )

        logger.info(
            "Total CO2e              : %.2f kg",
            df["CO2e_kg"].sum()
        )

        logger.info(
            "Production-energy corr. : %.4f",
            correlation
        )

        logger.info(
            "High-consumption events : %s",
            len(high_consumption)
        )

        logger.info(
            "Schedule deviations     : %s",
            schedule_deviation_count
        )

        logger.info(
            "Charts directory        : %s",
            CHARTS_DIR
        )

        logger.info(
            "Analysis outputs        : %s",
            OUTPUT_DIR
        )

        print()
        print("=" * 72)
        print("EDA EXECUTIVE SUMMARY")
        print("=" * 72)
        print(
            summary.to_string(
                index=False
            )
        )
        print("=" * 72)
        print(
            "EDA completed successfully."
        )
        print("=" * 72)

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
            "Unexpected error during EDA execution: %s",
            error
        )

        sys.exit(1)


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()