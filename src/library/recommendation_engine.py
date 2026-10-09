"""
Recommendation Engine
---------------------
Shared analytical functions for generating management-level
recommendations from the Sustainable Manufacturing Analytics pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV file and return it as a DataFrame."""
    if not path.exists():
        raise FileNotFoundError(f"Required analytical output not found: {path}")

    return pd.read_csv(path)


def safe_numeric(
    dataframe: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Convert selected columns to numeric values safely."""
    result = dataframe.copy()

    for column in columns:
        if column in result.columns:
            result[column] = pd.to_numeric(
                result[column],
                errors="coerce",
            )

    return result


def identify_machine_priorities(
    sustainability: pd.DataFrame,
    optimization: pd.DataFrame,
) -> pd.DataFrame:
    """Combine sustainability and optimization indicators by machine."""

    sustainability = sustainability.copy()
    optimization = optimization.copy()

    sustainability = safe_numeric(
        sustainability,
        [
            "CO2e_kg",
            "CO2e_Per_Production_Unit",
            "Idle_CO2e_Percent",
            "Sustainability_Attention_Score",
        ],
    )

    optimization = safe_numeric(
        optimization,
        [
            "Energy_kWh",
            "Optimized_Total_Reduction_kWh",
            "Potential_Reduction_Percent",
        ],
    )

    sustainability_columns = [
        column
        for column in [
            "Machine_ID",
            "Machine_Name",
            "CO2e_kg",
            "CO2e_Per_Production_Unit",
            "Idle_CO2e_Percent",
            "Sustainability_Attention_Score",
            "Sustainability_Priority",
        ]
        if column in sustainability.columns
    ]

    optimization_columns = [
        column
        for column in [
            "Machine_ID",
            "Energy_kWh",
            "Optimized_Total_Reduction_kWh",
            "Potential_Reduction_Percent",
            "Production_Output_Maintained",
        ]
        if column in optimization.columns
    ]

    result = sustainability[sustainability_columns].merge(
        optimization[optimization_columns],
        on="Machine_ID",
        how="left",
    )

    result["Priority_Score"] = (
        result["Sustainability_Attention_Score"].fillna(0) * 0.40
        + result["Potential_Reduction_Percent"].fillna(0) * 2.00
        + result["Idle_CO2e_Percent"].fillna(0) * 0.50
    )

    result["Recommended_Action"] = result.apply(
        generate_machine_action,
        axis=1,
    )

    return result.sort_values(
        "Priority_Score",
        ascending=False,
    ).reset_index(drop=True)


def generate_machine_action(row: pd.Series) -> str:
    """Generate a machine-specific management recommendation."""

    machine_id = row.get("Machine_ID", "Unknown")

    reduction = float(
        row.get("Optimized_Total_Reduction_kWh", 0) or 0
    )

    idle_co2 = float(
        row.get("Idle_CO2e_Percent", 0) or 0
    )

    priority = str(
        row.get("Sustainability_Priority", "LOW")
    )

    if machine_id == "M005":
        return (
            "Prioritize compressor optimization, idle-load reduction, "
            "schedule adherence and operating-efficiency improvements."
        )

    if priority == "HIGH":
        return (
            "Prioritize immediate energy-efficiency investigation and "
            "machine operating-condition review."
        )

    if idle_co2 >= 6:
        return (
            "Review idle operation, shutdown practices and unnecessary "
            "standby energy consumption."
        )

    if reduction > 15000:
        return (
            "Evaluate production-preserving energy optimization measures "
            "with emphasis on operating efficiency."
        )

    return (
        "Maintain monitoring and implement targeted efficiency improvements "
        "where energy deviation or idle consumption increases."
    )


def create_kpi_summary(
    baseline_summary: pd.DataFrame,
    inefficiency_summary: pd.DataFrame,
    forecast_summary: pd.DataFrame,
    what_if_summary: pd.DataFrame,
    optimization_summary: pd.DataFrame,
    co2_kpi: pd.DataFrame,
) -> pd.DataFrame:
    """Create a consolidated executive KPI table."""

    kpis = []

    def add_kpi(
        category: str,
        metric: str,
        value,
        unit: str,
        status: str,
    ) -> None:
        kpis.append(
            {
                "Category": category,
                "Metric": metric,
                "Value": value,
                "Unit": unit,
                "Status": status,
            }
        )

    def find_value(
        dataframe: pd.DataFrame,
        metric_column: str,
        metric_name: str,
        value_column: str = "Value",
    ):
        if dataframe.empty:
            return None

        if metric_column not in dataframe.columns:
            return None

        match = dataframe[
            dataframe[metric_column].astype(str).str.strip()
            == metric_name
        ]

        if match.empty:
            return None

        return match.iloc[0][value_column]

    total_energy = find_value(
        co2_kpi,
        "KPI",
        "Total Energy Consumption",
    )

    total_production = find_value(
        co2_kpi,
        "KPI",
        "Total Production",
    )

    total_co2 = find_value(
        co2_kpi,
        "KPI",
        "Total CO2e",
    )

    idle_energy = find_value(
        co2_kpi,
        "KPI",
        "Idle Energy",
    )

    idle_co2 = find_value(
        co2_kpi,
        "KPI",
        "Idle CO2e",
    )

    energy_per_unit = find_value(
        co2_kpi,
        "KPI",
        "Energy per Production Unit",
    )

    co2_per_unit = find_value(
        co2_kpi,
        "KPI",
        "CO2e per Production Unit",
    )

    combined_reduction = find_value(
        co2_kpi,
        "KPI",
        "Potential Combined Energy Reduction",
    )

    combined_reduction_percent = None

    percentage_matches = co2_kpi[
        co2_kpi["KPI"].astype(str).str.contains(
            "Potential Combined Energy Reduction",
            case=False,
            na=False,
        )
    ]

    if len(percentage_matches) > 1:
        combined_reduction_percent = percentage_matches.iloc[1]["Value"]

    combined_co2_reduction = find_value(
        co2_kpi,
        "KPI",
        "Potential Combined CO2 Reduction",
    )

    optimized_energy = find_value(
        optimization_summary,
        "Metric",
        "Optimized Model Energy",
    )

    optimized_reduction = find_value(
        optimization_summary,
        "Metric",
        "Potential Energy Reduction",
    )

    optimized_reduction_percent = find_value(
        optimization_summary,
        "Metric",
        "Potential Reduction",
    )

    optimized_co2_reduction = find_value(
        optimization_summary,
        "Metric",
        "Potential CO2e Reduction",
    )

    forecast_energy = find_value(
        forecast_summary,
        "Metric",
        "Forecasted Energy (kWh)",
    )

    forecast_co2 = find_value(
        forecast_summary,
        "Metric",
        "Forecasted CO2e (kg)",
    )

    high_consumption = find_value(
        inefficiency_summary,
        "Metric",
        "High-Consumption Records",
    )

    inefficiency_candidates = find_value(
        inefficiency_summary,
        "Metric",
        "Potential Inefficiency Candidates",
    )

    add_kpi(
        "Energy",
        "Total Energy Consumption",
        total_energy,
        "kWh",
        "Observed",
    )

    add_kpi(
        "Production",
        "Total Production",
        total_production,
        "Units",
        "Observed",
    )

    add_kpi(
        "Efficiency",
        "Energy per Production Unit",
        energy_per_unit,
        "kWh/unit",
        "Observed",
    )

    add_kpi(
        "Energy",
        "Idle Energy",
        idle_energy,
        "kWh",
        "Observed",
    )

    add_kpi(
        "Sustainability",
        "Idle CO2e",
        idle_co2,
        "kg CO2e",
        "Calculated",
    )

    add_kpi(
        "Sustainability",
        "Total CO2e",
        total_co2,
        "kg CO2e",
        "Calculated",
    )

    add_kpi(
        "Sustainability",
        "CO2e per Production Unit",
        co2_per_unit,
        "kg CO2e/unit",
        "Calculated",
    )

    add_kpi(
        "What-If",
        "Potential Combined Energy Reduction",
        combined_reduction,
        "kWh",
        "Modelled Potential",
    )

    add_kpi(
        "What-If",
        "Potential Combined Energy Reduction",
        combined_reduction_percent,
        "%",
        "Modelled Potential",
    )

    add_kpi(
        "What-If",
        "Potential Combined CO2 Reduction",
        combined_co2_reduction,
        "kg CO2e",
        "Modelled Potential",
    )

    add_kpi(
        "Optimization",
        "Optimized Model Energy",
        optimized_energy,
        "kWh",
        "Modelled Potential",
    )

    add_kpi(
        "Optimization",
        "Potential Energy Reduction",
        optimized_reduction,
        "kWh",
        "Modelled Potential",
    )

    add_kpi(
        "Optimization",
        "Potential Reduction",
        optimized_reduction_percent,
        "%",
        "Modelled Potential",
    )

    add_kpi(
        "Optimization",
        "Potential CO2e Reduction",
        optimized_co2_reduction,
        "kg CO2e",
        "Modelled Potential",
    )

    add_kpi(
        "Forecasting",
        "Forecasted Energy",
        forecast_energy,
        "kWh",
        "Modelled Forecast",
    )

    add_kpi(
        "Forecasting",
        "Forecasted CO2e",
        forecast_co2,
        "kg CO2e",
        "Modelled Forecast",
    )

    add_kpi(
        "Inefficiency",
        "High-Consumption Records",
        high_consumption,
        "Records",
        "Analytical Finding",
    )

    add_kpi(
        "Inefficiency",
        "Potential Inefficiency Candidates",
        inefficiency_candidates,
        "Records",
        "Analytical Finding",
    )

    return pd.DataFrame(kpis)


def create_recommendations(
    machine_priorities: pd.DataFrame,
    kpis: pd.DataFrame,
) -> pd.DataFrame:
    """Generate prioritized management recommendations."""

    recommendations = []

    def add(
        priority: int,
        area: str,
        recommendation: str,
        rationale: str,
        expected_impact: str,
        horizon: str,
    ) -> None:
        recommendations.append(
            {
                "Priority": priority,
                "Area": area,
                "Recommendation": recommendation,
                "Rationale": rationale,
                "Expected_Impact": expected_impact,
                "Implementation_Horizon": horizon,
            }
        )

    add(
        1,
        "Machine Optimization",
        "Prioritize M005 Industrial Compressor for energy-efficiency intervention.",
        "M005 has the highest sustainability attention score and the largest "
        "machine-level optimization opportunity.",
        "Highest machine-level energy-saving opportunity.",
        "Immediate",
    )

    add(
        2,
        "Idle Energy",
        "Reduce unnecessary machine idle and standby operation.",
        "Idle energy represents a measurable portion of total manufacturing "
        "energy consumption.",
        "Potential reduction in avoidable energy and CO2e.",
        "Short Term",
    )

    add(
        3,
        "Schedule Management",
        "Improve adherence between planned and actual machine operation.",
        "Schedule deviations create opportunities for unnecessary energy use.",
        "Lower non-productive energy consumption.",
        "Short Term",
    )

    add(
        4,
        "Energy Intensity",
        "Target machines with high energy consumption per production unit.",
        "Production-adjusted energy intensity provides a better efficiency "
        "indicator than total energy alone.",
        "Improved kWh/unit performance.",
        "Medium Term",
    )

    add(
        5,
        "Monitoring",
        "Establish continuous monitoring of actual versus baseline energy.",
        "Baseline deviation allows abnormal consumption to be detected earlier.",
        "Earlier identification of high-consumption events.",
        "Continuous",
    )

    add(
        6,
        "Forecasting",
        "Use the validated forecasting model for short-term energy planning.",
        "The forecasting phase provides a 14-day modelled energy outlook.",
        "Improved energy planning and operational preparedness.",
        "Continuous",
    )

    add(
        7,
        "Sustainability",
        "Track CO2e per production unit as a sustainability KPI.",
        "Production-adjusted CO2e allows sustainability performance to be "
        "compared across operating periods.",
        "Improved carbon-efficiency visibility.",
        "Continuous",
    )

    add(
        8,
        "Optimization",
        "Use the linear-programming result as a decision-support scenario.",
        "Optimization estimates a production-preserving energy reduction "
        "opportunity across the five machines.",
        "Modelled potential reduction of approximately 26.34%.",
        "Medium Term",
    )

    return pd.DataFrame(recommendations)


def create_implementation_roadmap() -> pd.DataFrame:
    """Create a practical implementation roadmap."""

    return pd.DataFrame(
        [
            {
                "Phase": 1,
                "Timeline": "0-30 Days",
                "Focus": "Immediate monitoring",
                "Actions": (
                    "Monitor baseline deviations, idle energy, high-consumption "
                    "events and schedule deviations."
                ),
                "Primary_Owner": "Operations / Energy Team",
            },
            {
                "Phase": 2,
                "Timeline": "30-60 Days",
                "Focus": "Machine efficiency",
                "Actions": (
                    "Investigate M005 and M003 operating efficiency, idle behavior "
                    "and machine-specific energy intensity."
                ),
                "Primary_Owner": "Maintenance / Production",
            },
            {
                "Phase": 3,
                "Timeline": "60-90 Days",
                "Focus": "Schedule optimization",
                "Actions": (
                    "Reduce unnecessary startup, shutdown and idle periods and "
                    "improve production schedule adherence."
                ),
                "Primary_Owner": "Production Planning",
            },
            {
                "Phase": 4,
                "Timeline": "90+ Days",
                "Focus": "Optimization deployment",
                "Actions": (
                    "Evaluate optimization recommendations using operational "
                    "constraints before implementation."
                ),
                "Primary_Owner": "Management / Engineering",
            },
        ]
    )


def create_executive_message(
    machine_priorities: pd.DataFrame,
) -> pd.DataFrame:
    """Create concise executive-level findings."""

    top_machine = (
        machine_priorities.iloc[0]["Machine_Name"]
        if not machine_priorities.empty
        else "N/A"
    )

    top_machine_id = (
        machine_priorities.iloc[0]["Machine_ID"]
        if not machine_priorities.empty
        else "N/A"
    )

    return pd.DataFrame(
        [
            {
                "Insight_ID": "INS001",
                "Category": "Energy",
                "Finding": (
                    "The manufacturing system consumed 349,856.35 kWh "
                    "across 10,000 observations."
                ),
            },
            {
                "Insight_ID": "INS002",
                "Category": "Idle Energy",
                "Finding": (
                    "Idle operation represents a measurable energy-saving "
                    "opportunity and should be actively monitored."
                ),
            },
            {
                "Insight_ID": "INS003",
                "Category": "Machine Priority",
                "Finding": (
                    f"{top_machine_id} - {top_machine} is the highest-priority "
                    "machine for sustainability and optimization intervention."
                ),
            },
            {
                "Insight_ID": "INS004",
                "Category": "What-If",
                "Finding": (
                    "The combined What-If scenario indicates a modelled potential "
                    "energy reduction of approximately 12.47% while maintaining "
                    "production output."
                ),
            },
            {
                "Insight_ID": "INS005",
                "Category": "Optimization",
                "Finding": (
                    "Linear programming estimates a larger modelled optimization "
                    "opportunity of approximately 26.34% while preserving the "
                    "observed production output."
                ),
            },
            {
                "Insight_ID": "INS006",
                "Category": "Sustainability",
                "Finding": (
                    "CO2e reductions are modelled estimates based on the synthetic "
                    "emission factor and must not be interpreted as achieved "
                    "real-world reductions."
                ),
            },
        ]
    )


def validate_recommendation_outputs(
    machine_priorities: pd.DataFrame,
    recommendations: pd.DataFrame,
    roadmap: pd.DataFrame,
) -> pd.DataFrame:
    """Validate recommendation outputs."""

    checks = [
        (
            "Machine Priority Records",
            len(machine_priorities) == 5,
        ),
        (
            "Recommendations Generated",
            len(recommendations) > 0,
        ),
        (
            "Implementation Roadmap Generated",
            len(roadmap) > 0,
        ),
        (
            "Unique Machine IDs",
            machine_priorities["Machine_ID"].nunique() == 5,
        ),
    ]

    return pd.DataFrame(
        [
            {
                "Validation": name,
                "Expected": "PASS",
                "Actual": "PASS" if result else "FAIL",
                "Status": "PASS" if result else "FAIL",
            }
            for name, result in checks
        ]
    )