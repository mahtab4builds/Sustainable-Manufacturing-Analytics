"""
Reusable analytical calculation functions.
"""

import numpy as np
import pandas as pd


def safe_divide(numerator, denominator):
    """
    Safely divide two values or pandas Series.

    Zero denominators are converted to NaN to prevent
    division-by-zero errors.
    """

    denominator = denominator.replace(
        0,
        np.nan
    ) if isinstance(denominator, pd.Series) else denominator

    return numerator / denominator


def calculate_percentage(value, total):
    """
    Calculate percentage safely.
    """

    if total == 0:
        return 0.0

    return (value / total) * 100


def calculate_energy_per_unit(
    energy,
    production
):
    """
    Calculate energy consumed per production unit.
    """

    if isinstance(production, pd.Series):

        production = production.replace(
            0,
            np.nan
        )

    elif production == 0:

        return np.nan

    return energy / production


def calculate_co2e(
    energy,
    emission_factor
):
    """
    Calculate CO2-equivalent emissions.
    """

    return energy * emission_factor


def calculate_deviation(
    actual,
    baseline
):
    """
    Calculate absolute deviation from baseline.
    """

    return actual - baseline


def calculate_deviation_percentage(
    actual,
    baseline
):
    """
    Calculate percentage deviation from baseline.
    """

    if isinstance(baseline, pd.Series):

        baseline = baseline.replace(
            0,
            np.nan
        )

    elif baseline == 0:

        return np.nan

    return (
        (actual - baseline)
        / baseline
    ) * 100