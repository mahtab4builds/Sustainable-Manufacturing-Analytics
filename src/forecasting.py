"""
SUSTAINABLE MANUFACTURING
ENERGY FORECASTING MODULE

Purpose
-------
Forecast future manufacturing energy demand using historical
energy-consumption data.

Workflow
--------
Historical Data
    ↓
Time-Series Preparation
    ↓
Daily Aggregation
    ↓
Chronological Train/Test Split
    ↓
Naive Persistence
    ↓
Rolling Average
    ↓
Random Forest Lag Model
    ↓
Model Validation
    ↓
Best Model Selection
    ↓
14-Day Future Forecast
    ↓
Validation & Reporting
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from dotenv import load_dotenv

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================================
# PROJECT PATH SETUP
# ============================================================================

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ============================================================================
# ENVIRONMENT CONFIGURATION
# ============================================================================

ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(ENV_FILE)


def get_env_path(
    keys: list[str],
    default: Path,
) -> Path:
    """
    Retrieve a project path from environment variables.

    Multiple environment-variable names are supported so that the
    forecasting module remains compatible with different project
    configuration conventions.
    """

    for key in keys:
        value = os.getenv(key)

        if value:
            path = Path(value)

            if not path.is_absolute():
                path = PROJECT_ROOT / path

            return path

    return default


INPUT_DATA = get_env_path(
    [
        "BASELINE_DATASET",
        "BASELINE_ENRICHED_DATASET",
        "CLEANED_DATASET",
    ],
    PROJECT_ROOT
    / "outputs"
    / "baseline"
    / "baseline_enriched_dataset.csv",
)


OUTPUT_DIR = get_env_path(
    [
        "FORECASTING_OUTPUT_DIR",
        "OUTPUT_DIR",
    ],
    PROJECT_ROOT / "outputs",
) / "forecasting"


CHART_DIR = OUTPUT_DIR / "charts"
MODEL_DIR = OUTPUT_DIR / "models"
FUTURE_DIR = OUTPUT_DIR / "future"


# ============================================================================
# LOGGING
# ============================================================================

LOGGER = logging.getLogger("forecasting")

if not LOGGER.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


# ============================================================================
# CONSTANTS
# ============================================================================

TIMESTAMP_COLUMN = "Timestamp"
ENERGY_COLUMN = "Energy_Consumption_kWh"

FORECAST_HORIZON_DAYS = 14
TEST_RATIO = 0.20

ROLLING_WINDOW = 7

RANDOM_STATE = 42

MIN_REQUIRED_DAILY_RECORDS = 30


# ============================================================================
# DIRECTORY MANAGEMENT
# ============================================================================

def initialize_directories() -> None:
    """
    Create all directories required by the forecasting pipeline.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CHART_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FUTURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOGGER.info(
        "Forecasting directories initialized."
    )


# ============================================================================
# DATA LOADING
# ============================================================================

def load_dataset() -> pd.DataFrame:
    """
    Load the baseline-enriched analytical dataset.

    Returns
    -------
    pandas.DataFrame
        Prepared manufacturing energy dataset.
    """

    if not INPUT_DATA.exists():

        raise FileNotFoundError(
            f"Forecasting input dataset not found:\n"
            f"{INPUT_DATA}"
        )

    LOGGER.info(
        "Loading baseline-enriched dataset..."
    )

    LOGGER.info(
        "Source: %s",
        INPUT_DATA,
    )

    df = pd.read_csv(
        INPUT_DATA,
        low_memory=False,
    )

    LOGGER.info(
        "Dataset loaded: %s rows × %s columns",
        f"{len(df):,}",
        len(df.columns),
    )

    required_columns = {
        TIMESTAMP_COLUMN,
        ENERGY_COLUMN,
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:

        raise ValueError(
            "Required forecasting columns are missing: "
            + ", ".join(sorted(missing_columns))
        )

    df[TIMESTAMP_COLUMN] = pd.to_datetime(
        df[TIMESTAMP_COLUMN],
        errors="coerce",
    )

    df[ENERGY_COLUMN] = pd.to_numeric(
        df[ENERGY_COLUMN],
        errors="coerce",
    )

    invalid_timestamps = (
        df[TIMESTAMP_COLUMN].isna().sum()
    )

    invalid_energy = (
        df[ENERGY_COLUMN].isna().sum()
    )

    if invalid_timestamps > 0:

        raise ValueError(
            f"Invalid timestamps detected: "
            f"{invalid_timestamps}"
        )

    if invalid_energy > 0:

        raise ValueError(
            f"Missing/invalid energy observations: "
            f"{invalid_energy}"
        )

    negative_energy = (
        df[ENERGY_COLUMN] < 0
    ).sum()

    if negative_energy > 0:

        raise ValueError(
            f"Negative energy observations detected: "
            f"{negative_energy}"
        )

    return df.sort_values(
        TIMESTAMP_COLUMN
    ).reset_index(drop=True)


# ============================================================================
# TIME-SERIES PREPARATION
# ============================================================================

def create_daily_energy_series(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate hourly machine-level energy observations
    into a daily factory-level energy time series.
    """

    LOGGER.info(
        "Creating daily energy time series..."
    )

    daily = (
        df.set_index(TIMESTAMP_COLUMN)
        [ENERGY_COLUMN]
        .resample("D")
        .sum()
        .reset_index()
    )

    daily.columns = [
        "Timestamp",
        "Energy_kWh",
    ]

    daily = daily.sort_values(
        "Timestamp"
    ).reset_index(drop=True)

    LOGGER.info(
        "Daily energy series generated: %s observations.",
        f"{len(daily):,}",
    )

    if len(daily) < MIN_REQUIRED_DAILY_RECORDS:

        raise ValueError(
            "Insufficient daily observations for forecasting. "
            f"At least {MIN_REQUIRED_DAILY_RECORDS} days are required."
        )

    return daily


# ============================================================================
# TRAIN / TEST SPLIT
# ============================================================================

def chronological_train_test_split(
    daily: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create a chronological train/test split.

    Time-series data must never be randomly shuffled because that
    would introduce future information into model training.
    """

    split_index = int(
        len(daily) * (1 - TEST_RATIO)
    )

    train = daily.iloc[
        :split_index
    ].copy()

    test = daily.iloc[
        split_index:
    ].copy()

    LOGGER.info(
        "Creating chronological train/test split..."
    )

    LOGGER.info(
        "Training observations: %s",
        len(train),
    )

    LOGGER.info(
        "Testing observations : %s",
        len(test),
    )

    LOGGER.info(
        "Training period: %s → %s",
        train["Timestamp"].min(),
        train["Timestamp"].max(),
    )

    LOGGER.info(
        "Testing period : %s → %s",
        test["Timestamp"].min(),
        test["Timestamp"].max(),
    )

    return train, test


# ============================================================================
# METRICS
# ============================================================================

def calculate_metrics(
    actual: pd.Series,
    predicted: pd.Series,
) -> Dict[str, float]:
    """
    Calculate standard forecasting evaluation metrics.
    """

    actual_values = np.asarray(
        actual,
        dtype=float,
    )

    predicted_values = np.asarray(
        predicted,
        dtype=float,
    )

    mae = mean_absolute_error(
        actual_values,
        predicted_values,
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual_values,
            predicted_values,
        )
    )

    non_zero_mask = (
        actual_values != 0
    )

    if non_zero_mask.any():

        mape = np.mean(
            np.abs(
                (
                    actual_values[non_zero_mask]
                    - predicted_values[non_zero_mask]
                )
                / actual_values[non_zero_mask]
            )
        ) * 100

    else:

        mape = np.nan

    return {
        "MAE_kWh": float(mae),
        "RMSE_kWh": float(rmse),
        "MAPE_Percent": float(mape),
    }


# ============================================================================
# NAIVE PERSISTENCE FORECAST
# ============================================================================

def naive_persistence_forecast(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> pd.Series:
    """
    Predict each future day using the most recently observed
    energy value.
    """

    LOGGER.info(
        "Running naive persistence forecast..."
    )

    last_value = train[
        "Energy_kWh"
    ].iloc[-1]

    return pd.Series(
        last_value,
        index=test.index,
        dtype=float,
    )


# ============================================================================
# ROLLING AVERAGE FORECAST
# ============================================================================

def rolling_average_forecast(
    train: pd.DataFrame,
    test: pd.DataFrame,
    window: int = ROLLING_WINDOW,
) -> pd.Series:
    """
    Forecast using the mean energy consumption of the most
    recent observations.
    """

    LOGGER.info(
        "Running %s-day rolling-average forecast...",
        window,
    )

    recent_mean = (
        train["Energy_kWh"]
        .tail(window)
        .mean()
    )

    return pd.Series(
        recent_mean,
        index=test.index,
        dtype=float,
    )


# ============================================================================
# RANDOM FOREST FEATURE ENGINEERING
# ============================================================================

def create_lag_features(
    series: pd.Series,
) -> pd.DataFrame:
    """
    Create lag-based features for machine-learning forecasting.
    """

    data = pd.DataFrame(
        {
            "Energy_kWh": series.values,
        }
    )

    data["Lag_1"] = (
        data["Energy_kWh"]
        .shift(1)
    )

    data["Lag_2"] = (
        data["Energy_kWh"]
        .shift(2)
    )

    data["Lag_3"] = (
        data["Energy_kWh"]
        .shift(3)
    )

    data["Lag_7"] = (
        data["Energy_kWh"]
        .shift(7)
    )

    data["Rolling_7"] = (
        data["Energy_kWh"]
        .shift(1)
        .rolling(7)
        .mean()
    )

    data["DayOfWeek"] = (
        np.arange(len(data)) % 7
    )

    return data


# ============================================================================
# RANDOM FOREST FORECAST
# ============================================================================

def random_forest_forecast(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> pd.Series:
    """
    Train a Random Forest model using historical lag features.
    """

    LOGGER.info(
        "Training Random Forest forecasting model..."
    )

    combined = pd.concat(
        [
            train,
            test,
        ],
        ignore_index=True,
    )

    feature_data = create_lag_features(
        combined["Energy_kWh"]
    )

    train_length = len(train)

    train_features = feature_data.iloc[
        :train_length
    ].copy()

    test_features = feature_data.iloc[
        train_length:
    ].copy()

    feature_columns = [
        "Lag_1",
        "Lag_2",
        "Lag_3",
        "Lag_7",
        "Rolling_7",
        "DayOfWeek",
    ]

    valid_train = train_features.dropna()

    if valid_train.empty:

        raise ValueError(
            "Insufficient observations for Random Forest lag model."
        )

    X_train = valid_train[
        feature_columns
    ]

    y_train = valid_train[
        "Energy_kWh"
    ]

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    predictions = []

    history = list(
        train["Energy_kWh"]
    )

    for _ in range(len(test)):

        lag_1 = history[-1]
        lag_2 = history[-2]
        lag_3 = history[-3]
        lag_7 = history[-7]

        rolling_7 = np.mean(
            history[-7:]
        )

        day_of_week = (
            len(history) % 7
        )

        X_future = pd.DataFrame(
            [
                [
                    lag_1,
                    lag_2,
                    lag_3,
                    lag_7,
                    rolling_7,
                    day_of_week,
                ]
            ],
            columns=feature_columns,
        )

        prediction = model.predict(
            X_future
        )[0]

        prediction = max(
            float(prediction),
            0.0,
        )

        predictions.append(
            prediction
        )

        history.append(
            prediction
        )

    return pd.Series(
        predictions,
        index=test.index,
        dtype=float,
    )


# ============================================================================
# MODEL COMPARISON
# ============================================================================

def evaluate_models(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate all forecasting models and identify the best
    validation model based on MAE.
    """

    LOGGER.info(
        "Evaluating forecasting models..."
    )

    results = []

    predictions = {}

    naive = naive_persistence_forecast(
        train,
        test,
    )

    predictions[
        "Naive Persistence"
    ] = naive

    rolling = rolling_average_forecast(
        train,
        test,
    )

    predictions[
        "7-Day Rolling Average"
    ] = rolling

    random_forest = random_forest_forecast(
        train,
        test,
    )

    predictions[
        "Random Forest Lag Model"
    ] = random_forest

    for model_name, prediction in predictions.items():

        metrics = calculate_metrics(
            test["Energy_kWh"],
            prediction,
        )

        results.append(
            {
                "Model": model_name,
                **metrics,
            }
        )

    comparison = pd.DataFrame(
        results
    )

    comparison = comparison.sort_values(
        by="MAE_kWh"
    ).reset_index(drop=True)

    comparison["Rank"] = (
        comparison.index + 1
    )

    best_model = comparison.iloc[0][
        "Model"
    ]

    LOGGER.info(
        "Model comparison completed."
    )

    LOGGER.info(
        "\n%s",
        comparison.to_string(
            index=False
        ),
    )

    LOGGER.info(
        "Best validation model: %s",
        best_model,
    )

    prediction_output = test[
        ["Timestamp", "Energy_kWh"]
    ].copy()

    prediction_output[
        "Actual_Energy_kWh"
    ] = prediction_output[
        "Energy_kWh"
    ]

    prediction_output = prediction_output.drop(
        columns=["Energy_kWh"]
    )

    for model_name, prediction in predictions.items():

        safe_name = (
            model_name
            .replace(" ", "_")
            .replace("-", "")
        )

        prediction_output[
            f"{safe_name}_Forecast_kWh"
        ] = prediction.values

    return comparison, prediction_output


# ============================================================================
# FUTURE FORECAST
# ============================================================================

def generate_future_forecast(
    daily: pd.DataFrame,
    best_model: str,
) -> pd.DataFrame:
    """
    Generate a future 14-day energy forecast.

    The selected validation model is reused for future forecasting.
    """

    LOGGER.info(
        "Generating %s-day future energy forecast...",
        FORECAST_HORIZON_DAYS,
    )

    last_timestamp = (
        daily["Timestamp"].max()
    )

    future_dates = pd.date_range(
        start=last_timestamp + pd.Timedelta(days=1),
        periods=FORECAST_HORIZON_DAYS,
        freq="D",
    )

    history = list(
        daily["Energy_kWh"]
    )

    forecasts = []

    if best_model == "Naive Persistence":

        last_value = history[-1]

        forecasts = [
            float(last_value)
            for _ in range(
                FORECAST_HORIZON_DAYS
            )
        ]

    elif best_model == "7-Day Rolling Average":

        for _ in range(
            FORECAST_HORIZON_DAYS
        ):

            prediction = np.mean(
                history[-ROLLING_WINDOW:]
            )

            prediction = max(
                float(prediction),
                0.0,
            )

            forecasts.append(
                prediction
            )

            history.append(
                prediction
            )

    elif best_model == "Random Forest Lag Model":

        combined = pd.DataFrame(
            {
                "Timestamp": daily[
                    "Timestamp"
                ],
                "Energy_kWh": daily[
                    "Energy_kWh"
                ],
            }
        )

        model_data = create_lag_features(
            combined["Energy_kWh"]
        )

        feature_columns = [
            "Lag_1",
            "Lag_2",
            "Lag_3",
            "Lag_7",
            "Rolling_7",
            "DayOfWeek",
        ]

        valid_model_data = model_data.dropna()

        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        model.fit(
            valid_model_data[
                feature_columns
            ],
            valid_model_data[
                "Energy_kWh"
            ],
        )

        for _ in range(
            FORECAST_HORIZON_DAYS
        ):

            X_future = pd.DataFrame(
                [
                    [
                        history[-1],
                        history[-2],
                        history[-3],
                        history[-7],
                        np.mean(history[-7:]),
                        len(history) % 7,
                    ]
                ],
                columns=feature_columns,
            )

            prediction = model.predict(
                X_future
            )[0]

            prediction = max(
                float(prediction),
                0.0,
            )

            forecasts.append(
                prediction
            )

            history.append(
                prediction
            )

    else:

        raise ValueError(
            f"Unsupported forecasting model: "
            f"{best_model}"
        )

    future = pd.DataFrame(
        {
            "Forecast_Date": future_dates,
            "Forecast_Energy_kWh": forecasts,
        }
    )

    future[
        "Forecast_Day"
    ] = np.arange(
        1,
        FORECAST_HORIZON_DAYS + 1,
    )

    future[
        "Forecast_CO2e_kg"
    ] = (
        future[
            "Forecast_Energy_kWh"
        ]
        * float(
            os.getenv(
                "SYNTHETIC_EMISSION_FACTOR",
                "0.70",
            )
        )
    )

    LOGGER.info(
        "Future forecast generated successfully."
    )

    return future


# ============================================================================
# CHARTS
# ============================================================================

def save_historical_energy_chart(
    daily: pd.DataFrame,
) -> None:
    """
    Save historical daily energy-consumption chart.
    """

    LOGGER.info(
        "Generating historical energy chart..."
    )

    figure, axis = plt.subplots(
        figsize=(12, 6)
    )

    axis.plot(
        daily["Timestamp"],
        daily["Energy_kWh"],
        linewidth=1.5,
    )

    axis.set_title(
        "Historical Daily Energy Consumption"
    )

    axis.set_xlabel(
        "Date"
    )

    axis.set_ylabel(
        "Energy Consumption (kWh)"
    )

    axis.grid(
        alpha=0.3
    )

    figure.tight_layout()

    output = (
        CHART_DIR
        / "01_historical_daily_energy.png"
    )

    figure.savefig(
        output,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    LOGGER.info(
        "Chart saved: %s",
        output.name,
    )


def save_validation_chart(
    validation: pd.DataFrame,
    best_model: str,
) -> None:
    """
    Save actual versus forecast validation chart.
    """

    LOGGER.info(
        "Generating model validation chart..."
    )

    forecast_column = (
        best_model
        .replace(" ", "_")
        .replace("-", "")
        + "_Forecast_kWh"
    )

    figure, axis = plt.subplots(
        figsize=(12, 6)
    )

    axis.plot(
        validation["Timestamp"],
        validation["Actual_Energy_kWh"],
        label="Actual Energy",
        linewidth=2,
    )

    axis.plot(
        validation["Timestamp"],
        validation[forecast_column],
        label=f"{best_model} Forecast",
        linewidth=2,
        linestyle="--",
    )

    axis.set_title(
        "Forecast Model Validation"
    )

    axis.set_xlabel(
        "Date"
    )

    axis.set_ylabel(
        "Energy Consumption (kWh)"
    )

    axis.legend()

    axis.grid(
        alpha=0.3
    )

    figure.tight_layout()

    output = (
        CHART_DIR
        / "02_forecast_validation.png"
    )

    figure.savefig(
        output,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    LOGGER.info(
        "Chart saved: %s",
        output.name,
    )


def save_future_forecast_chart(
    daily: pd.DataFrame,
    future: pd.DataFrame,
) -> None:
    """
    Save historical and future forecast visualization.
    """

    LOGGER.info(
        "Generating future forecast chart..."
    )

    figure, axis = plt.subplots(
        figsize=(12, 6)
    )

    recent_history = daily.tail(30)

    axis.plot(
        recent_history["Timestamp"],
        recent_history["Energy_kWh"],
        label="Historical Energy",
        linewidth=2,
    )

    axis.plot(
        future["Forecast_Date"],
        future["Forecast_Energy_kWh"],
        label="Future Forecast",
        linewidth=2,
        linestyle="--",
    )

    axis.set_title(
        "14-Day Future Energy Forecast"
    )

    axis.set_xlabel(
        "Date"
    )

    axis.set_ylabel(
        "Energy Consumption (kWh)"
    )

    axis.legend()

    axis.grid(
        alpha=0.3
    )

    figure.tight_layout()

    output = (
        CHART_DIR
        / "03_future_energy_forecast.png"
    )

    figure.savefig(
        output,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    LOGGER.info(
        "Chart saved: %s",
        output.name,
    )


# ============================================================================
# VALIDATION
# ============================================================================

def validate_forecasting_outputs(
    daily: pd.DataFrame,
    comparison: pd.DataFrame,
    future: pd.DataFrame,
) -> pd.DataFrame:
    """
    Perform final forecasting-output validation.
    """

    validation = []

    validation.append(
        {
            "Validation": "Daily Series Exists",
            "Expected": True,
            "Actual": len(daily) > 0,
        }
    )

    validation.append(
        {
            "Validation": "Model Comparison Exists",
            "Expected": True,
            "Actual": len(comparison) == 3,
        }
    )

    validation.append(
        {
            "Validation": "Forecast Horizon",
            "Expected": FORECAST_HORIZON_DAYS,
            "Actual": len(future),
        }
    )

    validation.append(
        {
            "Validation": "Negative Future Forecasts",
            "Expected": 0,
            "Actual": int(
                (
                    future[
                        "Forecast_Energy_kWh"
                    ]
                    < 0
                ).sum()
            ),
        }
    )

    validation.append(
        {
            "Validation": "Forecast Metrics Available",
            "Expected": True,
            "Actual": bool(
                comparison[
                    [
                        "MAE_kWh",
                        "RMSE_kWh",
                        "MAPE_Percent",
                    ]
                ]
                .notna()
                .all()
                .all()
            ),
        }
    )

    result = pd.DataFrame(
        validation
    )

    result["Status"] = np.where(
        result["Expected"]
        == result["Actual"],
        "PASS",
        "FAIL",
    )

    return result


# ============================================================================
# EXECUTIVE SUMMARY
# ============================================================================

def create_forecasting_summary(
    daily: pd.DataFrame,
    comparison: pd.DataFrame,
    future: pd.DataFrame,
    best_model: str,
) -> pd.DataFrame:
    """
    Create a concise executive summary for the forecasting phase.
    """

    best_row = comparison[
        comparison["Model"]
        == best_model
    ].iloc[0]

    summary = pd.DataFrame(
        {
            "Metric": [
                "Daily Observations",
                "Forecast Horizon (Days)",
                "Historical Energy (kWh)",
                "Average Daily Energy (kWh)",
                "Best Validation Model",
                "Best Model MAE (kWh)",
                "Best Model RMSE (kWh)",
                "Best Model MAPE (%)",
                "Forecasted Energy (kWh)",
                "Forecasted CO2e (kg)",
            ],
            "Value": [
                len(daily),
                FORECAST_HORIZON_DAYS,
                daily["Energy_kWh"].sum(),
                daily["Energy_kWh"].mean(),
                best_model,
                best_row["MAE_kWh"],
                best_row["RMSE_kWh"],
                best_row["MAPE_Percent"],
                future[
                    "Forecast_Energy_kWh"
                ].sum(),
                future[
                    "Forecast_CO2e_kg"
                ].sum(),
            ],
        }
    )

    return summary


# ============================================================================
# SAVE OUTPUTS
# ============================================================================

def save_outputs(
    daily: pd.DataFrame,
    comparison: pd.DataFrame,
    validation_predictions: pd.DataFrame,
    future: pd.DataFrame,
    validation: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    """
    Save all forecasting analytical outputs.
    """

    daily.to_csv(
        OUTPUT_DIR
        / "daily_energy_timeseries.csv",
        index=False,
    )

    comparison.to_csv(
        OUTPUT_DIR
        / "forecast_model_comparison.csv",
        index=False,
    )

    validation_predictions.to_csv(
        OUTPUT_DIR
        / "forecast_validation.csv",
        index=False,
    )

    future.to_csv(
        FUTURE_DIR
        / "future_energy_forecast.csv",
        index=False,
    )

    validation.to_csv(
        OUTPUT_DIR
        / "forecasting_validation.csv",
        index=False,
    )

    summary.to_csv(
        OUTPUT_DIR
        / "forecasting_summary.csv",
        index=False,
    )

    LOGGER.info(
        "Forecasting analytical outputs saved successfully."
    )


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main() -> None:
    """
    Execute the complete energy forecasting pipeline.
    """

    LOGGER.info(
        "=" * 72
    )

    LOGGER.info(
        "SUSTAINABLE MANUFACTURING - ENERGY FORECASTING"
    )

    LOGGER.info(
        "=" * 72
    )

    try:

        initialize_directories()

        df = load_dataset()

        daily = create_daily_energy_series(
            df
        )

        train, test = (
            chronological_train_test_split(
                daily
            )
        )

        (
            comparison,
            validation_predictions,
        ) = evaluate_models(
            train,
            test,
        )

        best_model = comparison.iloc[0][
            "Model"
        ]

        future = generate_future_forecast(
            daily,
            best_model,
        )

        save_historical_energy_chart(
            daily
        )

        save_validation_chart(
            validation_predictions,
            best_model,
        )

        save_future_forecast_chart(
            daily,
            future,
        )

        validation = (
            validate_forecasting_outputs(
                daily,
                comparison,
                future,
            )
        )

        if (
            validation["Status"]
            == "FAIL"
        ).any():

            raise RuntimeError(
                "Forecasting validation failed."
            )

        summary = create_forecasting_summary(
            daily,
            comparison,
            future,
            best_model,
        )

        save_outputs(
            daily,
            comparison,
            validation_predictions,
            future,
            validation,
            summary,
        )

        LOGGER.info(
            "=" * 72
        )

        LOGGER.info(
            "ENERGY FORECASTING COMPLETED SUCCESSFULLY"
        )

        LOGGER.info(
            "=" * 72
        )

        LOGGER.info(
            "Daily observations      : %s",
            len(daily),
        )

        LOGGER.info(
            "Training observations   : %s",
            len(train),
        )

        LOGGER.info(
            "Testing observations    : %s",
            len(test),
        )

        LOGGER.info(
            "Best validation model   : %s",
            best_model,
        )

        LOGGER.info(
            "Forecast horizon        : %s days",
            FORECAST_HORIZON_DAYS,
        )

        LOGGER.info(
            "Forecast output         : %s",
            FUTURE_DIR
            / "future_energy_forecast.csv",
        )

        LOGGER.info(
            "Forecasting output      : %s",
            OUTPUT_DIR,
        )

        print(
            "\n"
            + "=" * 72
        )

        print(
            "FORECASTING EXECUTIVE SUMMARY"
        )

        print(
            "=" * 72
        )

        print(
            summary.to_string(
                index=False
            )
        )

        print(
            "\n"
            + "=" * 72
        )

        print(
            "FORECASTING VALIDATION"
        )

        print(
            "=" * 72
        )

        print(
            validation.to_string(
                index=False
            )
        )

        print(
            "\nForecasting completed successfully."
        )

        print(
            "=" * 72
        )

    except Exception as exc:

        LOGGER.exception(
            "Unexpected error during forecasting: %s",
            exc,
        )

        raise


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()