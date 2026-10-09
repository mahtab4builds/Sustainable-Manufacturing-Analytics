"""
Reusable validation utilities for analytical datasets.
"""

import pandas as pd


def validate_required_columns(
    dataframe,
    required_columns
):
    """
    Verify that all required columns exist.
    """

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    return True


def validate_record_count(
    dataframe,
    expected_count
):
    """
    Validate the expected number of records.
    """

    actual_count = len(dataframe)

    if actual_count != expected_count:

        raise ValueError(
            f"Record count mismatch. "
            f"Expected {expected_count}, "
            f"found {actual_count}."
        )

    return True


def validate_no_negative_values(
    dataframe,
    columns
):
    """
    Validate that specified numeric columns contain
    no negative values.
    """

    issues = {}

    for column in columns:

        if column not in dataframe.columns:
            continue

        negative_count = (
            dataframe[column] < 0
        ).sum()

        issues[column] = int(
            negative_count
        )

    invalid = {
        column: count
        for column, count in issues.items()
        if count > 0
    }

    if invalid:

        raise ValueError(
            f"Negative values detected: {invalid}"
        )

    return True


def validation_summary(
    dataframe,
    expected_records=None
):
    """
    Generate a compact validation summary.
    """

    summary = {
        "records": len(dataframe),
        "columns": len(dataframe.columns),
        "duplicate_rows": int(
            dataframe.duplicated().sum()
        ),
        "missing_values": int(
            dataframe.isna().sum().sum()
        ),
    }

    if expected_records is not None:

        summary["record_count_valid"] = (
            len(dataframe)
            == expected_records
        )

    return summary