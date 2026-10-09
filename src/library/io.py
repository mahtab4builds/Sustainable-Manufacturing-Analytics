"""
Reusable data input/output utilities.
"""

from pathlib import Path

import pandas as pd

from .logger import get_logger


logger = get_logger(__name__)


def ensure_directory(path):
    """
    Create a directory if it does not already exist.
    """

    directory = Path(path)

    directory.mkdir(
        parents=True,
        exist_ok=True
    )

    return directory


def load_excel(file_path, sheet_name=0):
    """
    Load an Excel worksheet into a pandas DataFrame.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Excel file not found: {file_path}"
        )

    logger.info(
        "Loading Excel data: %s",
        file_path
    )

    dataframe = pd.read_excel(
        file_path,
        sheet_name=sheet_name
    )

    logger.info(
        "Excel data loaded: %d rows × %d columns",
        dataframe.shape[0],
        dataframe.shape[1]
    )

    return dataframe


def load_csv(file_path):
    """
    Load a CSV file into a pandas DataFrame.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {file_path}"
        )

    logger.info(
        "Loading CSV data: %s",
        file_path
    )

    dataframe = pd.read_csv(file_path)

    logger.info(
        "CSV data loaded: %d rows × %d columns",
        dataframe.shape[0],
        dataframe.shape[1]
    )

    return dataframe


def save_csv(dataframe, file_path, index=False):
    """
    Save a DataFrame to CSV.
    """

    file_path = Path(file_path)

    ensure_directory(file_path.parent)

    dataframe.to_csv(
        file_path,
        index=index
    )

    logger.info(
        "CSV saved: %s",
        file_path
    )


def save_dataframe(dataframe, file_path, index=False):
    """
    Generic DataFrame saving utility.

    Automatically selects CSV or Excel based on file extension.
    """

    file_path = Path(file_path)

    ensure_directory(file_path.parent)

    extension = file_path.suffix.lower()

    if extension == ".csv":

        dataframe.to_csv(
            file_path,
            index=index
        )

    elif extension in {".xlsx", ".xls"}:

        dataframe.to_excel(
            file_path,
            index=index
        )

    else:

        raise ValueError(
            f"Unsupported output format: {extension}"
        )

    logger.info(
        "DataFrame saved: %s",
        file_path
    )