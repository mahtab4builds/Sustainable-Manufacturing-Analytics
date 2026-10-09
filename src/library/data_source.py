# ============================================================
# SUSTAINABLE MANUFACTURING ANALYTICS
# DATA SOURCE LIBRARY
# ============================================================

from pathlib import Path
import json
import logging
import time
import os

import pandas as pd
import requests
from dotenv import load_dotenv


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(ENV_FILE)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_SOURCE_TYPE = os.getenv(
    "DATA_SOURCE_TYPE",
    "auto"
).strip().lower()

DATA_FILE = os.getenv(
    "DATA_FILE",
    ""
).strip()

DATA_DIR = os.getenv(
    "DATA_DIR",
    "data"
).strip()

DATA_FILE_PATTERN = os.getenv(
    "DATA_FILE_PATTERN",
    "*.xlsx"
).strip()

API_URL = os.getenv(
    "API_URL",
    ""
).strip()

API_TIMEOUT = int(
    os.getenv(
        "API_TIMEOUT",
        "30"
    )
)

API_METHOD = os.getenv(
    "API_METHOD",
    "GET"
).strip().upper()

API_HEADERS = os.getenv(
    "API_HEADERS",
    ""
).strip()

API_DATA_KEY = os.getenv(
    "API_DATA_KEY",
    ""
).strip()

API_RETRY_COUNT = int(
    os.getenv(
        "API_RETRY_COUNT",
        "3"
    )
)


# ============================================================
# MANUFACTURING DATA SCHEMA
# ============================================================

REQUIRED_COLUMNS = {
    "Record_ID",
    "Timestamp",
    "Machine_ID",
    "Actual_State",
    "Energy_Consumption_kWh",
    "Actual_Production_Units"
}


# ============================================================
# LOGGING
# ============================================================

LOG_LEVEL = os.getenv(
    "LOG_LEVEL",
    "INFO"
).upper()

logging.basicConfig(
    level=getattr(
        logging,
        LOG_LEVEL,
        logging.INFO
    ),
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================
# PATH RESOLUTION
# ============================================================

def resolve_path(path_value):
    """
    Convert a relative path into an absolute project path.
    """

    if not path_value:
        return None

    path = Path(path_value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


# ============================================================
# SCHEMA FUNCTIONS
# ============================================================

def get_missing_required_columns(df):
    """
    Return required columns missing from the DataFrame.
    """

    actual_columns = {
        str(column).strip()
        for column in df.columns
    }

    return REQUIRED_COLUMNS - actual_columns


def is_manufacturing_dataset(df):
    """
    Determine whether a DataFrame contains the
    minimum required manufacturing-energy schema.
    """

    missing_columns = get_missing_required_columns(
        df
    )

    return len(missing_columns) == 0


# ============================================================
# EXCEL SHEET DISCOVERY
# ============================================================

def find_manufacturing_sheet(file_path):
    """
    Automatically identify the worksheet containing
    the manufacturing-energy dataset.

    The function checks the schema of each worksheet.
    """

    logger.info(
        "Inspecting Excel workbook sheets: %s",
        file_path
    )

    try:

        excel_file = pd.ExcelFile(
            file_path
        )

    except Exception as exc:

        raise RuntimeError(
            f"Unable to open Excel workbook:\n"
            f"{file_path}\n"
            f"Reason: {exc}"
        ) from exc

    matching_sheets = []

    for sheet_name in excel_file.sheet_names:

        try:

            preview = pd.read_excel(
                file_path,
                sheet_name=sheet_name,
                nrows=5
            )

        except Exception as exc:

            logger.warning(
                "Unable to inspect sheet '%s': %s",
                sheet_name,
                exc
            )

            continue

        if is_manufacturing_dataset(
            preview
        ):

            matching_sheets.append(
                sheet_name
            )

    if not matching_sheets:

        raise ValueError(
            "No worksheet containing the required "
            "manufacturing-energy schema was found.\n"
            f"Workbook: {file_path}\n"
            f"Required columns: "
            f"{sorted(REQUIRED_COLUMNS)}"
        )

    if len(matching_sheets) > 1:

        logger.warning(
            "Multiple worksheets match the manufacturing "
            "schema. Using: %s",
            matching_sheets[0]
        )

    selected_sheet = matching_sheets[0]

    logger.info(
        "Manufacturing data worksheet detected: %s",
        selected_sheet
    )

    return selected_sheet


# ============================================================
# LOCAL FILE DISCOVERY
# ============================================================

def discover_local_file():
    """
    Find a compatible local dataset.

    Priority:

    1. Explicit DATA_FILE
    2. Automatic discovery from DATA_DIR
    """

    # --------------------------------------------------------
    # Explicit DATA_FILE
    # --------------------------------------------------------

    if DATA_FILE:

        file_path = resolve_path(
            DATA_FILE
        )

        if not file_path.exists():

            raise FileNotFoundError(
                f"Configured DATA_FILE does not exist:\n"
                f"{file_path}"
            )

        if not file_path.is_file():

            raise ValueError(
                f"Configured DATA_FILE is not a file:\n"
                f"{file_path}"
            )

        logger.info(
            "Using configured data file: %s",
            file_path
        )

        return file_path

    # --------------------------------------------------------
    # Automatic discovery
    # --------------------------------------------------------

    data_directory = resolve_path(
        DATA_DIR
    )

    if data_directory is None:

        raise ValueError(
            "DATA_DIR is not configured."
        )

    if not data_directory.exists():

        raise FileNotFoundError(
            f"Data directory does not exist:\n"
            f"{data_directory}"
        )

    if not data_directory.is_dir():

        raise ValueError(
            f"DATA_DIR is not a directory:\n"
            f"{data_directory}"
        )

    files = list(
        data_directory.glob(
            DATA_FILE_PATTERN
        )
    )

    # --------------------------------------------------------
    # Fallback formats
    # --------------------------------------------------------

    if not files:

        for pattern in [
            "*.xlsx",
            "*.xls",
            "*.csv"
        ]:

            files.extend(
                data_directory.glob(
                    pattern
                )
            )

    files = list(
        set(files)
    )

    if not files:

        raise FileNotFoundError(
            "No compatible dataset file found.\n"
            f"Directory: {data_directory}\n"
            f"Pattern: {DATA_FILE_PATTERN}"
        )

    # --------------------------------------------------------
    # Validate candidate files
    # --------------------------------------------------------

    compatible_files = []

    for file_path in files:

        try:

            suffix = file_path.suffix.lower()

            # ------------------------------------------------
            # CSV
            # ------------------------------------------------

            if suffix == ".csv":

                preview = pd.read_csv(
                    file_path,
                    nrows=5
                )

                if is_manufacturing_dataset(
                    preview
                ):

                    compatible_files.append(
                        file_path
                    )

            # ------------------------------------------------
            # Excel
            # ------------------------------------------------

            elif suffix in {
                ".xlsx",
                ".xls"
            }:

                find_manufacturing_sheet(
                    file_path
                )

                compatible_files.append(
                    file_path
                )

        except Exception as exc:

            logger.warning(
                "Skipping incompatible file '%s': %s",
                file_path.name,
                exc
            )

    if not compatible_files:

        raise ValueError(
            "No local file contains the required "
            "manufacturing-energy schema."
        )

    # --------------------------------------------------------
    # Select newest compatible file
    # --------------------------------------------------------

    compatible_files.sort(
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )

    selected_file = compatible_files[0]

    if len(compatible_files) > 1:

        logger.warning(
            "Multiple compatible datasets found. "
            "Using most recently modified file: %s",
            selected_file
        )

    logger.info(
        "Automatically discovered data file: %s",
        selected_file
    )

    return selected_file


# ============================================================
# LOCAL DATA LOADING
# ============================================================

def load_local_data():
    """
    Load CSV or Excel data into a Pandas DataFrame.
    """

    file_path = discover_local_file()

    suffix = file_path.suffix.lower()

    logger.info(
        "Loading local dataset: %s",
        file_path
    )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    if suffix == ".csv":

        df = pd.read_csv(
            file_path
        )

    # --------------------------------------------------------
    # Excel
    # --------------------------------------------------------

    elif suffix in {
        ".xlsx",
        ".xls"
    }:

        sheet_name = find_manufacturing_sheet(
            file_path
        )

        logger.info(
            "Loading worksheet: %s",
            sheet_name
        )

        df = pd.read_excel(
            file_path,
            sheet_name=sheet_name
        )

    else:

        raise ValueError(
            f"Unsupported file format: {suffix}"
        )

    # --------------------------------------------------------
    # Empty dataset
    # --------------------------------------------------------

    if df.empty:

        raise ValueError(
            f"The selected dataset is empty:\n"
            f"{file_path}"
        )

    # --------------------------------------------------------
    # Final schema validation
    # --------------------------------------------------------

    missing_columns = (
        get_missing_required_columns(
            df
        )
    )

    if missing_columns:

        raise ValueError(
            "Dataset schema validation failed.\n"
            f"Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    logger.info(
        "Local dataset loaded successfully: "
        "%s rows × %s columns",
        len(df),
        len(df.columns)
    )

    return df


# ============================================================
# API HEADERS
# ============================================================

def parse_api_headers():
    """
    Convert API_HEADERS JSON into a dictionary.
    """

    if not API_HEADERS:
        return {}

    try:

        headers = json.loads(
            API_HEADERS
        )

    except json.JSONDecodeError as exc:

        raise ValueError(
            "API_HEADERS must contain valid JSON."
        ) from exc

    if not isinstance(
        headers,
        dict
    ):

        raise ValueError(
            "API_HEADERS must be a JSON object."
        )

    return headers


# ============================================================
# API RESPONSE NORMALIZATION
# ============================================================

def normalize_api_response(response):
    """
    Convert JSON or CSV API responses into
    a Pandas DataFrame.
    """

    content_type = response.headers.get(
        "Content-Type",
        ""
    ).lower()

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    if "json" in content_type:

        try:

            payload = response.json()

        except ValueError as exc:

            raise ValueError(
                "API returned invalid JSON."
            ) from exc

    else:

        # ----------------------------------------------------
        # Try JSON
        # ----------------------------------------------------

        try:

            payload = response.json()

        except ValueError:

            # ------------------------------------------------
            # Try CSV
            # ------------------------------------------------

            from io import StringIO

            try:

                return pd.read_csv(
                    StringIO(
                        response.text
                    )
                )

            except Exception as exc:

                raise ValueError(
                    "API response is neither valid JSON "
                    "nor CSV."
                ) from exc

    # --------------------------------------------------------
    # Nested API data
    # --------------------------------------------------------

    if API_DATA_KEY:

        if not isinstance(
            payload,
            dict
        ):

            raise ValueError(
                "API_DATA_KEY is configured, but API "
                "response is not a JSON object."
            )

        if API_DATA_KEY not in payload:

            raise ValueError(
                f"API_DATA_KEY '{API_DATA_KEY}' "
                "was not found in the API response."
            )

        payload = payload[
            API_DATA_KEY
        ]

    # --------------------------------------------------------
    # JSON list
    # --------------------------------------------------------

    if isinstance(
        payload,
        list
    ):

        return pd.json_normalize(
            payload
        )

    # --------------------------------------------------------
    # JSON object
    # --------------------------------------------------------

    if isinstance(
        payload,
        dict
    ):

        return pd.json_normalize(
            payload
        )

    raise ValueError(
        "Unsupported API response structure."
    )


# ============================================================
# API DATA LOADING
# ============================================================

def load_api_data():
    """
    Load manufacturing data from an API.

    Features:

    - GET/POST support
    - timeout
    - retry mechanism
    - JSON
    - CSV
    - nested JSON
    - schema validation
    """

    if not API_URL:

        raise ValueError(
            "API_URL is empty. "
            "Configure API_URL before using API mode."
        )

    headers = parse_api_headers()

    logger.info(
        "Connecting to API: %s",
        API_URL
    )

    last_error = None

    for attempt in range(
        1,
        API_RETRY_COUNT + 1
    ):

        try:

            response = requests.request(
                method=API_METHOD,
                url=API_URL,
                headers=headers,
                timeout=API_TIMEOUT
            )

            response.raise_for_status()

            df = normalize_api_response(
                response
            )

            if df.empty:

                raise ValueError(
                    "API returned an empty dataset."
                )

            # ------------------------------------------------
            # API schema validation
            # ------------------------------------------------

            missing_columns = (
                get_missing_required_columns(
                    df
                )
            )

            if missing_columns:

                raise ValueError(
                    "API schema validation failed.\n"
                    f"Missing required columns: "
                    f"{sorted(missing_columns)}"
                )

            logger.info(
                "API dataset loaded successfully: "
                "%s rows × %s columns",
                len(df),
                len(df.columns)
            )

            return df

        except Exception as exc:

            last_error = exc

            logger.warning(
                "API attempt %s/%s failed: %s",
                attempt,
                API_RETRY_COUNT,
                exc
            )

            if attempt < API_RETRY_COUNT:

                time.sleep(
                    min(
                        2 ** (attempt - 1),
                        5
                    )
                )

    raise RuntimeError(
        "Unable to load data from API after "
        f"{API_RETRY_COUNT} attempts. "
        f"Last error: {last_error}"
    )


# ============================================================
# CENTRAL DATA LOADER
# ============================================================

def load_data():
    """
    Central data-loading interface.

    Supported modes:

        auto
        local
        api

    Returns:

        pandas.DataFrame
    """

    logger.info(
        "Data source mode: %s",
        DATA_SOURCE_TYPE
    )

    # --------------------------------------------------------
    # LOCAL
    # --------------------------------------------------------

    if DATA_SOURCE_TYPE == "local":

        return load_local_data()

    # --------------------------------------------------------
    # API
    # --------------------------------------------------------

    if DATA_SOURCE_TYPE == "api":

        return load_api_data()

    # --------------------------------------------------------
    # AUTO
    # --------------------------------------------------------

    if DATA_SOURCE_TYPE == "auto":

        if API_URL:

            logger.info(
                "AUTO mode: API_URL detected. "
                "Using API source."
            )

            return load_api_data()

        logger.info(
            "AUTO mode: API_URL not configured. "
            "Using local data source."
        )

        return load_local_data()

    # --------------------------------------------------------
    # Invalid mode
    # --------------------------------------------------------

    raise ValueError(
        "Invalid DATA_SOURCE_TYPE.\n"
        "Allowed values: auto, local, api"
    )


# ============================================================
# MODULE TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 72)
    print("DATA SOURCE LIBRARY")
    print("=" * 72)

    print(
        f"PROJECT_ROOT      : {PROJECT_ROOT}"
    )

    print(
        f"DATA_SOURCE_TYPE  : {DATA_SOURCE_TYPE}"
    )

    print(
        f"DATA_FILE         : "
        f"{DATA_FILE or '[AUTO DISCOVERY]'}"
    )

    print(
        f"DATA_DIR          : {DATA_DIR}"
    )

    print(
        f"DATA_FILE_PATTERN : {DATA_FILE_PATTERN}"
    )

    print(
        f"API_URL           : "
        f"{API_URL or '[NOT CONFIGURED]'}"
    )

    print("=" * 72)

    try:

        df = load_data()

        print()
        print("DATA LOAD SUCCESSFUL")
        print("-" * 72)

        print(
            f"Rows              : {len(df)}"
        )

        print(
            f"Columns           : {len(df.columns)}"
        )

        print()
        print("Columns detected:")

        print(
            ", ".join(
                df.columns.astype(str)
            )
        )

        print("=" * 72)

    except Exception as exc:

        print()
        print("DATA LOAD FAILED")
        print("-" * 72)

        print(
            f"Reason: {exc}"
        )

        print("=" * 72)