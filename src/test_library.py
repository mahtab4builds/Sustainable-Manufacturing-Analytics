"""
Shared Library Test
Sustainable Manufacturing Analytics
"""

from library.config import (
    PROJECT_NAME,
    DATA_FILE,
    OUTPUT_DIR,
    EMISSION_FACTOR,
)

from library.constants import (
    VALID_STATES,
    MACHINE_IDS,
)

from library.logger import get_logger
from library.io import ensure_directory
from library.calculations import (
    calculate_percentage,
    calculate_co2e,
)


logger = get_logger(__name__)


def main():
    """Run shared-library validation tests."""

    logger.info("=" * 72)
    logger.info("SHARED LIBRARY TEST")
    logger.info("=" * 72)

    logger.info("Project: %s", PROJECT_NAME)
    logger.info("Data file: %s", DATA_FILE)
    logger.info("Output directory: %s", OUTPUT_DIR)
    logger.info("Emission factor: %.2f", EMISSION_FACTOR)

    logger.info(
        "Valid machine IDs: %s",
        ", ".join(MACHINE_IDS)
    )

    logger.info(
        "Valid machine states: %s",
        ", ".join(VALID_STATES)
    )

    percentage = calculate_percentage(
        25,
        100
    )

    co2e = calculate_co2e(
        100,
        EMISSION_FACTOR
    )

    logger.info(
        "Percentage test: %.2f%%",
        percentage
    )

    logger.info(
        "CO2e test: %.2f kg CO2e",
        co2e
    )

    ensure_directory(OUTPUT_DIR)

    logger.info("=" * 72)
    logger.info("SHARED LIBRARY TEST PASSED")
    logger.info("=" * 72)


if __name__ == "__main__":
    main()