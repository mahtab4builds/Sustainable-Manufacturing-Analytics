"""
Sustainable Manufacturing Analytics
Master Pipeline Runner

Executes the complete analytical workflow in the correct dependency order.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import time
from pathlib import Path


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"


# ============================================================
# LOGGING CONFIGURATION
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)


# ============================================================
# PIPELINE CONFIGURATION
# ============================================================

PIPELINE_STAGES = [
    (
        "Exploratory Data Analysis",
        "eda.py",
    ),
    (
        "Data Cleaning & Validation",
        "data_cleaning.py",
    ),
    (
        "Energy Baseline Development",
        "baseline.py",
    ),
    (
        "Inefficiency Analysis",
        "inefficiency_analysis.py",
    ),
    (
        "Energy Forecasting",
        "forecasting.py",
    ),
    (
        "What-If Scenario Analysis",
        "what_if_analysis.py",
    ),
    (
        "CO2 & Sustainability Analysis",
        "co2_analysis.py",
    ),
    (
        "Energy Optimization",
        "optimization.py",
    ),
    (
        "Recommendations & Executive Insights",
        "recommendations.py",
    ),
]


# ============================================================
# DISPLAY HELPERS
# ============================================================

def print_header() -> None:
    """Display the master pipeline header."""

    print()
    print("=" * 80)
    print("SUSTAINABLE MANUFACTURING ANALYTICS")
    print("MASTER ANALYTICAL PIPELINE")
    print("=" * 80)
    print()


def print_stage_header(
    stage_number: int,
    total_stages: int,
    stage_name: str,
) -> None:
    """Display the current pipeline stage."""

    print()
    print("-" * 80)
    print(
        f"STAGE {stage_number}/{total_stages}: {stage_name}"
    )
    print("-" * 80)


# ============================================================
# STAGE EXECUTION
# ============================================================

def run_stage(
    stage_number: int,
    total_stages: int,
    stage_name: str,
    script_name: str,
) -> float:
    """
    Execute one analytical stage and return execution time.
    """

    print_stage_header(
        stage_number,
        total_stages,
        stage_name,
    )

    script_path = SRC_DIR / script_name

    if not script_path.exists():
        raise FileNotFoundError(
            f"Pipeline script not found: {script_path}"
        )

    logger.info(
        "Executing: %s",
        script_path,
    )

    start_time = time.perf_counter()

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
        ],
        cwd=str(PROJECT_ROOT),
        check=False,
    )

    elapsed_time = time.perf_counter() - start_time

    if result.returncode != 0:
        raise RuntimeError(
            f"Stage failed: {stage_name} "
            f"(exit code {result.returncode})"
        )

    logger.info(
        "%s completed successfully in %.2f seconds.",
        stage_name,
        elapsed_time,
    )

    return elapsed_time


# ============================================================
# PIPELINE EXECUTION
# ============================================================

def run_pipeline() -> None:
    """Execute all analytical stages sequentially."""

    print_header()

    total_stages = len(
        PIPELINE_STAGES
    )

    pipeline_start = time.perf_counter()

    completed_stages = []

    for index, (
        stage_name,
        script_name,
    ) in enumerate(
        PIPELINE_STAGES,
        start=1,
    ):

        elapsed = run_stage(
            index,
            total_stages,
            stage_name,
            script_name,
        )

        completed_stages.append(
            {
                "stage": stage_name,
                "script": script_name,
                "time": elapsed,
            }
        )

    total_elapsed = (
        time.perf_counter()
        - pipeline_start
    )

    print()
    print("=" * 80)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 80)

    for item in completed_stages:

        print(
            f"✓ {item['stage']:<45}"
            f"{item['time']:>8.2f}s"
        )

    print("-" * 80)

    print(
        f"Total pipeline execution time:"
        f" {total_elapsed:.2f} seconds"
    )

    print("=" * 80)

    print()
    print(
        "SUSTAINABLE MANUFACTURING ANALYTICS"
    )

    print(
        "PIPELINE COMPLETED SUCCESSFULLY"
    )

    print("=" * 80)
    print()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """Application entry point."""

    try:

        run_pipeline()

    except KeyboardInterrupt:

        logger.warning(
            "Pipeline execution interrupted by user."
        )

        sys.exit(1)

    except Exception as exc:

        logger.exception(
            "Pipeline execution failed: %s",
            exc,
        )

        sys.exit(1)


if __name__ == "__main__":
    main()