import os
import subprocess
import sys

from auth import authenticate_user


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")


PHASES = {
    "1": ("EDA", "eda.py"),
    "2": ("Data Cleaning & Validation", "data_cleaning.py"),
    "3": ("Energy Baseline", "baseline.py"),
    "4": ("Inefficiency Analysis", "inefficiency_analysis.py"),
    "5": ("Energy Forecasting", "forecasting.py"),
    "6": ("What-If Scenario Analysis", "what_if_analysis.py"),
    "7": ("CO2 & Sustainability Analysis", "co2_analysis.py"),
    "8": ("Energy Optimization", "optimization.py"),
    "9": ("Recommendations & Executive Insights", "recommendations.py"),
}


def print_header():
    print()
    print("=" * 72)
    print("SUSTAINABLE MANUFACTURING ANALYTICS")
    print("=" * 72)
    print("Energy Consumption & Carbon Footprint Optimization Analytics")
    print("=" * 72)


def print_menu():
    print()
    print("-" * 72)
    print("ANALYTICS MENU")
    print("-" * 72)

    for key, (name, _) in PHASES.items():
        print(f"{key}. {name}")

    print("10. Run Complete Analytics Pipeline")
    print("0. Exit")

    print("-" * 72)


def run_phase(script_name, phase_name):
    script_path = os.path.join(SRC_DIR, script_name)

    print()
    print("=" * 72)
    print(f"STARTING: {phase_name}")
    print("=" * 72)

    if not os.path.exists(script_path):
        print()
        print(f"ERROR: {script_name} was not found.")
        print(f"Expected location: {script_path}")
        return False

    result = subprocess.run(
        [sys.executable, script_path],
        cwd=PROJECT_ROOT
    )

    print()

    if result.returncode == 0:
        print("=" * 72)
        print(f"{phase_name} COMPLETED SUCCESSFULLY")
        print("=" * 72)
        return True

    print("=" * 72)
    print(f"{phase_name} FAILED")
    print("=" * 72)

    return False


def run_complete_pipeline():
    print()
    print("=" * 72)
    print("STARTING COMPLETE ANALYTICS PIPELINE")
    print("=" * 72)

    for key, (phase_name, script_name) in PHASES.items():
        success = run_phase(script_name, phase_name)

        if not success:
            print()
            print("=" * 72)
            print("PIPELINE STOPPED")
            print("=" * 72)
            print(f"Failed phase: {phase_name}")
            return False

    print()
    print("=" * 72)
    print("COMPLETE ANALYTICS PIPELINE FINISHED")
    print("=" * 72)
    print("All analytical phases completed successfully.")

    return True


def main():
    print_header()

    if not authenticate_user():
        sys.exit(1)

    print()
    print("=" * 72)
    print("AUTHENTICATED SESSION STARTED")
    print("=" * 72)

    while True:
        print_menu()

        choice = input("Enter your choice: ").strip()

        if choice == "0":
            print()
            print("=" * 72)
            print("SESSION TERMINATED")
            print("=" * 72)
            print("Thank you for using Sustainable Manufacturing Analytics.")
            break

        if choice == "10":
            success = run_complete_pipeline()

            if success:
                print()
                print("All analytical outputs are ready for Power BI.")

            continue

        if choice in PHASES:
            phase_name, script_name = PHASES[choice]
            run_phase(script_name, phase_name)
            continue

        print()
        print("Invalid choice.")
        print("Please select a number from 0 to 10.")


if __name__ == "__main__":
    main()