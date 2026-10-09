# ============================================================
# SUSTAINABLE MANUFACTURING ANALYTICS
# MOCK API SERVER
# ============================================================

from flask import Flask, jsonify, request
import pandas as pd
from pathlib import Path


# ============================================================
# APPLICATION
# ============================================================

app = Flask(__name__)


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "Sustainable_Manufacturing_Energy_Analytics_10000.xlsx"
)


# ============================================================
# LOAD DATA ONCE
# ============================================================

print("Loading manufacturing dataset...")

DF = pd.read_excel(
    DATA_FILE,
    sheet_name="Manufacturing_Energy_Data"
)

print(
    f"Dataset loaded: {len(DF)} rows × {len(DF.columns)} columns"
)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route(
    "/health",
    methods=["GET"]
)
def health():

    return jsonify({
        "status": "healthy",
        "service": "Sustainable Manufacturing Mock API"
    })


# ============================================================
# MANUFACTURING ENERGY API
# ============================================================

@app.route(
    "/api/manufacturing-energy",
    methods=["GET"]
)
def manufacturing_energy():

    # --------------------------------------------------------
    # Optional record limit
    # Example:
    # /api/manufacturing-energy?limit=5
    # --------------------------------------------------------

    limit = request.args.get(
        "limit",
        default=None,
        type=int
    )

    if limit is not None:

        if limit < 1:

            return jsonify({
                "error": "limit must be greater than 0"
            }), 400

        data = DF.head(limit)

    else:

        data = DF

    records = data.to_dict(
        orient="records"
    )

    return jsonify(records)


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print("=" * 72)
    print("SUSTAINABLE MANUFACTURING MOCK API")
    print("=" * 72)

    print()
    print("Health:")
    print("http://127.0.0.1:5000/health")

    print()
    print("API:")
    print("http://127.0.0.1:5000/api/manufacturing-energy")

    print()
    print("Test API with 1 record:")
    print(
        "http://127.0.0.1:5000/"
        "api/manufacturing-energy?limit=1"
    )

    print()
    print("Press CTRL+C to stop the server.")
    print("=" * 72)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )