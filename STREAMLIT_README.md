# Streamlit Dashboard — Sustainable Manufacturing Analytics

This adds a professional web dashboard to the existing analytics project. It does **not** replace the original command-line application.

## 1. Open the project folder

Extract the project ZIP. Open PowerShell in the folder that contains `app.py`, `src`, `data`, and `outputs`.

## 2. Install dependencies

If you already use a Python virtual environment for this project, activate it first. Then run:

```powershell
python -m pip install -r requirements.txt
```

Or create a virtual environment first:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, use its Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 3. Launch the dashboard

From the project root:

```powershell
python -m streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`, and may open it in your browser. If it does not, copy the URL into Chrome or Edge.

You can also double-click `Run_Streamlit_Dashboard.bat` after installing dependencies.

## 4. Run the existing analytics when needed

The dashboard reads the generated files in `outputs`. If those outputs are already present, you do not need to start the mock API just to view the dashboard.

To regenerate the outputs, run the existing CLI workflow in a separate terminal:

```powershell
python src/main.py
```

Select option `10` to run the complete analytics pipeline. When it finishes, return to the Streamlit browser tab and use **About the Project → Refresh dashboard data**.

The mock API is only needed by workflows that explicitly use it. The Streamlit dashboard as supplied reads the generated CSV outputs and does not call the mock API.

## Troubleshooting

- **`No module named streamlit`:** install Streamlit with the same Python interpreter used to launch the app.
- **Missing output message:** confirm `outputs` is beside `app.py` and the analytics CSV files exist.
- **Port already in use:** run `python -m streamlit run app.py --server.port 8502`.
- **Changes do not appear:** use Streamlit's rerun option or the refresh button.
- **Partial charts:** check that the corresponding output CSV exists and contains the expected columns. Optional missing outputs should not stop the whole app.

## Data interpretation

The dataset is synthetic. The project's emission factor is labelled demonstration-only. Forecasts and scenario/optimization savings are modelled estimates, not measured or guaranteed real-world results. The project contains baseline values from different analysis modules; the dashboard keeps their definitions separate rather than silently reconciling them.

## UI polish update
The latest UI pass adds a more refined dark sustainability theme, a clearer sidebar navigation rail with active/hover states, icon-like page labels, stronger KPI hierarchy, subtle card and chart hover feedback, restrained entrance motion, and reduced-motion support. The visual changes do not replace the underlying analytics. See `UI_REDESIGN_PROMPT.md` for the design brief.
