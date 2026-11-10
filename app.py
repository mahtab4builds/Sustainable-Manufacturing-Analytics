from pathlib import Path
import io
import re
import hashlib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
DATA = ROOT / "data" / "Sustainable_Manufacturing_Energy_Analytics_10000.xlsx"

st.set_page_config(
    page_title="Sustainability Intelligence | Manufacturing Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)

BG, PANEL, TEXT, MUTED = "#0B1220", "#111C2E", "#E8EEF7", "#9BAAC0"
GREEN, TEAL, BLUE, AMBER, RED = "#38D39F", "#42C6C5", "#6AA9FF", "#F6BD60", "#FF7B84"
GRID = "rgba(148,163,184,.13)"
LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", color=TEXT, size=12),
    margin=dict(l=18, r=18, t=42, b=18),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    hoverlabel=dict(bgcolor=PANEL, font_color=TEXT),
)

st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root {{ color-scheme: dark; }}
html,body,[class*="css"]{{font-family:'DM Sans',sans-serif}}
.stApp{{background:radial-gradient(ellipse at 12% 0%,rgba(56,211,159,.09),transparent 32%),radial-gradient(ellipse at 90% 3%,rgba(66,198,197,.065),transparent 28%),{BG};color:{TEXT}}}
[data-testid="stHeader"]{{background:rgba(11,18,32,.76);backdrop-filter:blur(14px)}}
[data-testid="stSidebar"]{{background:linear-gradient(180deg,#101D30 0%,#0B1220 76%);border-right:1px solid rgba(148,163,184,.18)}}
[data-testid="stSidebar"] *{{color:{TEXT}}}
.block-container{{padding-top:1.6rem;padding-bottom:2.8rem;max-width:1540px}}
h1,h2,h3{{font-family:'Manrope',sans-serif;letter-spacing:-.035em;color:{TEXT}}}
p,label{{color:{MUTED}}}
[data-testid="stMetric"]{{background:linear-gradient(145deg,rgba(24,40,64,.98),rgba(15,27,45,.98));border:1px solid rgba(148,163,184,.17);border-radius:16px;padding:17px;min-height:112px}}
[data-testid="stMetricLabel"] p{{color:{MUTED}!important;font-size:.79rem;font-weight:600}}
[data-testid="stMetricValue"]{{color:{TEXT}!important;font-family:'Manrope',sans-serif;font-weight:800}}
div[data-testid="stPlotlyChart"]{{background:rgba(17,28,46,.65);border:1px solid rgba(148,163,184,.16);border-radius:16px;padding:7px}}
.hero{{background:linear-gradient(115deg,rgba(20,43,61,.98),rgba(17,28,46,.96) 58%,rgba(20,56,56,.82));border:1px solid rgba(66,198,197,.26);border-radius:22px;padding:28px 30px;margin-bottom:22px}}
.eyebrow,.kicker{{color:{GREEN};font-size:.7rem;font-weight:800;letter-spacing:.16em;text-transform:uppercase;margin-bottom:8px}}
.hero-title{{font-family:'Manrope',sans-serif;font-weight:800;letter-spacing:-.05em;color:{TEXT};font-size:clamp(1.9rem,3vw,2.7rem);line-height:1.12;margin:0 0 12px}}
.hero-copy{{color:#B6C5D8;font-size:1rem;line-height:1.65;max-width:900px;margin:0}}
.section-title{{font-family:'Manrope',sans-serif;font-size:1.4rem;font-weight:800;letter-spacing:-.035em;color:{TEXT};margin:0 0 4px}}
.subtitle{{color:{MUTED};font-size:.9rem;line-height:1.55;margin-bottom:15px}}
.insight{{background:linear-gradient(145deg,rgba(22,38,60,.96),rgba(15,27,45,.96));border:1px solid rgba(148,163,184,.16);border-radius:15px;padding:17px;min-height:145px;margin-bottom:14px}}
.insight-label{{color:{GREEN};font-size:.67rem;font-weight:800;letter-spacing:.12em;text-transform:uppercase}}
.insight-title{{color:{TEXT};font-weight:750;margin:8px 0;font-family:'Manrope',sans-serif}}
.insight-copy{{color:{MUTED};font-size:.86rem;line-height:1.55}}
.pill{{display:inline-block;padding:5px 10px;border-radius:99px;background:rgba(56,211,159,.1);color:{GREEN};border:1px solid rgba(56,211,159,.22);font-size:.7rem;font-weight:800}}
[data-testid="stDataFrame"]{{border:1px solid rgba(148,163,184,.18);border-radius:12px;overflow:hidden}}
.stButton button,.stDownloadButton button{{border-radius:10px;font-weight:700}}
hr{{border-color:rgba(148,163,184,.14)!important}}
</style>""", unsafe_allow_html=True)


# ---------- General helpers ----------
def heading(kicker, title, subtitle):
    st.markdown(
        f'<div class="kicker">{kicker}</div><div class="section-title">{title}</div>'
        f'<div class="subtitle">{subtitle}</div>',
        unsafe_allow_html=True,
    )


def style(fig, height=350):
    fig.update_layout(**LAYOUT, height=height)
    fig.update_xaxes(showgrid=False, linecolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig


def fmt(value, decimals=2):
    try:
        if pd.isna(value) or not np.isfinite(float(value)):
            return "—"
        return f"{float(value):,.{decimals}f}"
    except (TypeError, ValueError):
        return "—"


def show_table(df, height=320):
    if df is None or df.empty:
        st.info("There is not enough information in the selected dataset for this analysis.")
    else:
        st.dataframe(df, use_container_width=True, hide_index=True, height=height)


def show_insight(label, title, copy):
    st.markdown(
        f'<div class="insight"><div class="insight-label">{label}</div>'
        f'<div class="insight-title">{title}</div><div class="insight-copy">{copy}</div></div>',
        unsafe_allow_html=True,
    )


def norm_name(value):
    return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())


ALIASES = {
    "timestamp": [
        "Timestamp", "Date_Time", "Datetime", "DateTime", "Date", "Time",
        "Recorded_At", "Reading_Date", "Production_Date",
    ],
    "energy": [
        "Energy_Consumption_kWh", "Energy_Consumed_kWh", "Energy_kWh",
        "Total_Energy_kWh", "Energy", "Power_Used", "Energy_Usage",
        "Electricity_Consumption", "Consumption_kWh", "kWh",
    ],
    "production": [
        "Actual_Production_Units", "Production_Units", "Total_Production_Units",
        "Production", "Output_Units", "Output", "Units_Produced",
        "Quantity_Produced", "Production_Output",
    ],
    "machine": [
        "Machine_Name", "Machine_ID", "Machine", "Equipment",
        "Equipment_Name", "Asset", "Asset_ID", "Machine_Code",
    ],
    "state": [
        "Actual_State", "Machine_State", "Operating_State", "State",
        "Status", "Machine_Status",
    ],
    "shift": ["Shift", "Shift_Name", "Work_Shift", "Production_Shift"],
    "idle_energy": [
        "Idle_Energy_kWh", "Idle_Energy", "Standby_Energy_kWh",
        "Idle_Consumption", "Idle_Power_Consumption",
    ],
    "baseline": ["Baseline_Energy_kWh", "Energy_Baseline", "Baseline_Energy"],
    "co2": ["CO2e_kg", "CO2_kg", "CO2e", "Carbon_Emissions_kg", "Emissions_kg"],
    "temperature": ["Ambient_Temperature_C", "Temperature_C", "Temperature"],
}
FIELD_LABELS = {
    "timestamp": "Date / time",
    "energy": "Energy consumption",
    "production": "Production output",
    "machine": "Machine / equipment",
    "state": "Machine operating state",
    "shift": "Shift",
    "idle_energy": "Idle energy",
    "baseline": "Baseline energy",
    "co2": "CO₂e emissions",
    "temperature": "Ambient temperature",
}


def auto_detect(columns):
    lookup = {norm_name(c): c for c in columns}
    result = {}
    for field, aliases in ALIASES.items():
        found = None
        for alias in aliases:
            if norm_name(alias) in lookup:
                found = lookup[norm_name(alias)]
                break
        result[field] = found
    return result


def read_uploaded(uploaded_file, sheet_name=None):
    content = uploaded_file.getvalue()
    filename = uploaded_file.name.lower()
    if filename.endswith(".csv"):
        try:
            return pd.read_csv(io.BytesIO(content))
        except UnicodeDecodeError:
            return pd.read_csv(io.BytesIO(content), encoding="latin-1")
    excel = pd.ExcelFile(io.BytesIO(content))
    selected = sheet_name if sheet_name in excel.sheet_names else None
    if selected is None:
        # Prefer the worksheet with the most rows and columns; README sheets are usually small.
        sizes = []
        for sheet in excel.sheet_names:
            sample = pd.read_excel(io.BytesIO(content), sheet_name=sheet, nrows=5)
            sizes.append((len(sample.columns), sheet))
        selected = max(sizes, key=lambda item: item[0])[1] if sizes else excel.sheet_names[0]
        # If the workbook contains a clearly named raw-data sheet, prefer it.
        preferred = [s for s in excel.sheet_names if any(k in norm_name(s) for k in ("manufacturingenergydata", "rawdata", "dataset", "data"))]
        if preferred:
            selected = preferred[0]
    return pd.read_excel(io.BytesIO(content), sheet_name=selected)


def derive_shift_from_schedule(records, schedule):
    """Assign each production record to the scheduled shift covering its timestamp."""
    if records.empty or schedule is None or schedule.empty:
        return records, 0
    required_records = {"Timestamp", "Machine_ID"}
    required_schedule = {"Machine_ID", "Shift", "Planned_Start", "Planned_End"}
    if not required_records.issubset(records.columns) or not required_schedule.issubset(schedule.columns):
        return records, 0

    result = records.copy()
    if "Shift" in result.columns and result["Shift"].notna().any():
        return result, 0

    result["Timestamp"] = pd.to_datetime(result["Timestamp"], errors="coerce")
    sched = schedule.copy()
    sched["Planned_Start"] = pd.to_datetime(sched["Planned_Start"], errors="coerce")
    sched["Planned_End"] = pd.to_datetime(sched["Planned_End"], errors="coerce")
    sched = sched.dropna(subset=["Machine_ID", "Shift", "Planned_Start", "Planned_End"])
    result["Shift"] = pd.Series(index=result.index, dtype="object")

    # Process one machine at a time. searchsorted finds the latest shift start
    # without creating a many-to-many join between hourly records and schedules.
    for machine_id, record_index in result.groupby("Machine_ID", dropna=False).groups.items():
        if pd.isna(machine_id):
            continue
        machine_schedule = sched[sched["Machine_ID"].astype(str) == str(machine_id)].sort_values("Planned_Start")
        if machine_schedule.empty:
            continue
        idx = pd.Index(record_index)
        times = result.loc[idx, "Timestamp"]
        valid = times.notna()
        if not valid.any():
            continue
        starts = machine_schedule["Planned_Start"].to_numpy(dtype="datetime64[ns]")
        ends = machine_schedule["Planned_End"].to_numpy(dtype="datetime64[ns]")
        positions = np.searchsorted(starts, times.loc[valid].to_numpy(dtype="datetime64[ns]"), side="right") - 1
        valid_positions = positions >= 0
        valid_indices = times.loc[valid].index.to_numpy()
        if valid_positions.any():
            candidate_positions = positions[valid_positions]
            candidate_times = times.loc[valid].to_numpy(dtype="datetime64[ns]")[valid_positions]
            inside = candidate_times < ends[candidate_positions]
            target_indices = valid_indices[valid_positions][inside]
            shift_values = machine_schedule["Shift"].to_numpy()[candidate_positions[inside]]
            result.loc[target_indices, "Shift"] = shift_values

    assigned = int(result["Shift"].notna().sum())
    return result, assigned


def load_default_schedule():
    if not DATA.exists():
        return pd.DataFrame()
    try:
        book = pd.ExcelFile(DATA)
        if "Production_Schedule" in book.sheet_names:
            return pd.read_excel(DATA, sheet_name="Production_Schedule")
    except Exception:
        pass
    return pd.DataFrame()


def load_default_data():
    cleaned = OUT / "cleaned_data" / "manufacturing_energy_cleaned.csv"
    if cleaned.exists():
        try:
            return pd.read_csv(cleaned), "Project cleaned dataset"
        except Exception:
            pass
    if DATA.exists():
        try:
            xls = pd.ExcelFile(DATA)
            target = "Manufacturing_Energy_Data" if "Manufacturing_Energy_Data" in xls.sheet_names else xls.sheet_names[0]
            return pd.read_excel(DATA, sheet_name=target), "Project source dataset"
        except Exception:
            pass
    return pd.DataFrame(), "No default dataset found"


def prepare_dataset(raw, mapping):
    df = raw.copy()
    # Standardized analytical columns are added; original columns are retained.
    for field, original in mapping.items():
        canonical = {
            "timestamp": "Timestamp", "energy": "Energy_Consumption_kWh",
            "production": "Actual_Production_Units", "machine": "Machine_Name",
            "state": "Actual_State", "shift": "Shift", "idle_energy": "Idle_Energy_kWh",
            "baseline": "Baseline_Energy_kWh", "co2": "CO2e_kg",
            "temperature": "Ambient_Temperature_C",
        }[field]
        if original and original in df.columns:
            df[canonical] = df[original]
    for col in [
        "Energy_Consumption_kWh", "Actual_Production_Units", "Idle_Energy_kWh",
        "Baseline_Energy_kWh", "CO2e_kg", "Ambient_Temperature_C",
    ]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "Timestamp" in df:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce", dayfirst=True)
    return df


def aggregate_data(df, emission_factor):
    energy_col = "Energy_Consumption_kWh"
    production_col = "Actual_Production_Units"
    machine_col = "Machine_Name"
    has_energy = energy_col in df and pd.to_numeric(df[energy_col], errors="coerce").notna().any()
    has_prod = production_col in df and pd.to_numeric(df[production_col], errors="coerce").notna().any()
    energy_series = pd.to_numeric(df[energy_col], errors="coerce") if has_energy else pd.Series(dtype=float)
    prod_series = pd.to_numeric(df[production_col], errors="coerce") if has_prod else pd.Series(dtype=float)
    total_energy = float(energy_series.sum()) if has_energy else np.nan
    total_prod = float(prod_series.sum()) if has_prod else np.nan

    machine = pd.DataFrame()
    if has_energy and machine_col in df:
        temp = df.assign(_energy=energy_series)
        group = temp.groupby(machine_col, dropna=False)
        machine = group.agg(Total_Energy_kWh=("_energy", "sum")).reset_index().rename(columns={machine_col: "Machine_Name"})
        if has_prod:
            prod = temp.assign(_production=prod_series).groupby(machine_col, dropna=False)["_production"].sum()
            machine["Total_Production_Units"] = machine["Machine_Name"].map(prod)
            machine["Energy_Per_Unit_kWh"] = machine["Total_Energy_kWh"] / machine["Total_Production_Units"].replace(0, np.nan)
        if "Idle_Energy_kWh" in df:
            idle = temp.assign(_idle=pd.to_numeric(df["Idle_Energy_kWh"], errors="coerce")).groupby(machine_col, dropna=False)["_idle"].sum()
            machine["Total_Idle_Energy_kWh"] = machine["Machine_Name"].map(idle)
        machine = machine.sort_values("Total_Energy_kWh", ascending=False)

    state = pd.DataFrame()
    if "Actual_State" in df:
        state = df["Actual_State"].fillna("Unknown").astype(str).value_counts().rename_axis("Actual_State").reset_index(name="Observations")

    shift = pd.DataFrame()
    if "Shift" in df and has_energy:
        temp = df.assign(_energy=energy_series)
        grouped = temp.groupby("Shift", dropna=False).agg(Energy_kWh=("_energy", "sum")).reset_index()
        if has_prod:
            grouped["Production_Units"] = temp.assign(_production=prod_series).groupby("Shift", dropna=False)["_production"].sum().values
            grouped["Energy_Per_Unit_kWh"] = grouped["Energy_kWh"] / grouped["Production_Units"].replace(0, np.nan)
        else:
            grouped["Energy_Per_Unit_kWh"] = np.nan
        shift = grouped.rename(columns={"Energy_kWh": "Total_Energy_kWh"})

    idle = pd.DataFrame()
    if "Idle_Energy_kWh" in df and machine_col in df:
        idle = df.groupby(machine_col, dropna=False)["Idle_Energy_kWh"].sum().reset_index()
        idle = idle.rename(columns={machine_col: "Machine_ID", "Idle_Energy_kWh": "Idle_Energy_kWh"})
        idle["Idle_Energy_Percentage"] = idle["Idle_Energy_kWh"] / total_energy * 100 if total_energy and np.isfinite(total_energy) else np.nan

    hourly = pd.DataFrame()
    if has_energy and "Timestamp" in df:
        temp = df.assign(_energy=energy_series, Hour=df["Timestamp"].dt.hour).dropna(subset=["Hour"])
        hourly = temp.groupby("Hour")["_energy"].agg(Average_Energy_kWh="mean", Minimum_Energy_kWh="min", Maximum_Energy_kWh="max").reset_index()

    baseline = pd.DataFrame()
    if has_energy:
        base = pd.to_numeric(df["Baseline_Energy_kWh"], errors="coerce") if "Baseline_Energy_kWh" in df else pd.Series(dtype=float)
        if base.notna().any():
            baseline_value = float(base.sum())
            baseline_label = "Dataset-provided baseline"
        else:
            # Explicit reference estimate: median observed energy per record, scaled to record count.
            baseline_value = float(energy_series.median() * len(energy_series.dropna()))
            baseline_label = "Median-per-record reference (estimated)"
        baseline = pd.DataFrame([
            {"Metric": "Actual Energy (kWh)", "Value": total_energy},
            {"Metric": "Calculated Baseline Energy (kWh)", "Value": baseline_value},
            {"Metric": "Energy Deviation (kWh)", "Value": total_energy - baseline_value},
        ])

    co2_series = pd.to_numeric(df["CO2e_kg"], errors="coerce") if "CO2e_kg" in df else pd.Series(dtype=float)
    if co2_series.notna().any():
        total_co2 = float(co2_series.sum())
        co2_source = "Dataset-provided CO₂e column"
    elif has_energy:
        total_co2 = total_energy * emission_factor
        co2_source = f"Estimated using {emission_factor:.3f} kg CO₂e/kWh"
    else:
        total_co2 = np.nan
        co2_source = "Unavailable"
    co2_intensity = total_co2 / total_prod if has_prod and total_prod else np.nan

    exec_kpi = pd.DataFrame([
        {"Metric": "Total Energy Consumption", "Value": total_energy},
        {"Metric": "Total Production", "Value": total_prod},
        {"Metric": "Energy per Production Unit", "Value": total_energy / total_prod if has_energy and has_prod and total_prod else np.nan},
        {"Metric": "Idle Energy", "Value": float(pd.to_numeric(df["Idle_Energy_kWh"], errors="coerce").sum()) if "Idle_Energy_kWh" in df else np.nan},
        {"Metric": "Total CO2e", "Value": total_co2},
        {"Metric": "CO2e per Production Unit", "Value": co2_intensity},
    ])

    forecast = pd.DataFrame()
    if has_energy:
        if "Timestamp" in df and df["Timestamp"].notna().any():
            daily = df.assign(_energy=energy_series).dropna(subset=["Timestamp"]).set_index("Timestamp")["_energy"].resample("D").sum().dropna()
            daily = daily[daily.index.notna()]
        else:
            daily = pd.Series(energy_series.dropna().tail(30).values)
        if len(daily) >= 2:
            recent = daily.tail(min(14, len(daily))).astype(float)
            average = float(recent.mean())
            x = np.arange(len(recent), dtype=float)
            slope = float(np.polyfit(x, recent.values, 1)[0]) if len(recent) >= 2 else 0.0
            slope = float(np.clip(slope, -average * 0.05, average * 0.05))
            if isinstance(recent.index, pd.DatetimeIndex):
                last_date = recent.index.max()
                dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=14, freq="D")
                values = [max(0.0, average + slope * (i + 1)) for i in range(14)]
                forecast = pd.DataFrame({"Timestamp": dates, "Forecast_Energy_kWh": values, "Forecast_Horizon_Day": range(1, 15)})
            else:
                forecast = pd.DataFrame({"Forecast_Horizon_Day": range(1, 15), "Forecast_Energy_kWh": [max(0.0, average + slope * (i + 1)) for i in range(14)]})

    scenarios = pd.DataFrame()
    optimization = pd.DataFrame()
    if has_energy:
        idle_total = float(pd.to_numeric(df["Idle_Energy_kWh"], errors="coerce").sum()) if "Idle_Energy_kWh" in df else 0.0
        scenarios = pd.DataFrame([
            {"Scenario": "REDUCE_IDLE_ENERGY_10_PERCENT", "Potential_Energy_Reduction_kWh": idle_total * .10},
            {"Scenario": "REDUCE_IDLE_ENERGY_20_PERCENT", "Potential_Energy_Reduction_kWh": idle_total * .20},
            {"Scenario": "IMPROVE_TOTAL_EFFICIENCY_5_PERCENT", "Potential_Energy_Reduction_kWh": total_energy * .05},
            {"Scenario": "IMPROVE_TOTAL_EFFICIENCY_10_PERCENT", "Potential_Energy_Reduction_kWh": total_energy * .10},
        ])
        if not machine.empty:
            optimization = machine[["Machine_Name", "Total_Energy_kWh"]].copy()
            optimization["Optimized_Total_Reduction_kWh"] = optimization["Total_Energy_kWh"] * .05
            optimization["Potential_Reduction_Percent"] = 5.0
            optimization["Optimized_Energy_kWh"] = optimization["Total_Energy_kWh"] - optimization["Optimized_Total_Reduction_kWh"]
            if np.isfinite(emission_factor):
                optimization["Potential_CO2_Reduction_kg"] = optimization["Optimized_Total_Reduction_kWh"] * emission_factor

    ranking = pd.DataFrame()
    if not machine.empty and "Energy_Per_Unit_kWh" in machine:
        ranking = machine[["Machine_Name", "Energy_Per_Unit_kWh"]].copy()
        ranking["Sustainability_Attention_Score"] = ranking["Energy_Per_Unit_kWh"].rank(pct=True) * 100
        ranking = ranking.sort_values("Sustainability_Attention_Score", ascending=False)

    return {
        "has_energy": has_energy, "has_prod": has_prod, "total_energy": total_energy,
        "total_prod": total_prod, "intensity": total_energy / total_prod if has_energy and has_prod and total_prod else np.nan,
        "idle_total": float(pd.to_numeric(df["Idle_Energy_kWh"], errors="coerce").sum()) if "Idle_Energy_kWh" in df else np.nan,
        "total_co2": total_co2, "co2_intensity": co2_intensity, "co2_source": co2_source,
        "machine": machine, "state": state, "shift": shift, "idle": idle, "hourly": hourly,
        "baseline": baseline, "forecast": forecast, "scenarios": scenarios,
        "optimization": optimization, "ranking": ranking,
    }


# ---------- Sidebar, upload and source selection ----------
NAV = [
    "Overview", "Dataset Upload & Analysis", "Energy Performance",
    "Machine Efficiency", "Operations", "Forecast & What-If",
    "CO₂ & Optimization", "Recommendations", "About the Project",
]
with st.sidebar:
    st.markdown(
        f'<div style="padding:8px 2px 18px"><div style="color:{GREEN};font-size:.68rem;letter-spacing:.15em;font-weight:800">SUSTAINABILITY INTELLIGENCE</div>'
        f'<div style="font-family:Manrope,sans-serif;font-size:1.25rem;font-weight:800;line-height:1.2;margin-top:8px">Manufacturing<br>Analytics</div>'
        f'<div style="font-size:.77rem;color:{MUTED};margin-top:8px">Energy · Efficiency · Carbon</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    page = st.radio("NAVIGATION", NAV, label_visibility="visible")
    st.markdown("---")
    st.markdown("**DATASET SOURCE**")
    uploaded_file = st.file_uploader("Upload CSV or Excel workbook", type=["csv", "xlsx"], key="dataset_upload_v2")
    sheet_choice = None
    if uploaded_file is not None and uploaded_file.name.lower().endswith(".xlsx"):
        try:
            workbook = pd.ExcelFile(io.BytesIO(uploaded_file.getvalue()))
            preferred = next((s for s in workbook.sheet_names if "manufacturingenergydata" in norm_name(s)), workbook.sheet_names[0])
            default_index = workbook.sheet_names.index(preferred)
            sheet_choice = st.selectbox("Worksheet", workbook.sheet_names, index=default_index, key=f"sheet_{uploaded_file.name}")
        except Exception as exc:
            st.error(f"Could not inspect workbook: {exc}")
    if uploaded_file is not None:
        st.caption(f"Active upload: {uploaded_file.name}")
        if st.button("Use project dataset instead", use_container_width=True):
            st.session_state.pop("dataset_upload_v2", None)
            st.rerun()
    st.markdown("---")
    st.caption("Python · Pandas · Plotly · Streamlit")

# Load selected dataset. An upload always takes precedence over the project default.
upload_error = None
if uploaded_file is not None:
    try:
        raw = read_uploaded(uploaded_file, sheet_choice)
        source_label = f"Uploaded: {uploaded_file.name}" + (f" · {sheet_choice}" if sheet_choice else "")
    except Exception as exc:
        raw = pd.DataFrame()
        source_label = "Upload could not be read"
        upload_error = str(exc)
else:
    raw, source_label = load_default_data()

# A workbook may keep shift information in a related schedule worksheet rather than
# repeating it in every observation row. Load that sheet and align shifts by machine
# and actual timestamp so that shift analytics work across the whole application.
schedule_context = pd.DataFrame()
if uploaded_file is not None and uploaded_file.name.lower().endswith(".xlsx"):
    try:
        workbook = pd.ExcelFile(io.BytesIO(uploaded_file.getvalue()))
        if "Production_Schedule" in workbook.sheet_names:
            schedule_context = pd.read_excel(io.BytesIO(uploaded_file.getvalue()), sheet_name="Production_Schedule")
    except Exception:
        schedule_context = pd.DataFrame()
elif uploaded_file is None:
    schedule_context = load_default_schedule()

derived_shift_count = 0
if not raw.empty and not schedule_context.empty:
    raw, derived_shift_count = derive_shift_from_schedule(raw, schedule_context)

if raw.empty and not upload_error:
    upload_error = "No data rows were found in the selected dataset."

# Automatic mapping is used immediately. Users can override mappings on the upload page.
detected = auto_detect(raw.columns) if not raw.empty else {key: None for key in ALIASES}
dataset_token = hashlib.md5(uploaded_file.getvalue()).hexdigest()[:12] if uploaded_file is not None else "project_default"
mapping_state_key = f"manual_mapping_{dataset_token}"
if mapping_state_key not in st.session_state:
    st.session_state[mapping_state_key] = {}
mapping = detected.copy()
mapping.update({k: v for k, v in st.session_state[mapping_state_key].items() if v in raw.columns})
src = prepare_dataset(raw, mapping) if not raw.empty else pd.DataFrame()

# If the source is a workbook containing project data, retain its native analytical columns.
emission_factor = 0.7
if not raw.empty and "CO2e_kg" in src and pd.to_numeric(src["CO2e_kg"], errors="coerce").notna().any():
    emission_factor = 0.7
metrics = aggregate_data(src, emission_factor) if not src.empty else aggregate_data(pd.DataFrame(), emission_factor)
machine = metrics["machine"]
state = metrics["state"]
shift = metrics["shift"]
idle = metrics["idle"]
hourly = metrics["hourly"]
baseline = metrics["baseline"]
forecast = metrics["forecast"]
scenarios = metrics["scenarios"]
optimization = metrics["optimization"]
ranking = metrics["ranking"]

with st.sidebar:
    st.markdown("---")
    st.markdown("**PROJECT STATUS**")
    st.markdown(f'<span class="pill">{"UPLOAD ACTIVE" if uploaded_file is not None and upload_error is None else "PROJECT DATA" if not raw.empty else "NO DATA"}</span>', unsafe_allow_html=True)
    st.caption(f"{len(src):,} records · {len(src.columns):,} columns" if not src.empty else "No usable dataset loaded")
    st.caption("All dashboard pages use the active dataset.")

if upload_error:
    st.error(upload_error)


# ---------- Page: Dataset Upload & Analysis ----------
if page == "Dataset Upload & Analysis":
    heading("00 / Data source", "Dataset upload & analysis", "Choose a dataset, confirm its worksheet and map its fields. The active dataset then drives every dashboard page.")
    if raw.empty:
        st.info("Upload a CSV or Excel workbook from the sidebar to begin. Without an upload, the application uses the project's default dataset.")
    else:
        st.success(f"Active dataset: {source_label}")
        if derived_shift_count:
            st.info(f"Shift information was successfully matched from the workbook's Production_Schedule worksheet for {derived_shift_count:,} records using machine IDs and actual timestamps.")
        c = st.columns(4)
        c[0].metric("Records", f"{len(raw):,}")
        c[1].metric("Columns", f"{len(raw.columns):,}")
        c[2].metric("Duplicate rows", f"{int(raw.duplicated().sum()):,}")
        c[3].metric("Missing cells", f"{int(raw.isna().sum().sum()):,}")
        with st.expander("Preview source data", expanded=True):
            st.dataframe(raw.head(100), use_container_width=True, hide_index=True)
        st.markdown("---")
        heading("Field configuration", "Map your dataset columns", "Automatic matching is attempted first. Change any mapping when your file uses different names. Select Not available when a field does not exist.")
        options = ["Not available"] + list(raw.columns)
        changed = False
        cols_layout = st.columns(2)
        for index, field in enumerate(ALIASES):
            detected_col = mapping.get(field)
            current = detected_col if detected_col in raw.columns else "Not available"
            widget_key = f"map_{field}_{dataset_token}"
            if widget_key not in st.session_state:
                st.session_state[widget_key] = current
            with cols_layout[index % 2]:
                selected = st.selectbox(
                    FIELD_LABELS[field],
                    options,
                    index=options.index(st.session_state[widget_key]) if st.session_state[widget_key] in options else 0,
                    key=widget_key,
                )
            if selected != "Not available":
                st.session_state[mapping_state_key][field] = selected
            else:
                st.session_state[mapping_state_key].pop(field, None)
            if selected != current:
                changed = True
        if changed:
            st.info("Your field selections are saved for this session. Other pages use the updated mapping after the next rerun.")
            if st.button("Apply field mapping", type="primary"):
                st.rerun()
        # Refresh mapping immediately from widget values for this run.
        mapping = detected.copy()
        for field in ALIASES:
            key = f"map_{field}_{dataset_token}"
            choice = st.session_state.get(key, "Not available")
            mapping[field] = None if choice == "Not available" else choice
        src = prepare_dataset(raw, mapping)
        metrics = aggregate_data(src, emission_factor)
        machine, state, shift, idle, hourly = metrics["machine"], metrics["state"], metrics["shift"], metrics["idle"], metrics["hourly"]
        baseline, forecast, scenarios, optimization, ranking = metrics["baseline"], metrics["forecast"], metrics["scenarios"], metrics["optimization"], metrics["ranking"]

        st.markdown("---")
        heading("Data quality", "Column-level quality profile", "Review missing values, data types and distinct values before interpreting the results.")
        quality = pd.DataFrame({
            "Column": raw.columns,
            "Data type": [str(raw[c].dtype) for c in raw.columns],
            "Missing values": [int(raw[c].isna().sum()) for c in raw.columns],
            "Missing (%)": [round(float(raw[c].isna().mean() * 100), 2) for c in raw.columns],
            "Unique values": [int(raw[c].nunique(dropna=True)) for c in raw.columns],
        })
        show_table(quality, min(480, 110 + 34 * len(quality)))
        st.markdown("---")
        heading("Field readiness", "Available analysis fields", "A check of which core measures can be calculated from the current mapping.")
        readiness = pd.DataFrame([
            {"Analysis field": FIELD_LABELS[k], "Mapped source column": mapping.get(k) or "Not mapped", "Status": "Ready" if mapping.get(k) else "Unavailable"}
            for k in ALIASES
        ])
        show_table(readiness, 360)
        st.download_button("Download active dataset as CSV", raw.to_csv(index=False).encode("utf-8"), "active_dataset.csv", "text/csv")


# ---------- Recompute after mapping controls, for all pages ----------
if page != "Dataset Upload & Analysis" and not raw.empty:
    mapping = detected.copy()
    for field in ALIASES:
        key = f"map_{field}_{dataset_token}"
        choice = st.session_state.get(key)
        if choice and choice != "Not available" and choice in raw.columns:
            mapping[field] = choice
    src = prepare_dataset(raw, mapping)
    metrics = aggregate_data(src, emission_factor)
    machine, state, shift, idle, hourly = metrics["machine"], metrics["state"], metrics["shift"], metrics["idle"], metrics["hourly"]
    baseline, forecast, scenarios, optimization, ranking = metrics["baseline"], metrics["forecast"], metrics["scenarios"], metrics["optimization"], metrics["ranking"]


# ---------- Page: Overview ----------
if page == "Overview":
    st.markdown(f'<div class="hero"><div class="eyebrow">Sustainability intelligence · Active dataset</div><div class="hero-title">Make every kilowatt count.</div><p class="hero-copy">A unified view of energy, production, machine performance and carbon impact. Upload a new dataset to refresh the analytical story across every page.</p></div>', unsafe_allow_html=True)
    heading("Executive snapshot", "Performance at a glance", f"Current source: {source_label}. All available measures below are calculated from this dataset.")
    c = st.columns(4)
    c[0].metric("Total energy", f"{fmt(metrics['total_energy'])} kWh" if metrics["has_energy"] else "Unavailable")
    c[1].metric("Production output", f"{fmt(metrics['total_prod'])} units" if metrics["has_prod"] else "Unavailable")
    c[2].metric("Energy intensity", f"{fmt(metrics['intensity'], 3)} kWh/unit" if np.isfinite(metrics["intensity"]) else "Unavailable")
    c[3].metric("CO₂e emissions", f"{fmt(metrics['total_co2'])} kg" if np.isfinite(metrics["total_co2"]) else "Unavailable")
    c = st.columns(4)
    c[0].metric("Idle energy", f"{fmt(metrics['idle_total'])} kWh" if np.isfinite(metrics["idle_total"]) else "Unavailable")
    c[1].metric("CO₂e intensity", f"{fmt(metrics['co2_intensity'], 3)} kg/unit" if np.isfinite(metrics["co2_intensity"]) else "Unavailable")
    forecast_total = forecast["Forecast_Energy_kWh"].sum() if not forecast.empty else np.nan
    c[2].metric("14-day forecast", f"{fmt(forecast_total)} kWh" if np.isfinite(forecast_total) else "Unavailable")
    opportunity = scenarios["Potential_Energy_Reduction_kWh"].max() if not scenarios.empty else np.nan
    c[3].metric("Modelled opportunity", f"{fmt(opportunity)} kWh" if np.isfinite(opportunity) else "Unavailable")
    left, right = st.columns([1.5, 1])
    with left:
        heading("Energy profile", "Consumption over time", "Daily energy consumption from the active dataset.")
        if metrics["has_energy"] and "Timestamp" in src and src["Timestamp"].notna().any():
            trend = src.dropna(subset=["Timestamp"]).assign(_energy=src["Energy_Consumption_kWh"]).groupby(src["Timestamp"].dt.date)["_energy"].sum().reset_index()
            trend.columns = ["Date", "Energy (kWh)"]
            fig = px.area(trend, x="Date", y="Energy (kWh)", color_discrete_sequence=[GREEN])
            fig.update_traces(line=dict(width=2), fillcolor="rgba(56,211,159,.12)")
            st.plotly_chart(style(fig, 340), use_container_width=True)
        else:
            st.info("A daily trend requires both a mapped date/time field and an energy field.")
    with right:
        heading("Machine mix", "Energy by machine", "Energy contribution by equipment.")
        if not machine.empty:
            fig = px.bar(machine.sort_values("Total_Energy_kWh"), x="Total_Energy_kWh", y="Machine_Name", orientation="h", text_auto=".2s", color="Total_Energy_kWh", color_continuous_scale=["#1B4965", GREEN])
            fig.update_layout(coloraxis_showscale=False); fig.update_yaxes(title=None); fig.update_xaxes(title="Energy (kWh)")
            st.plotly_chart(style(fig, 340), use_container_width=True)
        else:
            st.info("Map a machine field and an energy field to display the machine mix.")
    heading("Decision support", "What deserves attention?", "These observations are derived from the active dataset.")
    a, b, c = st.columns(3)
    top_machine = machine.iloc[0]["Machine_Name"] if not machine.empty else "Not available"
    top_energy = machine.iloc[0]["Total_Energy_kWh"] if not machine.empty else np.nan
    with a: show_insight("Highest energy use", str(top_machine), f"{fmt(top_energy)} kWh recorded." if np.isfinite(top_energy) else "Machine and energy fields are required.")
    with b: show_insight("Idle energy", f"{fmt(metrics['idle_total'])} kWh" if np.isfinite(metrics["idle_total"]) else "Not available", "Based on the mapped idle-energy field; no idle estimate is invented.")
    with c: show_insight("Carbon estimate", f"{fmt(metrics['total_co2'])} kg CO₂e" if np.isfinite(metrics["total_co2"]) else "Not available", metrics["co2_source"])


# ---------- Page: Energy Performance ----------
elif page == "Energy Performance":
    heading("01 / Energy", "Energy performance", "Explore energy use, reference baselines, intensity and operating patterns from the active dataset.")
    filtered = src.copy()
    if "Timestamp" in filtered and filtered["Timestamp"].notna().any():
        valid = filtered["Timestamp"].dropna()
        dates = st.date_input("Analysis date range", value=(valid.min().date(), valid.max().date()), min_value=valid.min().date(), max_value=valid.max().date())
        if isinstance(dates, tuple) and len(dates) == 2:
            filtered = filtered[filtered["Timestamp"].dt.date.between(dates[0], dates[1])]
    if "Machine_Name" in filtered:
        opts = sorted(filtered["Machine_Name"].dropna().astype(str).unique())
        chosen = st.multiselect("Machines", opts, default=opts)
        filtered = filtered[filtered["Machine_Name"].astype(str).isin(chosen)]
    if "Energy_Consumption_kWh" in filtered and "Timestamp" in filtered and filtered["Timestamp"].notna().any():
        trend = filtered.dropna(subset=["Timestamp"]).assign(_date=filtered["Timestamp"].dt.date).groupby("_date")["Energy_Consumption_kWh"].sum().reset_index()
        trend.columns = ["Date", "Energy (kWh)"]
        fig = px.line(trend, x="Date", y="Energy (kWh)", markers=True, color_discrete_sequence=[GREEN])
        st.plotly_chart(style(fig, 360), use_container_width=True)
    else:
        st.info("The trend requires mapped date/time and energy columns.")
    left, right = st.columns(2)
    with left:
        heading("Baseline", "Observed vs reference energy", "A provided baseline is preferred. If absent, the page shows a clearly labelled median-per-record reference.")
        if not baseline.empty:
            fig = px.bar(baseline, x="Metric", y="Value", color="Metric", text_auto=".3s", color_discrete_sequence=[GREEN, BLUE, AMBER])
            fig.update_layout(showlegend=False); fig.update_xaxes(title=None, tickangle=-10); fig.update_yaxes(title="Energy (kWh)")
            st.plotly_chart(style(fig, 320), use_container_width=True)
        else:
            st.info("Map an energy field to calculate the reference comparison.")
    with right:
        heading("Energy intensity", "Energy per production unit", "Machine-level energy divided by production where both fields exist.")
        if not machine.empty and "Energy_Per_Unit_kWh" in machine:
            fig = px.bar(machine.sort_values("Energy_Per_Unit_kWh"), x="Energy_Per_Unit_kWh", y="Machine_Name", orientation="h", text_auto=".2f", color="Energy_Per_Unit_kWh", color_continuous_scale=["#1B4965", GREEN])
            fig.update_layout(coloraxis_showscale=False); fig.update_xaxes(title="kWh per unit"); fig.update_yaxes(title=None)
            st.plotly_chart(style(fig, 320), use_container_width=True)
        else:
            st.info("Energy intensity requires energy, production and machine fields.")
    if not hourly.empty:
        heading("Load profile", "Average energy by hour", "Hourly profile derived from the mapped date/time field.")
        fig = px.bar(hourly, x="Hour", y="Average_Energy_kWh", color_discrete_sequence=[TEAL], hover_data=["Minimum_Energy_kWh", "Maximum_Energy_kWh"])
        fig.update_xaxes(title="Hour of day"); fig.update_yaxes(title="Average energy (kWh)")
        st.plotly_chart(style(fig, 310), use_container_width=True)
    st.download_button("Download filtered source data", filtered.to_csv(index=False).encode("utf-8"), "energy_performance_filtered.csv", "text/csv")


# ---------- Page: Machine Efficiency ----------
elif page == "Machine Efficiency":
    heading("02 / Assets", "Machine efficiency", "Compare energy demand, production output and idle consumption by machine.")
    if not machine.empty:
        left, right = st.columns(2)
        with left:
            fig = px.bar(machine.sort_values("Total_Energy_kWh"), x="Total_Energy_kWh", y="Machine_Name", orientation="h", text_auto=".2s", color="Total_Energy_kWh", color_continuous_scale=["#1B4965", GREEN])
            fig.update_layout(coloraxis_showscale=False); fig.update_xaxes(title="Total energy (kWh)"); fig.update_yaxes(title=None)
            st.plotly_chart(style(fig, 340), use_container_width=True)
        with right:
            if "Total_Production_Units" in machine:
                fig = px.scatter(machine, x="Total_Production_Units", y="Total_Energy_kWh", color="Machine_Name", hover_name="Machine_Name", size="Total_Idle_Energy_kWh" if "Total_Idle_Energy_kWh" in machine else None)
                fig.update_xaxes(title="Production (units)"); fig.update_yaxes(title="Energy (kWh)")
                st.plotly_chart(style(fig, 340), use_container_width=True)
            else:
                st.info("Map production output to compare energy against production.")
        if "Energy_Per_Unit_kWh" in machine:
            fig = px.bar(machine.sort_values("Energy_Per_Unit_kWh"), x="Energy_Per_Unit_kWh", y="Machine_Name", orientation="h", text_auto=".2f", color="Energy_Per_Unit_kWh", color_continuous_scale=["#1B4965", AMBER])
            fig.update_layout(coloraxis_showscale=False); fig.update_xaxes(title="Energy intensity (kWh/unit)"); fig.update_yaxes(title=None)
            st.plotly_chart(style(fig, 320), use_container_width=True)
        show_table(machine.round(3), 340)
        st.download_button("Download machine analysis", machine.to_csv(index=False).encode("utf-8"), "machine_efficiency.csv", "text/csv")
    else:
        st.info("Machine efficiency requires a mapped machine/equipment field and an energy field.")


# ---------- Page: Operations ----------
elif page == "Operations":
    heading("03 / Operations", "Operational intelligence", "Inspect machine states, shift patterns and idle energy from the active dataset.")
    left, right = st.columns(2)
    with left:
        heading("Machine states", "Operating-state mix", "Count of observations in each mapped state.")
        if not state.empty:
            fig = px.pie(state, names="Actual_State", values="Observations", hole=.66, color_discrete_sequence=[GREEN, AMBER, BLUE, RED])
            fig.update_traces(textposition="outside", textinfo="label+percent"); fig.update_layout(showlegend=False)
            st.plotly_chart(style(fig, 330), use_container_width=True)
        else:
            st.info("Map a machine operating-state field to show the state mix.")
    with right:
        heading("Shift comparison", "Energy intensity by shift", "Compare energy per production unit across shifts.")
        if not shift.empty and shift["Energy_Per_Unit_kWh"].notna().any():
            fig = px.bar(shift, x="Shift", y="Energy_Per_Unit_kWh", color="Shift", text_auto=".2f", color_discrete_sequence=[GREEN, TEAL, BLUE])
            fig.update_layout(showlegend=False); fig.update_yaxes(title="kWh per unit")
            st.plotly_chart(style(fig, 330), use_container_width=True)
        elif not shift.empty:
            fig = px.bar(shift, x="Shift", y="Total_Energy_kWh", color="Shift", text_auto=".2s")
            fig.update_layout(showlegend=False); fig.update_yaxes(title="Energy (kWh)")
            st.plotly_chart(style(fig, 330), use_container_width=True)
        else:
            st.info("Shift analysis is unavailable because this dataset has no mapped shift field (or no energy values). If the workbook contains a Production_Schedule worksheet, its shift schedule is matched automatically using machine IDs and timestamps.")
    if not idle.empty:
        heading("Idle energy", "Where idle consumption accumulates", "Uses the uploaded idle-energy measure; it is not inferred from machine state.")
        fig = px.bar(idle.sort_values("Idle_Energy_kWh"), x="Idle_Energy_kWh", y="Machine_ID", orientation="h", text_auto=".2s", color="Idle_Energy_Percentage", color_continuous_scale=["#1B4965", AMBER])
        fig.update_layout(coloraxis_showscale=False); fig.update_xaxes(title="Idle energy (kWh)"); fig.update_yaxes(title=None)
        st.plotly_chart(style(fig, 310), use_container_width=True)
        show_table(idle.round(3), 260)
    else:
        st.info("Map idle energy and machine fields to show machine-level idle consumption.")
    if "Maintenance_Flag" in src:
        heading("Maintenance", "Maintenance flag distribution", "Counts of maintenance flags supplied in the dataset.")
        flag = src["Maintenance_Flag"].fillna("Unknown").astype(str).value_counts().rename_axis("Flag").reset_index(name="Records")
        fig = px.bar(flag, x="Flag", y="Records", color="Flag")
        st.plotly_chart(style(fig, 280), use_container_width=True)


# ---------- Page: Forecast & What-If ----------
elif page == "Forecast & What-If":
    heading("04 / Planning", "Forecast & what-if analysis", "A transparent short-horizon estimate and scenario calculations based on the active dataset.")
    if not forecast.empty:
        heading("Forecast", "14-day energy outlook", "A simple trend extrapolation from recent daily totals. This is a planning estimate, not a validated production forecast.")
        if "Timestamp" in forecast:
            fig = px.line(forecast, x="Timestamp", y="Forecast_Energy_kWh", markers=True, color_discrete_sequence=[TEAL])
            fig.update_xaxes(title="Forecast date")
        else:
            fig = px.line(forecast, x="Forecast_Horizon_Day", y="Forecast_Energy_kWh", markers=True, color_discrete_sequence=[TEAL])
            fig.update_xaxes(title="Forecast horizon day")
        fig.update_yaxes(title="Forecast energy (kWh)")
        st.plotly_chart(style(fig, 340), use_container_width=True)
        c = st.columns(3)
        c[0].metric("Forecast horizon", f"{len(forecast)} days")
        c[1].metric("Forecast total", f"{fmt(forecast['Forecast_Energy_kWh'].sum())} kWh")
        c[2].metric("Daily average", f"{fmt(forecast['Forecast_Energy_kWh'].mean())} kWh")
    else:
        st.info("Forecast requires an energy field and at least two usable daily or sequential observations.")
    if not scenarios.empty:
        heading("Scenario lab", "Potential reduction scenarios", "Illustrative percentage reductions applied to observed energy or idle energy. These are assumptions, not achieved savings.")
        fig = px.bar(scenarios, x="Potential_Energy_Reduction_kWh", y="Scenario", orientation="h", text_auto=".2s", color="Potential_Energy_Reduction_kWh", color_continuous_scale=["#1B4965", GREEN])
        fig.update_layout(coloraxis_showscale=False); fig.update_xaxes(title="Potential energy reduction (kWh)"); fig.update_yaxes(title=None)
        st.plotly_chart(style(fig, 320), use_container_width=True)
        show_table(scenarios.round(3), 260)
        st.download_button("Download scenario analysis", scenarios.to_csv(index=False).encode("utf-8"), "what_if_scenarios.csv", "text/csv")
    else:
        st.info("Scenario analysis requires a usable energy field.")


# ---------- Page: CO2 & Optimization ----------
elif page == "CO₂ & Optimization":
    heading("05 / Sustainability", "CO₂ & optimization", "Estimate carbon impact and compare transparent, assumption-based optimization opportunities.")
    c = st.columns(4)
    c[0].metric("Total CO₂e", f"{fmt(metrics['total_co2'])} kg" if np.isfinite(metrics["total_co2"]) else "Unavailable")
    c[1].metric("CO₂e per unit", f"{fmt(metrics['co2_intensity'], 3)} kg/unit" if np.isfinite(metrics["co2_intensity"]) else "Unavailable")
    c[2].metric("Energy consumed", f"{fmt(metrics['total_energy'])} kWh" if metrics["has_energy"] else "Unavailable")
    c[3].metric("Emission source", "Dataset" if "Dataset-provided" in metrics["co2_source"] else "Estimated" if np.isfinite(metrics["total_co2"]) else "Unavailable")
    st.caption(f"Carbon calculation: {metrics['co2_source']}. If CO₂e is estimated, the default demonstration factor is 0.7 kg CO₂e/kWh; change this assumption in the code before using a different region or electricity mix.")
    if "CO2e_kg" in src and "Machine_Name" in src:
        carbon = src.groupby("Machine_Name")["CO2e_kg"].sum().reset_index()
        fig = px.bar(carbon.sort_values("CO2e_kg"), x="CO2e_kg", y="Machine_Name", orientation="h", text_auto=".2s", color="CO2e_kg", color_continuous_scale=["#1B4965", GREEN])
        fig.update_layout(coloraxis_showscale=False); fig.update_xaxes(title="CO₂e (kg)"); fig.update_yaxes(title=None)
        st.plotly_chart(style(fig, 320), use_container_width=True)
    elif np.isfinite(metrics["total_co2"]):
        st.info("The dataset has no machine-level CO₂e column. Total emissions are available, but machine-level carbon allocation is not shown.")
    if not ranking.empty:
        heading("Priority", "Energy-intensity attention ranking", "Machines are ranked by observed energy per production unit; this is a screening indicator, not an audited sustainability score.")
        fig = px.bar(ranking, x="Sustainability_Attention_Score", y="Machine_Name", orientation="h", text_auto=".1f", color="Sustainability_Attention_Score", color_continuous_scale=["#1B4965", AMBER, RED])
        fig.update_layout(coloraxis_showscale=False); fig.update_xaxes(title="Relative attention score"); fig.update_yaxes(title=None)
        st.plotly_chart(style(fig, 310), use_container_width=True)
    if not optimization.empty:
        heading("Optimization", "Illustrative machine-level opportunity", "A 5% reduction scenario is applied uniformly to each machine's observed energy. It is a what-if assumption, not an optimized solution.")
        fig = px.bar(optimization.sort_values("Optimized_Total_Reduction_kWh"), x="Optimized_Total_Reduction_kWh", y="Machine_Name", orientation="h", text_auto=".2s", color="Optimized_Total_Reduction_kWh", color_continuous_scale=["#1B4965", GREEN])
        fig.update_layout(coloraxis_showscale=False); fig.update_xaxes(title="Illustrative reduction (kWh)"); fig.update_yaxes(title=None)
        st.plotly_chart(style(fig, 320), use_container_width=True)
        show_table(optimization.round(3), 280)
    else:
        st.info("Machine-level optimization requires machine and energy fields.")


# ---------- Page: Recommendations ----------
elif page == "Recommendations":
    heading("06 / Action", "Recommendations & implementation roadmap", "Practical investigation priorities generated from the current dataset. Recommendations are rule-based and should be validated by operations teams.")
    recs = []
    if metrics["has_energy"] and not machine.empty:
        top = machine.iloc[0]
        recs.append(("Energy concentration", f"Review {top['Machine_Name']}", f"This machine has the highest recorded energy total ({fmt(top['Total_Energy_kWh'])} kWh). Check its operating schedule, load and maintenance history."))
    if np.isfinite(metrics["idle_total"]) and metrics["idle_total"] > 0:
        recs.append(("Idle consumption", "Review standby practices", f"The dataset records {fmt(metrics['idle_total'])} kWh of idle energy. Validate shutdown and standby procedures before estimating savings."))
    if metrics["has_energy"] and metrics["has_prod"] and np.isfinite(metrics["intensity"]):
        recs.append(("Energy productivity", "Track energy per unit", f"Overall energy intensity is {fmt(metrics['intensity'], 3)} kWh per unit. Compare like-for-like products and shifts before drawing conclusions."))
    if not state.empty:
        common_state = state.iloc[0]["Actual_State"]
        recs.append(("Operating states", f"Review {common_state} observations", "Check whether state distribution matches the planned production schedule."))
    if not recs:
        recs.append(("Data readiness", "Map core manufacturing fields", "Map energy, production and machine columns on the Dataset Upload & Analysis page to unlock operational recommendations."))
    for i in range(0, len(recs), 2):
        cols = st.columns(2)
        for j, item in enumerate(recs[i:i+2]):
            with cols[j]:
                show_insight(*item)
    st.markdown("---")
    heading("Delivery plan", "Suggested implementation sequence", "A practical order for validating and acting on analytical findings.")
    roadmap = pd.DataFrame([
        {"Phase": "1 · Validate", "Focus": "Data quality", "Actions": "Confirm units, timestamps, machine identifiers, missing values and measurement boundaries."},
        {"Phase": "2 · Investigate", "Focus": "Operational causes", "Actions": "Review high-energy assets, idle periods, shift patterns and maintenance records."},
        {"Phase": "3 · Pilot", "Focus": "Controlled intervention", "Actions": "Test one operational change and compare energy per unit against a comparable baseline."},
        {"Phase": "4 · Monitor", "Focus": "Sustained performance", "Actions": "Track energy, output and emissions consistently; revise assumptions with measured results."},
    ])
    show_table(roadmap, 260)
    st.download_button("Download recommendations", pd.DataFrame(recs, columns=["Category", "Finding", "Action"]).to_csv(index=False).encode("utf-8"), "recommendations.csv", "text/csv")


# ---------- Page: About ----------
elif page == "About the Project":
    heading("Project profile", "Sustainable Manufacturing Analytics", "Energy Consumption & Carbon Footprint Optimization Analytics")
    st.markdown('<div class="hero"><div class="eyebrow">BCA · Data analytics project</div><div class="hero-title">From operational data to sustainability decisions.</div><p class="hero-copy">A dataset-driven workflow for data preparation, energy baselines, machine analysis, forecasting, scenario modelling, carbon estimation and operational recommendations.</p></div>', unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        st.markdown("#### Active dataset")
        st.write(source_label)
        st.metric("Records", f"{len(raw):,}" if not raw.empty else "0")
        st.metric("Columns", f"{len(raw.columns):,}" if not raw.empty else "0")
        st.markdown("#### Technology stack")
        st.write("Python · Pandas · NumPy · Plotly · Streamlit")
    with right:
        st.markdown("#### Interpretation notes")
        st.warning("Forecasts and scenario reductions are modelled estimates. They are not guaranteed savings or validated forecasts.")
        st.info("CO₂e uses the dataset's emissions field when available. Otherwise, the app estimates emissions using a configurable demonstration factor of 0.7 kg CO₂e/kWh.")
    st.markdown("#### Active field mapping")
    mapping_view = pd.DataFrame([
        {"Analytical field": FIELD_LABELS[k], "Source column": mapping.get(k) or "Not mapped"}
        for k in ALIASES
    ])
    show_table(mapping_view, 360)
    if st.button("Refresh application data", type="primary"):
        st.cache_data.clear()
        st.rerun()

st.markdown('<div style="color:#75869E;font-size:.75rem;border-top:1px solid rgba(148,163,184,.18);padding-top:14px;margin-top:30px">SUSTAINABLE MANUFACTURING ANALYTICS · Dataset-driven dashboard · Modelled opportunities are not achieved savings.</div>', unsafe_allow_html=True)
