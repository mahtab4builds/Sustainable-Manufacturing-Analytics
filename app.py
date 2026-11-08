from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs'
DATA = ROOT / 'data' / 'Sustainable_Manufacturing_Energy_Analytics_10000.xlsx'

st.set_page_config(page_title='Sustainability Intelligence | Manufacturing Analytics', layout='wide', initial_sidebar_state='expanded')

BG, PANEL, TEXT, MUTED, GREEN, TEAL, BLUE, AMBER, RED = '#0B1220', '#111C2E', '#E8EEF7', '#9BAAC0', '#38D39F', '#42C6C5', '#6AA9FF', '#F6BD60', '#FF7B84'
BORDER = 'rgba(148,163,184,.18)'
st.markdown(f'''<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root {{ color-scheme: dark; }}
html,body,[class*="css"]{{font-family:'DM Sans',sans-serif}}
.stApp{{background:radial-gradient(ellipse at 12% 0%,rgba(56,211,159,.09),transparent 32%),radial-gradient(ellipse at 90% 3%,rgba(66,198,197,.065),transparent 28%),{BG};color:{TEXT}}}
[data-testid="stHeader"]{{background:rgba(11,18,32,.76);backdrop-filter:blur(14px)}}
[data-testid="stSidebar"]{{background:linear-gradient(180deg,#101D30 0%,#0B1220 76%);border-right:1px solid {BORDER};box-shadow:10px 0 35px rgba(0,0,0,.12)}}
[data-testid="stSidebar"] *{{color:{TEXT}}}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p{{color:{MUTED}}}
.block-container{{padding-top:1.6rem;padding-bottom:2.8rem;max-width:1540px;animation:pageEnter .55s ease both}}
h1,h2,h3{{font-family:'Manrope',sans-serif;letter-spacing:-.035em;color:{TEXT}}} p,label{{color:{MUTED}}}
/* Sidebar navigation: spacious, pill-shaped and responsive to hover/focus. */
[data-testid="stSidebar"] [data-testid="stRadio"] > label{{font-size:.68rem;font-weight:800;letter-spacing:.15em;color:#7F91AA;text-transform:uppercase;margin:0 0 10px 5px}}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"]{{gap:5px}}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label{{position:relative;display:flex;align-items:center;padding:10px 12px;border:1px solid transparent;border-radius:12px;background:rgba(255,255,255,.018);transition:background .22s ease,border-color .22s ease,transform .22s ease,box-shadow .22s ease;cursor:pointer;min-height:42px}}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label:hover{{background:rgba(56,211,159,.075);border-color:rgba(56,211,159,.16);transform:translateX(3px)}}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked){{background:linear-gradient(100deg,rgba(56,211,159,.16),rgba(66,198,197,.06));border-color:rgba(56,211,159,.28);box-shadow:inset 3px 0 0 {GREEN},0 5px 16px rgba(0,0,0,.08)}}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label p{{font-size:.88rem;font-weight:600;color:#B9C7D8;transition:color .2s ease}}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked) p{{color:#F4FFF9}}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] input{{accent-color:{GREEN}}}
/* KPI cards: subtle accent, clean hierarchy, small lift on hover. */
[data-testid="stMetric"]{{position:relative;overflow:hidden;background:linear-gradient(145deg,rgba(24,40,64,.98),rgba(15,27,45,.98));border:1px solid rgba(148,163,184,.17);border-radius:17px;padding:18px 19px;min-height:116px;box-shadow:0 10px 28px rgba(0,0,0,.13);transition:transform .22s ease,border-color .22s ease,box-shadow .22s ease}}
[data-testid="stMetric"]:before{{content:"";position:absolute;left:0;top:0;width:100%;height:2px;background:linear-gradient(90deg,{GREEN},{TEAL},transparent);opacity:.8}}
[data-testid="stMetric"]:hover{{transform:translateY(-3px);border-color:rgba(56,211,159,.34);box-shadow:0 15px 32px rgba(0,0,0,.22)}}
[data-testid="stMetricLabel"] p{{color:{MUTED}!important;font-size:.79rem;font-weight:600;letter-spacing:.015em}}
[data-testid="stMetricValue"]{{color:{TEXT}!important;font-family:'Manrope',sans-serif;font-weight:800;letter-spacing:-.035em}}
[data-testid="stMetricDelta"]{{font-size:.77rem}}
div[data-testid="stPlotlyChart"]{{background:linear-gradient(145deg,rgba(17,28,46,.8),rgba(13,23,39,.74));border:1px solid rgba(148,163,184,.16);border-radius:17px;padding:8px;box-shadow:0 8px 26px rgba(0,0,0,.08);transition:border-color .2s ease,transform .2s ease}}
div[data-testid="stPlotlyChart"]:hover{{border-color:rgba(66,198,197,.3);transform:translateY(-1px)}}
.hero{{background:linear-gradient(115deg,rgba(20,43,61,.98),rgba(17,28,46,.96) 58%,rgba(20,56,56,.82));border:1px solid rgba(66,198,197,.26);border-radius:24px;padding:30px 32px;margin-bottom:24px;position:relative;overflow:hidden;box-shadow:0 18px 44px rgba(0,0,0,.14);animation:heroEnter .7s cubic-bezier(.2,.7,.2,1) both}}
.hero:after{{content:"";position:absolute;width:230px;height:230px;right:-60px;top:-105px;border:1px solid rgba(56,211,159,.24);border-radius:50%;box-shadow:0 0 0 28px rgba(56,211,159,.025),0 0 0 58px rgba(56,211,159,.018);pointer-events:none}}
.eyebrow,.kicker{{color:{GREEN};font-size:.7rem;font-weight:800;letter-spacing:.16em;text-transform:uppercase;margin-bottom:8px}}
.hero-title{{font-family:'Manrope',sans-serif;font-weight:800;letter-spacing:-.05em;color:{TEXT};font-size:clamp(1.9rem,3vw,2.8rem);line-height:1.12;margin:0 0 12px}}
.hero-copy{{color:#B6C5D8;font-size:1rem;line-height:1.7;max-width:850px;margin:0}}
.section-title{{font-family:'Manrope',sans-serif;font-size:1.42rem;font-weight:800;letter-spacing:-.035em;color:{TEXT};margin:0 0 4px}}
.subtitle{{color:{MUTED};font-size:.9rem;line-height:1.55;margin-bottom:15px}}
.insight{{background:linear-gradient(145deg,rgba(22,38,60,.96),rgba(15,27,45,.96));border:1px solid rgba(148,163,184,.16);border-radius:16px;padding:18px 19px;height:100%;min-height:195px;margin-bottom:20px;box-shadow:0 8px 22px rgba(0,0,0,.08);transition:transform .22s ease,border-color .22s ease,box-shadow .22s ease}}
.insight:hover{{transform:translateY(-3px);border-color:rgba(56,211,159,.28);box-shadow:0 14px 28px rgba(0,0,0,.16)}}
.insight-label{{color:{GREEN};font-size:.67rem;font-weight:800;letter-spacing:.12em;text-transform:uppercase}}
.insight-title{{color:{TEXT};font-weight:750;margin:8px 0;font-family:'Manrope',sans-serif}}
.insight-copy{{color:{MUTED};font-size:.86rem;line-height:1.6}}
.pill{{display:inline-block;padding:5px 10px;border-radius:99px;background:rgba(56,211,159,.1);color:{GREEN};border:1px solid rgba(56,211,159,.22);font-size:.7rem;font-weight:800;letter-spacing:.04em}}
.footer{{color:#75869E;font-size:.75rem;border-top:1px solid {BORDER};padding-top:14px;margin-top:30px}}
[data-testid="stDataFrame"]{{border:1px solid {BORDER};border-radius:13px;overflow:hidden}}
.stButton button,.stDownloadButton button{{border-radius:11px;font-weight:700;transition:transform .18s ease,box-shadow .18s ease,border-color .18s ease}}
.stButton button:hover,.stDownloadButton button:hover{{transform:translateY(-1px);box-shadow:0 7px 18px rgba(0,0,0,.18)}}
[data-testid="stSelectbox"] [data-baseweb="select"], [data-testid="stMultiSelect"] [data-baseweb="select"]{{border-radius:10px}}
hr{{border-color:rgba(148,163,184,.14)!important}}
@keyframes pageEnter{{from{{opacity:0;transform:translateY(7px)}}to{{opacity:1;transform:translateY(0)}}}}
@keyframes heroEnter{{from{{opacity:0;transform:translateY(10px) scale(.995)}}to{{opacity:1;transform:translateY(0) scale(1)}}}}
@media (prefers-reduced-motion: reduce){{*,*::before,*::after{{animation:none!important;transition:none!important;scroll-behavior:auto!important}}}}
@media (max-width: 768px){{.block-container{{padding-left:1rem;padding-right:1rem}}.hero{{padding:22px 20px;border-radius:18px}}[data-testid="stMetric"]{{padding:14px;min-height:100px}}}}
</style>''',unsafe_allow_html=True)

LAYOUT = dict(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family='DM Sans, sans-serif', color=TEXT, size=12), margin=dict(l=18,r=18,t=42,b=18), legend=dict(orientation='h',yanchor='bottom',y=1.02,xanchor='left',x=0), hoverlabel=dict(bgcolor=PANEL,font_color=TEXT))
GRID = 'rgba(148,163,184,.13)'

@st.cache_data(show_spinner=False)
def csv(path):
    p = OUT / path
    if not p.exists(): return pd.DataFrame()
    try: return pd.read_csv(p)
    except Exception: return pd.DataFrame()

@st.cache_data(show_spinner=False)
def source_data():
    cleaned = csv('cleaned_data/manufacturing_energy_cleaned.csv')
    if not cleaned.empty: return cleaned
    if DATA.exists():
        try: return pd.read_excel(DATA)
        except Exception: return pd.DataFrame()
    return pd.DataFrame()

def cols(df, required): return not df.empty and all(c in df.columns for c in required)
def num(v, d=0):
    try: return '—' if pd.isna(v) else f'{float(v):,.{d}f}'
    except (TypeError,ValueError): return '—'
def kpi(df, keycol, key, valuecol='Value', default=np.nan):
    if cols(df,[keycol,valuecol]):
        vals=df.loc[df[keycol].astype(str).str.casefold().eq(key.casefold()),valuecol]
        if not vals.empty: return pd.to_numeric(vals.iloc[0],errors='coerce')
    return default
def style(fig, height=350):
    fig.update_layout(**LAYOUT,height=height)
    fig.update_xaxes(showgrid=False,linecolor=GRID,zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID,zerolinecolor=GRID)
    return fig
def heading(kicker, title, subtitle):
    st.markdown(f'<div class="kicker">{kicker}</div><div class="section-title">{title}</div><div class="subtitle">{subtitle}</div>',unsafe_allow_html=True)
def table(df,height=300):
    if df.empty: st.info('No data is available for this view yet. Run the existing analytics pipeline and refresh this page.')
    else: st.dataframe(df,use_container_width=True,hide_index=True,height=height)
def download(df,name):
    if not df.empty: st.download_button('Download CSV',df.to_csv(index=False).encode('utf-8'),name,'text/csv')
def insight(label,title,copy):
    st.markdown(f'<div class="insight"><div class="insight-label">{label}</div><div class="insight-title">{title}</div><div class="insight-copy">{copy}</div></div>',unsafe_allow_html=True)

# Read the generated outputs. The dashboard is intentionally independent of the mock API.
src = source_data()
if not src.empty:
    if 'Timestamp' in src: src['Timestamp'] = pd.to_datetime(src['Timestamp'],errors='coerce')
    elif 'Date' in src: src['Timestamp'] = pd.to_datetime(src['Date'],errors='coerce')
    for c in ['Energy_Consumption_kWh','Actual_Production_Units']: 
        if c in src: src[c] = pd.to_numeric(src[c],errors='coerce')
exec_kpi = csv('recommendations/executive_kpi_summary.csv')
baseline = csv('baseline/baseline_summary.csv')
sust_kpi = csv('co2_analysis/sustainability_kpis.csv')
machine = csv('machine_analysis.csv')
state = csv('machine_state_analysis.csv')
hourly = csv('hourly_load_profile.csv')
shift = csv('shift_analysis.csv')
idle = csv('idle_energy_analysis.csv')
scenarios = csv('what_if/scenario_comparison.csv')
forecast = csv('forecasting/future_energy_forecast.csv')
forecast_metrics = csv('forecasting/forecast_metrics.csv')
optimization = csv('optimization/optimization_opportunities.csv')
ranking = csv('co2_analysis/machine_sustainability_ranking.csv')
insights = csv('recommendations/executive_insights.csv')
recommendations = csv('recommendations/management_recommendations.csv')
roadmap = csv('recommendations/implementation_roadmap.csv')
actions = csv('recommendations/machine_action_plan.csv')

energy = kpi(exec_kpi,'Metric','Total Energy Consumption',default=349856.35)
production = kpi(exec_kpi,'Metric','Total Production',default=52491.21)
intensity = kpi(exec_kpi,'Metric','Energy per Production Unit',default=6.665046)
idle_energy = kpi(exec_kpi,'Metric','Idle Energy',default=22120.85)
co2 = kpi(exec_kpi,'Metric','Total CO2e',default=244899.445)
co2_intensity = kpi(exec_kpi,'Metric','CO2e per Production Unit',default=4.665532)
whatif_rows = exec_kpi[exec_kpi.get('Metric',pd.Series(dtype=str)).astype(str).eq('Potential Combined Energy Reduction')] if not exec_kpi.empty else pd.DataFrame()
whatif_energy = pd.to_numeric(whatif_rows['Value'].iloc[0],errors='coerce') if len(whatif_rows)>0 else np.nan
whatif_pct = pd.to_numeric(whatif_rows['Value'].iloc[1],errors='coerce') if len(whatif_rows)>1 else np.nan
opt_energy = kpi(exec_kpi,'Metric','Potential Energy Reduction')
opt_pct = kpi(exec_kpi,'Metric','Potential Reduction')

NAV = ['Overview','Energy Performance','Machine Efficiency','Operations','Forecast & What-If','CO₂ & Optimization','Recommendations','About the Project']
with st.sidebar:
    st.markdown(f'<div style="padding:8px 2px 20px"><div style="display:flex;align-items:center;gap:9px;margin-bottom:13px"><span style="display:inline-flex;width:34px;height:34px;align-items:center;justify-content:center;border-radius:11px;background:linear-gradient(135deg,rgba(56,211,159,.22),rgba(66,198,197,.1));border:1px solid rgba(56,211,159,.28);color:{GREEN};font-size:18px">✦</span><span style="font-size:.68rem;letter-spacing:.15em;font-weight:800;color:{GREEN}">SUSTAINABILITY INTELLIGENCE</span></div><div style="font-family:Manrope,sans-serif;font-size:1.22rem;font-weight:800;line-height:1.2;margin-top:7px;color:{TEXT}">Manufacturing<br>Analytics</div><div style="font-size:.77rem;color:{MUTED};margin-top:9px">Energy · Efficiency · Carbon</div></div>',unsafe_allow_html=True)
    st.markdown('---')
    page = st.radio('NAVIGATION',NAV,label_visibility='visible',format_func=lambda item: {'Overview':'◈  Overview','Energy Performance':'ϟ  Energy Performance','Machine Efficiency':'▦  Machine Efficiency','Operations':'⌁  Operations','Forecast & What-If':'↗  Forecast & What-If','CO₂ & Optimization':'♧  CO₂ & Optimization','Recommendations':'✳  Recommendations','About the Project':'ⓘ  About the Project'}[item])
    st.markdown('---')
    st.markdown('**PROJECT STATUS**')
    st.markdown(f'<span class="pill">{"DATA CONNECTED" if not src.empty else "OUTPUTS NOT FOUND"}</span>',unsafe_allow_html=True)
    st.caption(f'{len(src):,} source observations' if not src.empty else 'Place this app beside the outputs/ folder.')
    st.caption('Synthetic demonstration dataset · 5 machines')
    st.markdown('---')
    st.caption('Python · Streamlit · Plotly')

if page == 'Overview':
    st.markdown(f'''<div class="hero"><div class="eyebrow">Sustainability intelligence · Manufacturing analytics</div><div class="hero-title">Make every kilowatt count.</div><p class="hero-copy">A unified view of manufacturing energy, machine performance and carbon impact — turning operational data into practical opportunities for more efficient production.</p></div>''',unsafe_allow_html=True)
    heading('Executive snapshot','Performance at a glance','A concise view of energy use, production and sustainability impact across the manufacturing dataset.')
    c=st.columns(4)
    c[0].metric('Total energy',f'{num(energy)} kWh'); c[1].metric('Production output',f'{num(production)} units'); c[2].metric('Energy intensity',f'{num(intensity,2)} kWh / unit'); c[3].metric('CO₂e emissions',f'{num(co2)} kg')
    c=st.columns(4)
    c[0].metric('Idle energy',f'{num(idle_energy)} kWh'); c[1].metric('CO₂e intensity',f'{num(co2_intensity,2)} kg / unit'); c[2].metric('What-if energy opportunity',f'{num(whatif_energy)} kWh',help='Modelled scenario estimate, not measured savings.'); c[3].metric('Optimization opportunity',f'{num(opt_energy)} kWh',help='Modelled optimization potential, not achieved savings.')
    left,right=st.columns([1.5,1])
    with left:
        heading('Energy profile','Consumption over time','Daily energy consumption from the cleaned dataset.')
        if cols(src,['Timestamp','Energy_Consumption_kWh']):
            d=src.dropna(subset=['Timestamp']).assign(Date=lambda x:x.Timestamp.dt.date).groupby('Date',as_index=False)['Energy_Consumption_kWh'].sum(); d.columns=['Date','Energy (kWh)']
            f=px.area(d,x='Date',y='Energy (kWh)',color_discrete_sequence=[GREEN]); f.update_traces(line=dict(width=2),fillcolor='rgba(56,211,159,.12)'); st.plotly_chart(style(f,340),use_container_width=True)
        else: st.info('Daily trend is unavailable because the cleaned dataset is missing.')
    with right:
        heading('Machine mix','Energy by machine','Which assets account for the largest share of energy?')
        if cols(machine,['Machine_Name','Total_Energy_kWh']):
            d=machine.sort_values('Total_Energy_kWh'); f=px.bar(d,x='Total_Energy_kWh',y='Machine_Name',orientation='h',text_auto='.2s',color='Total_Energy_kWh',color_continuous_scale=['#1B4965',GREEN]); f.update_layout(coloraxis_showscale=False); f.update_yaxes(title=None); f.update_xaxes(title='Energy (kWh)'); st.plotly_chart(style(f,340),use_container_width=True)
        else: st.info('Machine analysis output is unavailable.')
    heading('Decision support','What deserves attention?','Priorities are generated from the project outputs and should guide investigation.')
    a,b,c=st.columns(3)
    if cols(machine,['Machine_Name','Total_Energy_kWh']):
        top=machine.loc[machine.Total_Energy_kWh.idxmax()]; name=str(top.Machine_Name); top_e=num(top.Total_Energy_kWh)
    else: name,top_e='Machine data unavailable','—'
    with a: insight('Highest energy use',name,f'{top_e} kWh recorded. Review operating profile and efficiency before choosing an intervention.')
    with b: insight('Idle energy',f'{num(idle_energy)} kWh','Review standby behavior, shift transitions and shutdown practices. Reductions shown elsewhere are modelled opportunities.')
    with c: insight('Carbon intensity',f'{num(co2_intensity,2)} kg CO₂e / unit','Emissions use the project’s synthetic demonstration factor, not an audited carbon inventory.')

elif page == 'Energy Performance':
    heading('01 / Energy','Energy performance','Explore energy use, baselines, intensity and operating patterns.')
    filtered=src.copy()
    if not filtered.empty and 'Timestamp' in filtered and filtered.Timestamp.notna().any():
        valid=filtered.Timestamp.dropna(); dates=st.date_input('Analysis date range',value=(valid.min().date(),valid.max().date()),min_value=valid.min().date(),max_value=valid.max().date())
        if isinstance(dates,tuple) and len(dates)==2: filtered=filtered[filtered.Timestamp.dt.date.between(dates[0],dates[1])]
    if not filtered.empty and 'Machine_Name' in filtered:
        opts=sorted(filtered.Machine_Name.dropna().astype(str).unique()); chosen=st.multiselect('Machines',opts,default=opts); filtered=filtered[filtered.Machine_Name.astype(str).isin(chosen)]
    if cols(filtered,['Timestamp','Energy_Consumption_kWh']):
        d=filtered.dropna(subset=['Timestamp']).assign(Date=lambda x:x.Timestamp.dt.date).groupby('Date',as_index=False)['Energy_Consumption_kWh'].sum(); d.columns=['Date','Energy (kWh)']; f=px.line(d,x='Date',y='Energy (kWh)',color_discrete_sequence=[GREEN]); f.update_traces(line=dict(width=2.5)); st.plotly_chart(style(f,360),use_container_width=True)
    else: st.info('Energy trend cannot be displayed because required source columns are unavailable.')
    l,r=st.columns(2)
    with l:
        heading('Baseline','Observed vs calculated baseline','Baseline definitions differ across project modules; this view uses the baseline_summary output.')
        if cols(baseline,['Metric','Value']):
            d=baseline[baseline.Metric.isin(['Actual Energy (kWh)','Calculated Baseline Energy (kWh)','Energy Deviation (kWh)'])].copy(); d.Value=pd.to_numeric(d.Value,errors='coerce'); f=px.bar(d,x='Metric',y='Value',text_auto='.3s',color='Metric',color_discrete_sequence=[GREEN,BLUE,AMBER]); f.update_layout(showlegend=False); f.update_xaxes(title=None,tickangle=-12); f.update_yaxes(title='Energy (kWh)'); st.plotly_chart(style(f,320),use_container_width=True)
        else: st.info('Baseline summary is unavailable.')
    with r:
        heading('Energy intensity','Energy per production unit','Compare energy intensity by machine.')
        if cols(machine,['Machine_Name','Energy_Per_Unit_kWh']):
            f=px.bar(machine.sort_values('Energy_Per_Unit_kWh'),x='Energy_Per_Unit_kWh',y='Machine_Name',orientation='h',text_auto='.2f',color='Energy_Per_Unit_kWh',color_continuous_scale=['#1B4965',GREEN]); f.update_layout(coloraxis_showscale=False); f.update_xaxes(title='kWh per unit'); f.update_yaxes(title=None); st.plotly_chart(style(f,320),use_container_width=True)
        else: st.info('Machine intensity output is unavailable.')
    if cols(hourly,['Hour','Average_Energy_kWh']):
        heading('Load profile','Average energy by hour','Use the daily profile to identify periods worth investigating.')
        f=px.bar(hourly,x='Hour',y='Average_Energy_kWh',color_discrete_sequence=[TEAL],hover_data=[c for c in ['Minimum_Energy_kWh','Maximum_Energy_kWh'] if c in hourly]); f.update_xaxes(title='Hour of day'); f.update_yaxes(title='Average energy (kWh)'); st.plotly_chart(style(f,310),use_container_width=True)
    if not filtered.empty: st.download_button('Download filtered source data',filtered.to_csv(index=False).encode(), 'energy_performance_filtered.csv','text/csv')

elif page == 'Machine Efficiency':
    heading('02 / Assets','Machine efficiency','Compare energy demand, output, idle consumption and sustainability priority across machines.')
    if cols(machine,['Machine_Name','Total_Energy_kWh','Total_Production_Units']):
        l,r=st.columns(2)
        with l:
            f=px.bar(machine.sort_values('Total_Energy_kWh'),x='Total_Energy_kWh',y='Machine_Name',orientation='h',text_auto='.2s',color='Total_Energy_kWh',color_continuous_scale=['#1B4965',GREEN]); f.update_layout(coloraxis_showscale=False); f.update_xaxes(title='Total energy (kWh)'); f.update_yaxes(title=None); st.plotly_chart(style(f,340),use_container_width=True)
        with r:
            f=px.scatter(machine,x='Total_Production_Units',y='Total_Energy_kWh',size='Total_Idle_Energy_kWh' if 'Total_Idle_Energy_kWh' in machine else None,color='Machine_Name',hover_name='Machine_Name',text='Machine_ID' if 'Machine_ID' in machine else None); f.update_traces(textposition='top center'); f.update_xaxes(title='Production (units)'); f.update_yaxes(title='Energy (kWh)'); st.plotly_chart(style(f,340),use_container_width=True)
        if cols(machine,['Machine_Name','Energy_Per_Unit_kWh']):
            f=px.bar(machine.sort_values('Energy_Per_Unit_kWh'),x='Energy_Per_Unit_kWh',y='Machine_Name',orientation='h',text_auto='.2f',color='Energy_Per_Unit_kWh',color_continuous_scale=['#1B4965',AMBER]); f.update_layout(coloraxis_showscale=False); f.update_xaxes(title='Energy intensity (kWh/unit)'); f.update_yaxes(title=None); st.plotly_chart(style(f,320),use_container_width=True)
        show=[c for c in ['Machine_ID','Machine_Name','Machine_Type','Total_Energy_kWh','Total_Production_Units','Energy_Per_Unit_kWh','Total_Idle_Energy_kWh','Running_Hours','Idle_Hours'] if c in machine.columns]; table(machine[show].round(2)); st.download_button('Download machine register',machine[show].to_csv(index=False).encode(),'machine_efficiency.csv','text/csv')
    else: st.warning('Machine-level analytics are missing. Check outputs/machine_analysis.csv.')

elif page == 'Operations':
    heading('03 / Operations','Operational intelligence','Understand machine states, shift patterns, idle energy and high-consumption events.')
    l,r=st.columns(2)
    with l:
        heading('Machine states','Operating-state mix','Share of observations classified by state.')
        if cols(state,['Actual_State','Observations']):
            f=px.pie(state,names='Actual_State',values='Observations',hole=.68,color_discrete_sequence=[GREEN,AMBER,BLUE,RED]); f.update_traces(textposition='outside',textinfo='label+percent'); f.update_layout(showlegend=False); st.plotly_chart(style(f,330),use_container_width=True)
        else: st.info('Machine state summary is unavailable.')
    with r:
        heading('Shift comparison','Energy intensity by shift','Compare energy per unit across shifts.')
        if cols(shift,['Shift','Energy_Per_Unit_kWh']):
            f=px.bar(shift,x='Shift',y='Energy_Per_Unit_kWh',color='Shift',text_auto='.2f',color_discrete_sequence=[GREEN,TEAL,BLUE]); f.update_layout(showlegend=False); f.update_yaxes(title='kWh per unit'); st.plotly_chart(style(f,330),use_container_width=True)
        else: st.info('Shift analysis is unavailable.')
    if cols(idle,['Machine_ID','Idle_Energy_kWh']):
        heading('Idle energy','Where idle consumption accumulates','Prioritize investigation based on measured idle energy, not only percentage.')
        d=idle.sort_values('Idle_Energy_kWh'); f=px.bar(d,x='Idle_Energy_kWh',y='Machine_ID',orientation='h',text_auto='.2s',color='Idle_Energy_Percentage' if 'Idle_Energy_Percentage' in d else 'Idle_Energy_kWh',color_continuous_scale=['#1B4965',AMBER]); f.update_layout(coloraxis_showscale=False); f.update_xaxes(title='Idle energy (kWh)'); f.update_yaxes(title=None); st.plotly_chart(style(f,310),use_container_width=True); table(idle.round(2),240)
    events=csv('potential_high_consumption_events.csv')
    if not events.empty:
        heading('Event review','Potential high-consumption observations','Analytical candidates for review, not confirmed equipment faults.'); st.caption(f'{len(events):,} candidate records are available.'); table(events.head(100),310); st.download_button('Download candidate events',events.to_csv(index=False).encode(),'potential_high_consumption_events.csv','text/csv')

elif page == 'Forecast & What-If':
    heading('04 / Planning','Forecast & what-if analysis','Explore the 14-day forecast and compare modelled intervention scenarios.')
    if cols(forecast,['Timestamp','Forecast_Energy_kWh']):
        fdata=forecast.copy(); fdata['Timestamp']=pd.to_datetime(fdata.Timestamp,errors='coerce'); fdata['Forecast_Energy_kWh']=pd.to_numeric(fdata.Forecast_Energy_kWh,errors='coerce')
        heading('Forecast','14-day energy forecast','Forecast output from the existing pipeline; predictions are estimates, not guarantees.')
        f=px.line(fdata,x='Timestamp',y='Forecast_Energy_kWh',markers=True,color_discrete_sequence=[TEAL],hover_data=['Forecast_Horizon_Day'] if 'Forecast_Horizon_Day' in fdata else None); f.update_traces(line=dict(width=2.5),marker=dict(size=6)); f.update_xaxes(title='Forecast date'); f.update_yaxes(title='Forecast energy (kWh)'); st.plotly_chart(style(f,340),use_container_width=True)
        c=st.columns(3); c[0].metric('Forecast horizon',f'{len(fdata)} days'); c[1].metric('Forecast total',f'{num(fdata.Forecast_Energy_kWh.sum())} kWh'); c[2].metric('Daily average',f'{num(fdata.Forecast_Energy_kWh.mean())} kWh')
    else: st.info('Forecast output is unavailable. Run the existing forecasting phase to generate it.')
    if cols(forecast_metrics,['Model','MAE_kWh']):
        heading('Model evaluation','Forecast validation metrics','Historical test metrics help compare models; they do not guarantee future accuracy.')
        fm=forecast_metrics.sort_values('MAE_kWh'); f=px.bar(fm,x='MAE_kWh',y='Model',orientation='h',text_auto='.1f',color='MAE_kWh',color_continuous_scale=['#1B4965',GREEN],hover_data=[c for c in ['RMSE_kWh','MAPE_Percent','Rank'] if c in fm]); f.update_layout(coloraxis_showscale=False); f.update_xaxes(title='MAE (kWh)'); f.update_yaxes(title=None); st.plotly_chart(style(f,270),use_container_width=True); table(fm.round(3),220)
    if cols(scenarios,['Scenario','Potential_Energy_Reduction_kWh']):
        heading('Scenario lab','Potential savings by scenario','All reductions are modelled estimates; review assumptions before interpreting them as achievable savings.')
        d=scenarios.copy(); d['Scenario_Label']=d.Scenario.astype(str).str.replace('_',' ').str.title(); f=px.bar(d.sort_values('Potential_Energy_Reduction_kWh'),x='Potential_Energy_Reduction_kWh',y='Scenario_Label',orientation='h',text_auto='.2s',color='Potential_Energy_Reduction_kWh',color_continuous_scale=['#1B4965',GREEN],hover_data=[c for c in ['Potential_Energy_Reduction_Percent','Potential_CO2_Reduction_kg','Production_Maintained'] if c in d]); f.update_layout(coloraxis_showscale=False); f.update_xaxes(title='Potential energy reduction (kWh)'); f.update_yaxes(title=None); st.plotly_chart(style(f,320),use_container_width=True)
        show=[c for c in ['Scenario_Label','Actual_Energy_kWh','Scenario_Energy_kWh','Potential_Energy_Reduction_kWh','Potential_Energy_Reduction_Percent','Potential_CO2_Reduction_kg','Production_Maintained'] if c in d]; table(d[show].round(2),270); st.download_button('Download scenario comparison',d[show].to_csv(index=False).encode(),'what_if_scenario_comparison.csv','text/csv')

elif page == 'CO₂ & Optimization':
    heading('05 / Sustainability','CO₂ & optimization','Connect energy performance to estimated emissions and modelled machine-level optimization opportunities.')
    if cols(sust_kpi,['KPI','Value']):
        c=st.columns(4); c[0].metric('Total CO₂e',f"{num(kpi(sust_kpi,'KPI','Total CO2e'))} kg"); c[1].metric('Baseline CO₂e',f"{num(kpi(sust_kpi,'KPI','Baseline CO2e'))} kg"); c[2].metric('Idle CO₂e',f"{num(kpi(sust_kpi,'KPI','Idle CO2e'))} kg"); c[3].metric('Emission factor',f"{num(kpi(sust_kpi,'KPI','Synthetic Emission Factor'),2)} kg/kWh")
        st.caption('The project labels its emission factor as synthetic / demonstration-only. These are estimates, not audited greenhouse-gas inventory figures.')
    l,r=st.columns(2)
    with l:
        heading('Carbon profile','CO₂e by machine','Estimated emissions using the configured emission factor.')
        if cols(machine,['Machine_Name','CO2e_kg']):
            f=px.bar(machine.sort_values('CO2e_kg'),x='CO2e_kg',y='Machine_Name',orientation='h',text_auto='.2s',color='CO2e_kg',color_continuous_scale=['#1B4965',GREEN]); f.update_layout(coloraxis_showscale=False); f.update_xaxes(title='CO₂e (kg)'); f.update_yaxes(title=None); st.plotly_chart(style(f,320),use_container_width=True)
        else: st.info('Machine CO₂ output is unavailable.')
    with r:
        heading('Priority','Sustainability attention ranking','A model-generated score to help direct investigation.')
        if cols(ranking,['Machine_Name','Sustainability_Attention_Score']):
            d=ranking.sort_values('Sustainability_Attention_Score'); f=px.bar(d,x='Sustainability_Attention_Score',y='Machine_Name',orientation='h',text_auto='.1f',color='Sustainability_Attention_Score',color_continuous_scale=['#1B4965',AMBER,RED]); f.update_layout(coloraxis_showscale=False); f.update_xaxes(title='Attention score'); f.update_yaxes(title=None); st.plotly_chart(style(f,320),use_container_width=True)
        else: st.info('Sustainability ranking is unavailable.')
    if cols(optimization,['Machine_Name','Optimized_Total_Reduction_kWh']):
        heading('Optimization','Modelled reduction opportunity','Potential reductions by machine under the project’s optimization assumptions.')
        d=optimization.sort_values('Optimized_Total_Reduction_kWh'); f=px.bar(d,x='Optimized_Total_Reduction_kWh',y='Machine_Name',orientation='h',text_auto='.2s',color='Potential_Reduction_Percent' if 'Potential_Reduction_Percent' in d else 'Optimized_Total_Reduction_kWh',color_continuous_scale=['#1B4965',GREEN],hover_data=[c for c in ['Optimized_Energy_kWh','Potential_Reduction_Percent','Production_Output_Maintained'] if c in d]); f.update_layout(coloraxis_showscale=False); f.update_xaxes(title='Potential reduction (kWh)'); f.update_yaxes(title=None); st.plotly_chart(style(f,330),use_container_width=True); table(d.round(2),280); st.download_button('Download optimization opportunities',d.to_csv(index=False).encode(),'optimization_opportunities.csv','text/csv')

elif page == 'Recommendations':
    heading('06 / Action','Recommendations & implementation roadmap','Translate analysis into a practical sequence of operational investigations.')
    
    if not insights.empty:
        heading('Executive insights','Key findings','Findings generated by the existing recommendation module.')
        for i in range(0, len(insights), 3):
            cols_row = st.columns(3)
            for j, (_, row) in enumerate(insights.iloc[i:i+3].iterrows()):
                with cols_row[j]:
                    insight(str(row.get('Category','Insight')), str(row.get('Finding','Review finding')), f"Reference: {row.get('Insight_ID','—')}")
                    
    if not recommendations.empty:
        heading('Management actions','Recommended interventions','Priority and horizon are taken from the project recommendation output.')
        sortcol='Priority' if 'Priority' in recommendations else recommendations.columns[0]
        for _,row in recommendations.sort_values(sortcol).iterrows():
            with st.container(border=True):
                l,r=st.columns([4,1])
                with l:
                    st.markdown(f"**{row.get('Area','Recommendation')}**"); st.write(row.get('Recommendation',''))
                    if pd.notna(row.get('Rationale',np.nan)): st.caption(str(row.get('Rationale')))
                with r:
                    st.markdown(f"**Priority {row.get('Priority','—')}**"); st.caption(str(row.get('Implementation_Horizon','')))
                    if pd.notna(row.get('Expected_Impact',np.nan)): st.caption(str(row.get('Expected_Impact')))
        st.download_button('Download recommendations',recommendations.to_csv(index=False).encode(),'management_recommendations.csv','text/csv')
        
    if not roadmap.empty:
        heading('Delivery plan','Implementation roadmap','A phased roadmap from monitoring to optimization deployment.')
        for _,row in roadmap.iterrows():
            with st.container(border=True):
                st.markdown(f"**Phase {row.get('Phase','')} · {row.get('Focus','')}**"); st.caption(f"{row.get('Timeline','')} · Owner: {row.get('Primary_Owner','Not specified')}"); st.write(row.get('Actions',''))
        st.download_button('Download roadmap',roadmap.to_csv(index=False).encode(),'implementation_roadmap.csv','text/csv')
        
    if not actions.empty:
        heading('Machine action plan','Machine-level priorities','Priorities and impact values are model-generated outputs.')
        show=[c for c in ['Machine_ID','Machine_Name','Sustainability_Priority','Highest_Priority','Recommendation_Count','Optimized_Total_Reduction_kWh','Total_Potential_CO2e_Impact_kg'] if c in actions]; table(actions[show].round(2),280)

elif page == 'About the Project':
    heading('Project profile','Sustainable Manufacturing Analytics','Energy Consumption & Carbon Footprint Optimization Analytics')
    st.markdown(f'''<div class="hero"><div class="eyebrow">BCA · Data analytics project</div><div class="hero-title">From operational data to sustainability decisions.</div><p class="hero-copy">This project combines data preparation, exploratory analysis, energy baselines, machine-state analysis, forecasting, scenario modelling, carbon estimation, optimization and management recommendations in one analytical workflow.</p></div>''',unsafe_allow_html=True)
    a,b=st.columns(2)
    with a:
        st.markdown('#### Project scope'); st.markdown('- Energy consumption and baseline analysis\n- Production and energy-intensity analysis\n- Machine-state and idle-energy investigation\n- Forecasting and what-if scenarios\n- CO₂-equivalent estimation and optimization opportunities\n- Decision-support recommendations and roadmap'); st.markdown('#### Technology stack'); st.markdown('**Python · Pandas · NumPy · Plotly · Scikit-learn · Streamlit**')
    with b:
        st.markdown('#### Dataset profile')
        if not src.empty:
            c=st.columns(3); c[0].metric('Records',f'{len(src):,}'); c[1].metric('Columns',f'{len(src.columns):,}'); c[2].metric('Machines',f"{src.Machine_ID.nunique():,}" if 'Machine_ID' in src else '—')
            if 'Timestamp' in src and src.Timestamp.notna().any(): st.write(f"**Time coverage:** {src.Timestamp.min():%d %b %Y} – {src.Timestamp.max():%d %b %Y}")
        else: st.info('Dataset profile unavailable.')
        st.markdown('#### Interpretation note'); st.warning('The dataset is synthetic and the emission factor is demonstration-only. Forecasts and savings are modelled estimates, not measured or guaranteed real-world results.')
    st.markdown('#### Output folders'); st.markdown('- **Energy & baseline:** `outputs/baseline/` and `outputs/cleaned_data/`\n- **Forecasting:** `outputs/forecasting/`\n- **What-if:** `outputs/what_if/`\n- **CO₂ & sustainability:** `outputs/co2_analysis/`\n- **Optimization:** `outputs/optimization/`\n- **Recommendations:** `outputs/recommendations/`')
    st.markdown('#### Refresh data')
    st.write('After rerunning the analytics pipeline, refresh the cached CSV files below.')
    if st.button('↻ Refresh dashboard data',type='primary'): st.cache_data.clear(); st.rerun()

st.markdown('<div class="footer">SUSTAINABLE MANUFACTURING ANALYTICS · Streamlit dashboard · Synthetic-data demonstration. Modelled opportunities are not achieved savings.</div>',unsafe_allow_html=True)
