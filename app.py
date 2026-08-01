
from __future__ import annotations

import io
import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from arch import arch_model
from scipy.optimize import brentq
from scipy.stats import norm


st.set_page_config(
    page_title="Mastering Volatility | Mountain Path Academy",
    page_icon="〽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAVY, BLUE, GOLD, DARK_GOLD = "#0B2545", "#0B5CAD", "#F3C84B", "#D4A017"
TEAL, GREEN, RED, PURPLE, ORANGE = "#13A89E", "#2E8B57", "#E45756", "#7C3AED", "#F28E2B"

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif}.stApp{background:linear-gradient(180deg,#F8FAFD,#EAF1F7)}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#081F3A,#124A78);color:#F7FAFC}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3,[data-testid="stSidebar"] p,[data-testid="stSidebar"] li{color:#F7FAFC}
.hero{background:linear-gradient(120deg,#071A2F 0%,#0B3B67 58%,#A97908 145%);padding:30px 34px;border-radius:22px;color:white;box-shadow:0 14px 34px rgba(7,26,47,.22);margin-bottom:16px;border:1px solid rgba(243,200,75,.35)}
.hero h1{font-size:2.25rem;margin:0 0 8px;color:white;font-weight:900}.hero p{margin:0;color:#DDEAF4;line-height:1.55}.eyebrow{color:#F3C84B;text-transform:uppercase;letter-spacing:.14em;font-weight:900;font-size:.76rem;margin-bottom:.55rem}
.section-title{font-size:1.42rem;font-weight:900;color:#0B2545;margin:18px 0 8px}.concept-card{background:white;border:1px solid #D9E5EF;border-top:5px solid #0B5CAD;padding:17px 18px;border-radius:15px;box-shadow:0 5px 16px rgba(18,54,84,.07);min-height:150px}.concept-card h3{color:#0B2545;font-size:1.05rem;margin:0 0 7px}.concept-card p{color:#3C5368;font-size:.91rem;line-height:1.5;margin:0}
.formula{background:linear-gradient(135deg,#FFF9E6,#FFF1B8);border:1px solid #E8C45B;border-left:6px solid #D4A017;padding:14px 18px;border-radius:12px;color:#3D3006;font-weight:800;margin:8px 0 14px}.teaching-note{background:#EAF7F5;border-left:5px solid #13A89E;padding:13px 16px;border-radius:10px;color:#153C3A;margin:10px 0}.warning-note{background:#FFF3E8;border-left:5px solid #F28E2B;padding:13px 16px;border-radius:10px;color:#57300A;margin:10px 0}
.result-box{background:linear-gradient(135deg,#0B2545,#0B5CAD);padding:16px 18px;border-radius:14px;color:white}.result-box .label{font-size:.78rem;text-transform:uppercase;letter-spacing:.08em;color:#CFE4F5}.result-box .value{font-size:1.55rem;font-weight:900;color:#F3C84B;margin-top:4px}.footer{background:linear-gradient(115deg,#081F3A,#124A78);color:#E6F1F8;padding:22px;border-radius:16px;margin-top:28px;text-align:center;border-top:4px solid #F3C84B}.footer a{color:#F3C84B!important;font-weight:800}
[data-testid="stMetric"]{background:#FFF;border:1px solid #DDE8F1;padding:13px;border-radius:14px}.stTabs [data-baseweb="tab-list"]{gap:9px!important;flex-wrap:wrap!important;background:#D8E3ED!important;padding:10px!important;border-radius:14px!important}.stTabs button[data-baseweb="tab"]{flex:1 1 145px!important;min-height:52px!important;background:#0B2545!important;border:2px solid #F3C84B!important;border-radius:10px!important;color:#F3C84B!important}.stTabs button[data-baseweb="tab"] p{color:#F3C84B!important;font-weight:850!important}.stTabs button[data-baseweb="tab"][aria-selected="true"]{background:linear-gradient(135deg,#F3C84B,#D4A017)!important}.stTabs button[data-baseweb="tab"][aria-selected="true"] p{color:#071A2F!important}.stButton button,.stDownloadButton button{background:#0B3B67!important;color:white!important;border-radius:10px!important;font-weight:800!important}
section[data-testid="stSidebar"] label p{color:#F3C84B!important;font-weight:850!important}section[data-testid="stSidebar"] div[data-testid="stSelectbox"] [data-baseweb="select"]>div{background:#FFF!important;border:2px solid #F3C84B!important}section[data-testid="stSidebar"] div[data-testid="stSelectbox"] *{color:#0B2545!important;-webkit-text-fill-color:#0B2545!important;font-weight:800!important}
@media(max-width:700px){.hero{padding:22px}.hero h1{font-size:1.7rem}}
</style>
""",
    unsafe_allow_html=True,
)

NIFTY_50 = {
    "NIFTY 50 Index": "^NSEI", "Adani Enterprises": "ADANIENT.NS", "Adani Ports": "ADANIPORTS.NS",
    "Apollo Hospitals": "APOLLOHOSP.NS", "Asian Paints": "ASIANPAINT.NS", "Axis Bank": "AXISBANK.NS",
    "Bajaj Auto": "BAJAJ-AUTO.NS", "Bajaj Finance": "BAJFINANCE.NS", "Bajaj Finserv": "BAJAJFINSV.NS",
    "Bharat Electronics": "BEL.NS", "Bharti Airtel": "BHARTIARTL.NS", "Cipla": "CIPLA.NS",
    "Coal India": "COALINDIA.NS", "Dr Reddy's Laboratories": "DRREDDY.NS", "Eicher Motors": "EICHERMOT.NS",
    "Eternal": "ETERNAL.NS", "Grasim Industries": "GRASIM.NS", "HCL Technologies": "HCLTECH.NS",
    "HDFC Bank": "HDFCBANK.NS", "HDFC Life": "HDFCLIFE.NS", "Hero MotoCorp": "HEROMOTOCO.NS",
    "Hindalco": "HINDALCO.NS", "Hindustan Unilever": "HINDUNILVR.NS", "ICICI Bank": "ICICIBANK.NS",
    "IndusInd Bank": "INDUSINDBK.NS", "Infosys": "INFY.NS", "ITC": "ITC.NS", "Jio Financial Services": "JIOFIN.NS",
    "JSW Steel": "JSWSTEEL.NS", "Kotak Mahindra Bank": "KOTAKBANK.NS", "Larsen & Toubro": "LT.NS",
    "Mahindra & Mahindra": "M&M.NS", "Maruti Suzuki": "MARUTI.NS", "Max Healthcare": "MAXHEALTH.NS",
    "Nestle India": "NESTLEIND.NS", "NTPC": "NTPC.NS", "Oil & Natural Gas Corp": "ONGC.NS",
    "Power Grid": "POWERGRID.NS", "Reliance Industries": "RELIANCE.NS", "SBI Life": "SBILIFE.NS",
    "Shriram Finance": "SHRIRAMFIN.NS", "State Bank of India": "SBIN.NS", "Sun Pharma": "SUNPHARMA.NS",
    "Tata Consumer": "TATACONSUM.NS", "Tata Motors": "TATAMOTORS.NS", "Tata Steel": "TATASTEEL.NS",
    "TCS": "TCS.NS", "Tech Mahindra": "TECHM.NS", "Titan": "TITAN.NS", "Trent": "TRENT.NS",
    "UltraTech Cement": "ULTRACEMCO.NS", "Wipro": "WIPRO.NS",
}


def section(title: str) -> None:
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)


def note(text: str, warning: bool = False) -> None:
    st.markdown(f"<div class='{'warning-note' if warning else 'teaching-note'}'>{text}</div>", unsafe_allow_html=True)


def card(title: str, body: str, color: str = BLUE) -> str:
    return f"<div class='concept-card' style='border-top-color:{color}'><h3>{title}</h3><p>{body}</p></div>"


def result_box(label: str, value: str) -> None:
    st.markdown(f"<div class='result-box'><div class='label'>{label}</div><div class='value'>{value}</div></div>", unsafe_allow_html=True)


def style_fig(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="white", font=dict(family="Inter", color=NAVY), margin=dict(l=35, r=25, t=60, b=35), hoverlabel=dict(bgcolor="white", font_color=NAVY))
    fig.update_xaxes(gridcolor="#E7EEF4"); fig.update_yaxes(gridcolor="#E7EEF4")
    return fig


@st.cache_data(ttl=900, show_spinner=False)
def load_prices(ticker: str, years: int) -> pd.DataFrame:
    end = date.today() + timedelta(days=1)
    start = end - timedelta(days=365 * years + 20)
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False, threads=False)
    if raw.empty:
        raise ValueError("No observations were returned by the market-data provider.")
    close = raw["Close"]
    if isinstance(close, pd.DataFrame): close = close.iloc[:, 0]
    out = pd.DataFrame({"Close": pd.to_numeric(close, errors="coerce")}).dropna()
    out["Log return"] = np.log(out["Close"] / out["Close"].shift(1))
    return out.dropna()


def demo_prices(years: int) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=252 * years)
    eps = rng.normal(size=len(idx)); variance = np.empty(len(idx)); returns = np.empty(len(idx)); variance[0] = 0.00016
    for i in range(1, len(idx)):
        variance[i] = 0.000002 + 0.08 * returns[i-1] ** 2 + 0.90 * variance[i-1]
        returns[i] = 0.00035 + math.sqrt(max(variance[i], 1e-10)) * eps[i]
    close = 1000 * np.exp(np.cumsum(returns))
    return pd.DataFrame({"Close": close, "Log return": returns}, index=idx)


def bs_price(spot: float, strike: float, t: float, rate: float, sigma: float, option: str, dividend: float = 0.0) -> float:
    if t <= 0 or sigma <= 0: return max(spot-strike, 0) if option == "Call" else max(strike-spot, 0)
    d1 = (math.log(spot/strike) + (rate-dividend + 0.5*sigma*sigma)*t) / (sigma*math.sqrt(t)); d2 = d1 - sigma*math.sqrt(t)
    if option == "Call": return spot*math.exp(-dividend*t)*norm.cdf(d1) - strike*math.exp(-rate*t)*norm.cdf(d2)
    return strike*math.exp(-rate*t)*norm.cdf(-d2) - spot*math.exp(-dividend*t)*norm.cdf(-d1)


def implied_vol(market: float, spot: float, strike: float, t: float, rate: float, option: str, dividend: float) -> float:
    intrinsic = max(spot*math.exp(-dividend*t)-strike*math.exp(-rate*t), 0) if option == "Call" else max(strike*math.exp(-rate*t)-spot*math.exp(-dividend*t), 0)
    upper = spot*math.exp(-dividend*t) if option == "Call" else strike*math.exp(-rate*t)
    if not intrinsic <= market <= upper: raise ValueError(f"Option price must lie between ₹{intrinsic:.2f} and ₹{upper:.2f} under these inputs.")
    return float(brentq(lambda vol: bs_price(spot, strike, t, rate, vol, option, dividend)-market, 1e-5, 5.0))


def fit_models(returns: pd.Series, horizon: int) -> tuple[pd.DataFrame, object, object]:
    r = (returns.dropna() * 100).clip(-25, 25)
    arch_fit = arch_model(r, mean="Constant", vol="ARCH", p=5, dist="t", rescale=False).fit(disp="off")
    garch_fit = arch_model(r, mean="Constant", vol="GARCH", p=1, q=1, dist="t", rescale=False).fit(disp="off")
    af = arch_fit.forecast(horizon=horizon, reindex=False).variance.iloc[-1].to_numpy()
    gf = garch_fit.forecast(horizon=horizon, reindex=False).variance.iloc[-1].to_numpy()
    dates = pd.bdate_range(returns.index[-1] + pd.Timedelta(days=1), periods=horizon)
    forecasts = pd.DataFrame({"ARCH(5) annualised %": np.sqrt(af*252), "GARCH(1,1) annualised %": np.sqrt(gf*252)}, index=dates)
    return forecasts, arch_fit, garch_fit


def build_excel_download(
    price_data: pd.DataFrame,
    comparison_data: pd.DataFrame,
    asset: str,
    ticker: str,
    period: str,
    data_source: str,
    as_of: date,
    latest_close: float,
    full_hist_vol: float,
    latest_rolling_vol: float,
    rolling_days: int,
) -> bytes:
    """Create an analysis-ready, formatted Excel workbook in memory."""
    output = io.BytesIO()
    export_data = price_data.reset_index().copy()
    export_data.columns = ["Date", "Adjusted Close (₹)", "Daily Log Return", "Rolling Volatility", "EWMA Volatility"]
    export_data["Date"] = pd.to_datetime(export_data["Date"]).dt.tz_localize(None)

    with pd.ExcelWriter(output, engine="xlsxwriter", datetime_format="dd-mmm-yyyy") as writer:
        workbook = writer.book
        navy, blue, gold, pale_blue, pale_gold = "#0B2545", "#0B5CAD", "#F3C84B", "#EAF1F7", "#FFF1B8"
        title_fmt = workbook.add_format({"bold": True, "font_color": "#FFFFFF", "bg_color": navy, "font_size": 18, "align": "left", "valign": "vcenter"})
        subtitle_fmt = workbook.add_format({"font_color": "#DDEAF4", "bg_color": navy, "font_size": 10, "align": "left", "valign": "vcenter"})
        section_fmt = workbook.add_format({"bold": True, "font_color": "#FFFFFF", "bg_color": blue, "font_size": 11, "align": "left", "valign": "vcenter"})
        label_fmt = workbook.add_format({"bold": True, "font_color": navy, "bg_color": pale_blue, "border": 0, "align": "left"})
        value_fmt = workbook.add_format({"font_color": navy, "bg_color": "#FFFFFF", "align": "right"})
        currency_fmt = workbook.add_format({"font_color": navy, "bg_color": "#FFFFFF", "num_format": '#,##0.00;[Red](#,##0.00);-'})
        percent_fmt = workbook.add_format({"font_color": navy, "bg_color": "#FFFFFF", "num_format": '0.00%;[Red](0.00%);-'})
        note_fmt = workbook.add_format({"font_color": "#3C5368", "bg_color": pale_gold, "text_wrap": True, "valign": "top"})
        header_fmt = workbook.add_format({"bold": True, "font_color": "#FFFFFF", "bg_color": navy, "border": 0, "align": "center", "valign": "vcenter", "text_wrap": True})

        summary = workbook.add_worksheet("Summary")
        writer.sheets["Summary"] = summary
        summary.hide_gridlines(2)
        summary.set_column("A:A", 29); summary.set_column("B:B", 24); summary.set_column("C:F", 15)
        summary.set_row(0, 30); summary.merge_range("A1:F1", "Mastering Basics of Volatility", title_fmt)
        summary.merge_range("A2:F2", "Mountain Path Academy · Formatted analysis download", subtitle_fmt)
        summary.merge_range("A4:F4", "Analysis profile", section_fmt)
        profile = [
            ("Instrument", asset), ("Ticker", ticker), ("Selected period", period),
            ("As-of date", as_of), ("Data source", data_source), ("Observations", len(price_data)),
        ]
        for row, (label, value) in enumerate(profile, start=4):
            summary.write(row, 0, label, label_fmt)
            if isinstance(value, date): summary.write_datetime(row, 1, pd.Timestamp(value).to_pydatetime(), workbook.add_format({"num_format": "dd-mmm-yyyy", "font_color": navy}))
            else: summary.write(row, 1, value, value_fmt)
        summary.merge_range("A12:F12", "Key volatility outputs", section_fmt)
        metrics = [
            ("Latest adjusted close (₹)", latest_close, currency_fmt),
            ("Full-period historic volatility", full_hist_vol / 100, percent_fmt),
            (f"Latest {rolling_days}-day rolling volatility", latest_rolling_vol / 100, percent_fmt),
        ]
        for row, (label, value, fmt) in enumerate(metrics, start=12):
            summary.write(row, 0, label, label_fmt); summary.write(row, 1, value, fmt)
        summary.merge_range("A17:F17", "Interpretation", section_fmt)
        summary.merge_range("A18:F20", "Volatility measures dispersion, not return direction or maximum loss. Historic volatility is backward-looking; implied volatility is model- and market-dependent; ARCH/GARCH forecasts remain conditional on the selected sample and specification.", note_fmt)
        summary.merge_range("A22:F22", "Source and usage note", section_fmt)
        summary.merge_range("A23:F25", f"Source: {data_source}. Market data may be delayed. Nifty 50 constituents may change. This workbook is educational material only—not investment, trading, or option-pricing advice.", note_fmt)
        summary.freeze_panes(3, 0)

        export_data.to_excel(writer, sheet_name="Daily Data", index=False, startrow=1)
        daily = writer.sheets["Daily Data"]
        daily.hide_gridlines(2); daily.freeze_panes(2, 1); daily.autofilter(1, 0, len(export_data) + 1, len(export_data.columns) - 1); daily.set_row(0, 30)
        daily.merge_range("A1:E1", f"{asset} ({ticker}) · Adjusted daily price and volatility series", title_fmt)
        for col, name in enumerate(export_data.columns): daily.write(1, col, name, header_fmt)
        daily.set_column("A:A", 14, workbook.add_format({"num_format": "dd-mmm-yyyy"}))
        price_fmt = workbook.add_format({"num_format": '#,##0.00;[Red](#,##0.00);-'})
        daily.set_column("B:B", 18, price_fmt)
        daily.set_column("C:C", 18, workbook.add_format({"num_format": '0.0000%;[Red](0.0000%);-'}))
        daily.set_column("D:E", 20, workbook.add_format({"num_format": '0.00%;[Red](0.00%);-'}))
        # App values are stored as percentage points; scale only the workbook display columns to true percentage values.
        for excel_row, (_, record) in enumerate(export_data.iterrows(), start=2):
            daily.write_number(excel_row, 1, float(record["Adjusted Close (₹)"]), price_fmt)
            if pd.notna(record["Rolling Volatility"]): daily.write_number(excel_row, 3, float(record["Rolling Volatility"]) / 100)
            if pd.notna(record["EWMA Volatility"]): daily.write_number(excel_row, 4, float(record["EWMA Volatility"]) / 100)
        chart = workbook.add_chart({"type": "line"})
        chart.add_series({"name": f"Rolling {rolling_days}D", "categories": ["Daily Data", 2, 0, len(export_data) + 1, 0], "values": ["Daily Data", 2, 3, len(export_data) + 1, 3], "line": {"color": "#7C3AED", "width": 2}})
        chart.add_series({"name": "EWMA", "categories": ["Daily Data", 2, 0, len(export_data) + 1, 0], "values": ["Daily Data", 2, 4, len(export_data) + 1, 4], "line": {"color": "#13A89E", "width": 2}})
        chart.set_title({"name": "Annualised volatility through time"}); chart.set_x_axis({"date_axis": True, "num_format": "dd-mmm"}); chart.set_y_axis({"name": "Volatility", "num_format": "0%"}); chart.set_legend({"position": "bottom"}); chart.set_style(10)
        daily.insert_chart("G3", chart, {"x_scale": 1.35, "y_scale": 1.2})

        comparison_data.to_excel(writer, sheet_name="Model Comparison", index=False, startrow=1)
        compare_sheet = writer.sheets["Model Comparison"]
        compare_sheet.hide_gridlines(2); compare_sheet.freeze_panes(2, 0); compare_sheet.set_row(0, 30)
        compare_sheet.merge_range("A1:D1", "Volatility model comparison", title_fmt)
        for col, name in enumerate(comparison_data.columns): compare_sheet.write(1, col, name, header_fmt)
        compare_sheet.set_column("A:A", 28); compare_sheet.set_column("B:B", 25, workbook.add_format({"num_format": '0.00%;[Red](0.00%);-'})); compare_sheet.set_column("C:C", 30); compare_sheet.set_column("D:D", 35)
        for excel_row, value in enumerate(comparison_data["Annualised volatility (%)"], start=2): compare_sheet.write_number(excel_row, 1, float(value) / 100)
        compare_sheet.autofilter(1, 0, len(comparison_data) + 1, 3)

        guide = workbook.add_worksheet("Methodology & Limits")
        writer.sheets["Methodology & Limits"] = guide
        guide.hide_gridlines(2); guide.set_column("A:A", 24); guide.set_column("B:B", 95); guide.set_row(0, 30)
        guide.merge_range("A1:B1", "Methodology, assumptions and limitations", title_fmt)
        guide.write_row("A3", ["Measure", "What it means and what to watch"], header_fmt)
        guidance = [
            ("Historic volatility", "Annualised standard deviation of daily log returns. Sensitive to sample period, frequency, corporate-action adjustments and the √252 convention."),
            ("Rolling volatility", "Highlights changing regimes. Short windows react quickly but are noisy; long windows are smoother but slower."),
            ("EWMA", "Weights recent squared returns more heavily using λ=0.94. The decay choice is an assumption."),
            ("Black–Scholes IV", "The volatility that makes a Black–Scholes price match the observed premium. Depends on model assumptions, market liquidity, rates, dividends, strike and expiry."),
            ("ARCH / GARCH", "Conditional variance forecasts based on past shocks and variance. Can miss jumps, leverage effects, structural breaks and exogenous events."),
            ("Important", "Volatility is not direction, expected return, Value at Risk, or maximum possible loss. Use these outputs as educational estimates, not trading recommendations."),
        ]
        body_wrap = workbook.add_format({"text_wrap": True, "valign": "top", "font_color": navy})
        for row, (measure, explanation) in enumerate(guidance, start=3):
            guide.write(row, 0, measure, label_fmt); guide.write(row, 1, explanation, body_wrap); guide.set_row(row, 42)
        guide.write(11, 0, "Data source", label_fmt); guide.write(11, 1, data_source, body_wrap)
        guide.write(12, 0, "Website", label_fmt); guide.write_url(12, 1, "https://themountainpathacademy.com/courses", string="https://themountainpathacademy.com/courses")
        guide.freeze_panes(2, 0)

        workbook.set_properties({"title": "Mastering Basics of Volatility", "subject": "Educational volatility analysis", "author": "Mountain Path Academy", "comments": "Generated by the Streamlit learning studio"})
    return output.getvalue()


with st.sidebar:
    st.markdown("## 〽️ Mastering Volatility")
    st.caption("MP2 · Interactive learning studio")
    asset_name = st.selectbox("Choose a Nifty 50 stock or index", list(NIFTY_50))
    years_label = st.selectbox("Daily-price period", ["1 year", "3 years", "5 years"], index=1)
    years = int(years_label[0]); rolling_window = st.slider("Rolling-volatility window", 10, 90, 21)
    forecast_horizon = st.select_slider("Forecast horizon (trading days)", options=[1, 5, 10, 21, 63], value=21)
    if st.button("↻ Refresh market data", use_container_width=True, help="Clears cached prices and requests the latest available daily observation."):
        load_prices.clear()
        st.rerun()
    st.markdown("---")
    st.markdown("### Guided journey\n1. Learn the volatility map\n2. Explore history\n3. Reverse-engineer IV\n4. Forecast changing risk\n5. Compare and interpret\n6. Test your learning")
    use_demo = st.toggle("Use classroom demo data", value=False, help="Useful if Yahoo Finance is temporarily unavailable.")

st.markdown("<div class='hero'><div class='eyebrow'>Mountain Path Academy · Applied Finance</div><h1>Mastering Basics of Volatility</h1><p>From realised price variation to option-implied expectations and ARCH/GARCH forecasts—learn what each measure says, what it misses, and how to use it responsibly.</p></div>", unsafe_allow_html=True)

try:
    data = demo_prices(years) if use_demo else load_prices(NIFTY_50[asset_name], years)
    source = "Reproducible classroom simulation" if use_demo else "Yahoo Finance · adjusted daily close"
except Exception as exc:
    st.warning(f"Live prices are unavailable ({exc}). A reproducible classroom series is shown so learning can continue.")
    data, source = demo_prices(years), "Reproducible classroom simulation (automatic fallback)"

data["Rolling volatility"] = data["Log return"].rolling(rolling_window).std() * np.sqrt(252) * 100
data["EWMA volatility"] = np.sqrt(data["Log return"].pow(2).ewm(alpha=1-0.94, adjust=False).mean() * 252) * 100
hist_vol = data["Log return"].std() * np.sqrt(252) * 100
down_vol = data.loc[data["Log return"] < 0, "Log return"].std() * np.sqrt(252) * 100
spot = float(data["Close"].iloc[-1])

m1,m2,m3,m4 = st.columns(4)
with m1: st.metric("Latest close", f"₹{spot:,.2f}")
with m2: st.metric("Annualised historic vol", f"{hist_vol:.2f}%")
with m3: st.metric(f"{rolling_window}-day rolling vol", f"{data['Rolling volatility'].iloc[-1]:.2f}%")
with m4: st.metric("Observations", f"{len(data):,}")
st.caption(f"{asset_name} ({NIFTY_50[asset_name]}) · {source} · Last observation: {data.index[-1].date():%d %b %Y}. Prices can be delayed and constituents can change.")

tabs = st.tabs(["🧭 Volatility map", "📉 Historic", "🎯 Implied (Black–Scholes)", "🔮 ARCH & GARCH", "⚖️ Compare", "🧪 Practice & quiz"])

with tabs[0]:
    section("One word, three different questions")
    c1,c2,c3 = st.columns(3)
    with c1: st.markdown(card("Historic volatility", "How widely did realised returns move? It is backward-looking and computed from observed returns.", BLUE), unsafe_allow_html=True)
    with c2: st.markdown(card("Implied volatility", "What volatility makes a pricing model match an observed option premium? It is a model-implied market quote—not a pure forecast.", GOLD), unsafe_allow_html=True)
    with c3: st.markdown(card("Forecast volatility", "Given volatility clustering, what conditional variance does a time-series model expect next?", TEAL), unsafe_allow_html=True)
    st.markdown("<div class='formula'>Daily log return: rₜ = ln(Pₜ/Pₜ₋₁) &nbsp; | &nbsp; Annualised historic volatility: σ = SD(rₜ) × √252</div>", unsafe_allow_html=True)
    section("A disciplined interpretation")
    st.markdown("Volatility measures dispersion, not direction. A 25% volatility estimate does **not** predict a 25% loss, and low volatility does not mean low fundamental or liquidity risk.")
    note("Annualisation by √252 assumes daily returns are sufficiently comparable over time. Autocorrelation, jumps, regime changes and non-trading periods weaken that simplification.", True)
    with st.expander("Core assumptions and common traps", expanded=True):
        st.markdown("""
- **Sampling matters:** close-to-close volatility ignores intraday paths; adjusted prices reduce corporate-action distortions.
- **Window choice matters:** short windows react quickly but are noisy; long windows are stable but slow.
- **Returns are not perfectly normal:** skewness and fat tails make extreme moves more frequent than a bell curve implies.
- **Volatility is latent:** every method estimates an unobservable, changing quantity.
- **Comparability needs consistency:** use the same return definition, frequency, annualisation and window.
""")

with tabs[1]:
    section("Price path and realised risk")
    fig = go.Figure(go.Scatter(x=data.index,y=data["Close"],line=dict(color=BLUE,width=2),name="Adjusted close")); fig.update_layout(title=f"{asset_name}: adjusted daily close",yaxis_title="Price (₹)"); st.plotly_chart(style_fig(fig),use_container_width=True)
    fig = go.Figure(); fig.add_trace(go.Scatter(x=data.index,y=data["Rolling volatility"],name=f"Rolling {rolling_window}D",line=dict(color=PURPLE))); fig.add_trace(go.Scatter(x=data.index,y=data["EWMA volatility"],name="EWMA λ=0.94",line=dict(color=TEAL))); fig.add_hline(y=hist_vol,line_dash="dash",line_color=GOLD,annotation_text="Full-period HV"); fig.update_layout(title="Annualised volatility through time",yaxis_title="Volatility (%)"); st.plotly_chart(style_fig(fig),use_container_width=True)
    c1,c2 = st.columns(2)
    with c1:
        hist = go.Figure(go.Histogram(x=data["Log return"]*100,nbinsx=55,marker_color=BLUE)); hist.update_layout(title="Distribution of daily log returns",xaxis_title="Daily return (%)",yaxis_title="Days"); st.plotly_chart(style_fig(hist,360),use_container_width=True)
    with c2:
        ranked = data["Log return"].abs().nlargest(10).sort_values()
        shock = go.Figure(go.Bar(x=ranked*100,y=ranked.index.strftime("%d %b %Y"),orientation="h",marker_color=ORANGE)); shock.update_layout(title="Ten largest absolute daily moves",xaxis_title="Absolute log return (%)"); st.plotly_chart(style_fig(shock,360),use_container_width=True)
    note(f"Full-period historic volatility is {hist_vol:.2f}%; downside-only volatility is {down_vol:.2f}%. This describes the selected sample—it does not guarantee the next period.")

with tabs[2]:
    section("Black–Scholes implied-volatility laboratory")
    st.markdown("Implied volatility is the volatility input that forces a model price to equal the observed option premium. The solver reverses Black–Scholes numerically.")
    st.markdown("<div class='formula'>Call = S₀e⁻ᑫᵀN(d₁) − Ke⁻ʳᵀN(d₂) &nbsp; | &nbsp; Put = Ke⁻ʳᵀN(−d₂) − S₀e⁻ᑫᵀN(−d₁)</div>", unsafe_allow_html=True)
    a,b,c = st.columns(3)
    with a:
        option_type = st.selectbox("Option type",["Call","Put"]); option_spot = st.number_input("Spot price (₹)",min_value=0.01,value=float(round(spot,2))); strike = st.number_input("Strike price (₹)",min_value=0.01,value=float(round(spot/50)*50))
    with b:
        days = st.number_input("Calendar days to expiry",min_value=1,max_value=1825,value=30); rate_pct = st.number_input("Risk-free rate (% p.a.)",value=6.50,step=.10); dividend_pct = st.number_input("Dividend yield (% p.a.)",value=1.00,step=.10)
    with c:
        reference_price = bs_price(option_spot,strike,days/365,rate_pct/100,max(hist_vol/100,.01),option_type,dividend_pct/100)
        market_price = st.number_input("Observed option premium (₹)",min_value=0.01,value=float(max(round(reference_price,2),.01)),help="Enter a market option premium; the app does not download an option chain.")
        try:
            iv = implied_vol(market_price,option_spot,strike,days/365,rate_pct/100,option_type,dividend_pct/100)
            result_box("Black–Scholes implied volatility",f"{iv:.2%}")
        except ValueError as exc: st.error(str(exc)); iv = np.nan
    if np.isfinite(iv):
        vols=np.linspace(max(.01,iv*.3),min(3,iv*1.8),100); prices=[bs_price(option_spot,strike,days/365,rate_pct/100,v,option_type,dividend_pct/100) for v in vols]
        fig=go.Figure(go.Scatter(x=vols*100,y=prices,line=dict(color=PURPLE,width=3))); fig.add_hline(y=market_price,line_dash="dash",line_color=GOLD,annotation_text="Observed premium"); fig.add_vline(x=iv*100,line_dash="dot",line_color=TEAL,annotation_text="Solved IV"); fig.update_layout(title="How volatility changes the model option price",xaxis_title="Volatility (%)",yaxis_title="Option value (₹)"); st.plotly_chart(style_fig(fig),use_container_width=True)
    note("Black–Scholes assumes frictionless trading, lognormal prices, continuous hedging, and constant volatility/rates. Real markets show volatility smiles, jumps, discrete trading, transaction costs and liquidity effects. IV also embeds supply, demand and model error.", True)

with tabs[3]:
    section("Forecasting conditional volatility")
    st.markdown("ARCH lets recent squared shocks drive variance. GARCH(1,1) adds yesterday's variance, usually producing the persistent clusters seen in financial returns.")
    st.markdown("<div class='formula'>ARCH(5): σ²ₜ = ω + Σαᵢε²ₜ₋ᵢ &nbsp; | &nbsp; GARCH(1,1): σ²ₜ = ω + αε²ₜ₋₁ + βσ²ₜ₋₁</div>", unsafe_allow_html=True)
    try:
        forecasts, arch_fit, garch_fit = fit_models(data["Log return"], forecast_horizon)
        c1,c2,c3=st.columns(3)
        with c1: st.metric("ARCH forecast · day 1",f"{forecasts.iloc[0,0]:.2f}%")
        with c2: st.metric("GARCH forecast · day 1",f"{forecasts.iloc[0,1]:.2f}%")
        persistence=float(garch_fit.params.get("alpha[1]",np.nan)+garch_fit.params.get("beta[1]",np.nan))
        with c3: st.metric("GARCH persistence α+β",f"{persistence:.3f}")
        fig=go.Figure(); fig.add_trace(go.Scatter(x=forecasts.index,y=forecasts.iloc[:,0],name="ARCH(5)",line=dict(color=ORANGE,width=3))); fig.add_trace(go.Scatter(x=forecasts.index,y=forecasts.iloc[:,1],name="GARCH(1,1)",line=dict(color=TEAL,width=3))); fig.add_hline(y=hist_vol,line_dash="dash",line_color=GOLD,annotation_text="Historical sample vol"); fig.update_layout(title=f"Variance-model forecasts: next {forecast_horizon} trading day(s)",yaxis_title="Annualised volatility (%)"); st.plotly_chart(style_fig(fig),use_container_width=True)
        params=pd.concat([arch_fit.params.rename("ARCH(5)"),garch_fit.params.rename("GARCH(1,1)")],axis=1); st.dataframe(params.style.format("{:.5f}"),use_container_width=True)
        if persistence >= .98: note("Estimated persistence is very high. Volatility shocks decay slowly; long-horizon estimates can be unstable and sensitive to the sample.",True)
    except Exception as exc: st.error(f"The variance models could not be fitted to this sample: {exc}")
    note("ARCH/GARCH forecasts are conditional on past returns and a chosen specification. They can miss structural breaks, leverage effects, jumps, intraday information and exogenous events. Statistical convergence is not proof of economic accuracy.",True)

with tabs[4]:
    section("Do the measures agree?")
    rows=[("Historic volatility",hist_vol,"Realised daily returns","Backward-looking benchmark"),(f"Rolling {rolling_window}-day",data["Rolling volatility"].iloc[-1],"Recent realised returns","Current regime monitor"),("EWMA (λ=0.94)",data["EWMA volatility"].iloc[-1],"Exponentially weighted returns","Fast risk update")]
    if 'forecasts' in locals(): rows += [("ARCH(5), day 1",forecasts.iloc[0,0],"Recent shocks","Short-run conditional forecast"),("GARCH(1,1), day 1",forecasts.iloc[0,1],"Shocks + prior variance","Persistent conditional forecast")]
    if 'iv' in locals() and np.isfinite(iv): rows += [("Black–Scholes IV",iv*100,"Observed option premium","Market/model-implied quote")]
    comparison=pd.DataFrame(rows,columns=["Measure","Annualised volatility (%)","Information source","Best interpreted as"])
    fig=go.Figure(go.Bar(x=comparison["Measure"],y=comparison["Annualised volatility (%)"],marker_color=[BLUE,PURPLE,TEAL,ORANGE,GREEN,GOLD][:len(comparison)],text=comparison["Annualised volatility (%)"].map(lambda x:f"{x:.1f}%"),textposition="outside")); fig.update_layout(title="Volatility is a family of estimates—not one truth",yaxis_title="Annualised volatility (%)"); st.plotly_chart(style_fig(fig),use_container_width=True)
    st.dataframe(comparison.style.format({"Annualised volatility (%)":"{:.2f}"}),use_container_width=True,hide_index=True)
    note("Differences are informative: IV may exceed realised estimates because options reflect forward uncertainty and risk premia; GARCH may jump after a shock; long-window HV may react slowly.")
    excel_bytes = build_excel_download(
        data, comparison, asset_name, NIFTY_50[asset_name], years_label, source,
        data.index[-1].date(), spot, hist_vol, data["Rolling volatility"].iloc[-1], rolling_window,
    )
    st.download_button(
        "⬇ Download formatted Excel analysis",
        excel_bytes,
        file_name=f"{NIFTY_50[asset_name].replace('^','')}_volatility_analysis.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

with tabs[5]:
    section("Worked practice")
    practice_returns=st.text_input("Five daily returns in %, separated by commas","1.2, -0.8, 0.5, -1.5, 0.9")
    try:
        pr=np.array([float(x.strip())/100 for x in practice_returns.split(",")]); practice_vol=pr.std(ddof=1)*math.sqrt(252)
        st.success(f"Sample daily standard deviation = {pr.std(ddof=1):.4%}; annualised volatility = {practice_vol:.2%}.")
        with st.expander("Show calculation logic"):
            st.markdown(
                "1. Convert percentages to decimals.  \n"
                "2. Compute sample standard deviation (`STDEV.S` in Excel).  \n"
                "3. Multiply by √252.  \n"
                "4. Interpret as dispersion—not expected return or maximum loss."
            )
    except Exception: st.info("Enter valid comma-separated numbers.")
    section("Knowledge check")
    questions=[
        ("What does annualised historic volatility primarily measure?",["Direction of return","Dispersion of realised returns","Maximum possible loss"],1),
        ("Implied volatility is obtained by…",["Averaging past returns","Inverting an option-pricing model","Reading a company's balance sheet"],1),
        ("Why can GARCH react to a market shock?",["Squared shocks enter conditional variance","It knows future news","It assumes constant variance"],0),
        ("Which statement is correct?",["High volatility always means negative return","Low volatility removes tail risk","Volatility does not predict direction"],2),
        ("A major Black–Scholes limitation is…",["It assumes constant volatility","It cannot price calls","It never uses time to expiry"],0),
    ]
    answers=[]
    for i,(q,opts,correct) in enumerate(questions): answers.append(st.radio(f"{i+1}. {q}",opts,index=None,key=f"quiz_{i}"))
    if st.button("Score my quiz"):
        score=sum(a==opts[correct] for a,(_,opts,correct) in zip(answers,questions)); st.success(f"Score: {score}/{len(questions)}")
        if score<len(questions): st.info("Review the map and model-limitations notes, then try again.")

st.markdown("<div class='footer'><strong>Mountain Path Academy</strong><br>Mastering Basics of Volatility · Educational material only—not investment, trading or option-pricing advice.<br><a href='https://themountainpathacademy.com/courses' target='_blank'>Explore courses</a> &nbsp;·&nbsp; <a href='https://themountainpathacademy.com/contact' target='_blank'>Contact & enrol</a></div>",unsafe_allow_html=True)
