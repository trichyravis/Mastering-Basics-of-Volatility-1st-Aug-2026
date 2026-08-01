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


with st.sidebar:
    st.markdown("## 〽️ Mastering Volatility")
    st.caption("MP2 · Interactive learning studio")
    asset_name = st.selectbox("Choose a Nifty 50 stock or index", list(NIFTY_50))
    years_label = st.selectbox("Daily-price period", ["1 year", "3 years", "5 years"], index=1)
    years = int(years_label[0]); rolling_window = st.slider("Rolling-volatility window", 10, 90, 21)
    forecast_horizon = st.select_slider("Forecast horizon (trading days)", options=[1, 5, 10, 21, 63], value=21)
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
    csv=data.reset_index().to_csv(index=False).encode(); st.download_button("Download analysed daily data (CSV)",csv,file_name=f"{NIFTY_50[asset_name].replace('^','')}_volatility.csv",mime="text/csv")

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
