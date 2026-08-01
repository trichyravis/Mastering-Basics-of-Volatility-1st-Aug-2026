# Mastering Basics of Volatility — Interactive Streamlit Learning Studio

A Mountain Path Academy educational app using the same navy-and-gold MP1 design system as **Mastering Returns**.

## What learners can do

- Select a Nifty 50 constituent or the Nifty 50 index.
- Load adjusted daily prices for 1, 3, or 5 years.
- Explore full-period, rolling, downside, and EWMA historic volatility.
- Reverse-engineer Black–Scholes implied volatility from an observed option premium.
- Fit ARCH(5) and GARCH(1,1) models and compare forward variance forecasts.
- Study assumptions, limitations, worked practice, and a scored knowledge check.
- Download the analysed daily dataset.

The app uses Yahoo Finance through `yfinance`. A reproducible classroom simulation is available in the sidebar and is also used automatically if live data is temporarily unavailable. Market prices may be delayed. The Nifty 50 list is an educational snapshot and should be reviewed when index constituents change.

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Upload this folder to a GitHub repository.
2. Select `app.py` as the main file.
3. Deploy.

Educational material only — not investment, trading, or option-pricing advice.
