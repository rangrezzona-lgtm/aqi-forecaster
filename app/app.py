"""Streamlit dashboard: Mumbai AQI forecasts (actual vs predicted).

Run from the project root:  streamlit run app/app.py
Needs data/processed/lstm_predictions.csv and all_results.csv,
which are created by src/lstm_model.py.
"""
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
PRED_PATH = ROOT / "data" / "processed" / "lstm_predictions.csv"
RESULTS_PATH = ROOT / "data" / "processed" / "all_results.csv"

st.set_page_config(page_title="Mumbai AQI Forecaster", page_icon="🌫", layout="wide")


def aqi_category(value):
    """Indian CPCB AQI categories."""
    if value <= 50:
        return "Good"
    if value <= 100:
        return "Satisfactory"
    if value <= 200:
        return "Moderate"
    if value <= 300:
        return "Poor"
    if value <= 400:
        return "Very Poor"
    return "Severe"


@st.cache_data
def load_data():
    preds = pd.read_csv(PRED_PATH, index_col="Date", parse_dates=True)
    results = pd.read_csv(RESULTS_PATH)
    return preds, results


st.title("🌫 Mumbai AQI Forecaster")
st.write(
    "Forecasts Mumbai's Air Quality Index 24 and 48 hours ahead. "
    "The chart shows the LSTM model's predictions on a held-out test period "
    "(about 115 days in 2020) against the real readings."
)

if not PRED_PATH.exists() or not RESULTS_PATH.exists():
    st.error(
        "Prediction files not found. Run `python src/lstm_model.py` first, "
        "so that data/processed/lstm_predictions.csv is created."
    )
    st.stop()

preds, results = load_data()

horizon = st.radio("Forecast horizon", ["24h", "48h"], horizontal=True)
actual = preds[f"actual_{horizon}"]
predicted = preds[f"pred_{horizon}"]

col1, col2, col3 = st.columns(3)
mae = (actual - predicted).abs().mean()
col1.metric("LSTM MAE (test)", f"{mae:.1f} AQI points")
col2.metric("Latest actual AQI", f"{actual.iloc[-1]:.0f}", aqi_category(actual.iloc[-1]))
col3.metric("Latest predicted AQI", f"{predicted.iloc[-1]:.0f}", aqi_category(predicted.iloc[-1]))

fig = go.Figure()
fig.add_trace(go.Scatter(x=preds.index, y=actual, name="Actual AQI", mode="lines"))
fig.add_trace(go.Scatter(x=preds.index, y=predicted, name="LSTM predicted",
                         mode="lines", line=dict(dash="dash")))
fig.update_layout(
    title=f"Actual vs predicted AQI ({horizon} ahead)",
    xaxis_title="Date the forecast was made", yaxis_title="AQI", hovermode="x unified",
    legend=dict(orientation="h", y=1.1),
)
st.plotly_chart(fig)

st.subheader("Model comparison (test set, lower is better)")
st.dataframe(results[results["Horizon"] == horizon].sort_values("MAE"),
             hide_index=True)

st.caption(
    "Data: Kaggle 'Air Quality Data in India (2015-2020)', Mumbai, May 2018 to Jul 2020. "
    "This is a historical test, not a live forecast. The test period overlaps the "
    "2020 COVID-19 lockdown. On this small dataset a Random Forest is as good as or "
    "better than the LSTM."
)
