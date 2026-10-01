"""Load, clean and feature-engineer Mumbai AQI data.

Expected input: city_day.csv from the Kaggle "Air Quality Data in India
(2015-2020)" dataset, saved as data/raw/city_day.csv.
Columns used: City, Date, PM2.5, PM10, NO2, SO2, CO, O3, AQI
"""
from pathlib import Path

import numpy as np
import pandas as pd

POLLUTANTS = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]


def load_city(path="data/raw/city_day.csv", city="Mumbai"):
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df[df["City"] == city].sort_values("Date").set_index("Date")
    cols = [c for c in POLLUTANTS + ["AQI"] if c in df.columns]
    return df[cols]


def clean(df):
    # Put every calendar day on the index so lags mean "N days ago"
    df = df.asfreq("D")
    # Fill short gaps only (max 3 days); long gaps stay NaN and get dropped
    df = df.interpolate(limit=3, limit_direction="both")
    return df


def season(month):
    if month in (12, 1, 2):
        return 0  # winter
    if month in (3, 4, 5):
        return 1  # summer
    if month in (6, 7, 8, 9):
        return 2  # monsoon
    return 3      # post-monsoon


def add_features(df, lags=(1, 2, 3, 7), horizons=(1, 2)):
    out = df.copy()
    out["dayofweek"] = out.index.dayofweek
    out["month"] = out.index.month
    out["season"] = out["month"].map(season)
    # cyclical encoding so Dec and Jan are "close"
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)

    for lag in lags:
        out[f"AQI_lag{lag}"] = out["AQI"].shift(lag)
    out["AQI_roll7_mean"] = out["AQI"].rolling(7).mean()
    out["AQI_roll7_std"] = out["AQI"].rolling(7).std()

    # Targets: AQI 1 and 2 days ahead (24h and 48h forecast)
    for h in horizons:
        out[f"target_{h}d"] = out["AQI"].shift(-h)
    return out.dropna()


def time_split(df, train=0.7, val=0.15):
    """Chronological split. Never shuffle time series."""
    n = len(df)
    i, j = int(n * train), int(n * (train + val))
    return df.iloc[:i], df.iloc[i:j], df.iloc[j:]


if __name__ == "__main__":
    raw = load_city()
    data = add_features(clean(raw))
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    data.to_csv("data/processed/mumbai_features.csv")
    tr, va, te = time_split(data)
    print(f"rows: {len(data)} | train {len(tr)} | val {len(va)} | test {len(te)}")
    print(data.head())
