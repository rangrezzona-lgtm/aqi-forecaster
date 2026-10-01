"""Baseline models for Mumbai AQI forecasting.

Run data_prep.py first (it creates data/processed/mumbai_features.csv).
Compares two baselines on the test set:
  1. Persistence: "tomorrow's AQI = today's AQI"
  2. Random Forest on the engineered features
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data_prep import time_split

HORIZONS = [1, 2]


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main():
    data = pd.read_csv("data/processed/mumbai_features.csv",
                       index_col="Date", parse_dates=True)
    train, val, test = time_split(data)
    targets = [f"target_{h}d" for h in HORIZONS]
    features = [c for c in data.columns if c not in targets]

    rows = []
    for h in HORIZONS:
        target = f"target_{h}d"

        # 1. Persistence baseline
        pred = test["AQI"]
        rows.append(("Persistence", f"{24 * h}h",
                     mean_absolute_error(test[target], pred),
                     rmse(test[target], pred)))

        # 2. Random Forest
        model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        model.fit(train[features], train[target])
        pred = model.predict(test[features])
        rows.append(("Random Forest", f"{24 * h}h",
                     mean_absolute_error(test[target], pred),
                     rmse(test[target], pred)))

        # Most important features (useful for your README and interviews)
        imp = pd.Series(model.feature_importances_, index=features)
        print(f"\nTop features for {24 * h}h forecast:")
        print(imp.sort_values(ascending=False).head(5).round(3).to_string())

    results = pd.DataFrame(rows, columns=["Model", "Horizon", "MAE", "RMSE"]).round(2)
    print("\nTest set results (lower is better):")
    print(results.to_string(index=False))
    results.to_csv("data/processed/baseline_results.csv", index=False)


if __name__ == "__main__":
    main()
