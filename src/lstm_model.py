"""LSTM forecaster for Mumbai AQI (24h and 48h ahead).

Input : last 14 days of pollutants + AQI + month (sin/cos)
Output: AQI 1 day ahead and 2 days ahead

Run data_prep.py and baselines.py first.
"""
import random

import numpy as np
import pandas as pd

from data_prep import time_split

WINDOW = 14
INPUT_COLS = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "AQI",
              "month_sin", "month_cos"]
TARGET_COLS = ["target_1d", "target_2d"]
SEED = 42


def make_windows(df, window=WINDOW):
    """Return X, y and the end-date of each window.

    A window is only kept if all `window` days are consecutive, so the
    model never sees a sequence that jumps over a gap in the data.
    """
    X, y, dates = [], [], []
    values = df[INPUT_COLS].values
    targets = df[TARGET_COLS].values
    idx = df.index
    for end in range(window - 1, len(df)):
        start = end - window + 1
        span = (idx[end] - idx[start]).days
        if span != window - 1:
            continue
        X.append(values[start:end + 1])
        y.append(targets[end])
        dates.append(idx[end])
    return np.array(X, dtype="float32"), np.array(y, dtype="float32"), pd.DatetimeIndex(dates)


def main():
    import torch
    import torch.nn as nn
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    data = pd.read_csv("data/processed/mumbai_features.csv",
                       index_col="Date", parse_dates=True)
    train_df, val_df, test_df = time_split(data)

    X, y, dates = make_windows(data)

    # Predict the CHANGE from today's AQI, not the raw AQI. The model then
    # starts from "tomorrow = today" and only has to learn the difference.
    base = X[:, -1, INPUT_COLS.index("AQI")][:, None]
    y_raw = y
    y = y - base

    tr = dates.isin(train_df.index)
    va = dates.isin(val_df.index)
    te = dates.isin(test_df.index)
    print(f"windows: train {tr.sum()} | val {va.sum()} | test {te.sum()}")

    # Scale using TRAIN statistics only (no peeking at val/test)
    x_mean = X[tr].reshape(-1, X.shape[2]).mean(axis=0)
    x_std = X[tr].reshape(-1, X.shape[2]).std(axis=0) + 1e-8
    y_mean, y_std = y[tr].mean(axis=0), y[tr].std(axis=0) + 1e-8
    Xs = (X - x_mean) / x_std
    ys = (y - y_mean) / y_std

    def to_t(a):
        return torch.tensor(a, dtype=torch.float32)

    Xtr, ytr = to_t(Xs[tr]), to_t(ys[tr])
    Xva, yva = to_t(Xs[va]), to_t(ys[va])
    Xte = to_t(Xs[te])

    class AQILSTM(nn.Module):
        def __init__(self, n_in, hidden=32, n_out=2):
            super().__init__()
            self.lstm = nn.LSTM(n_in, hidden, batch_first=True)
            self.drop = nn.Dropout(0.2)
            self.fc = nn.Linear(hidden, n_out)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(self.drop(out[:, -1, :]))

    model = AQILSTM(Xs.shape[2])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    best_val, best_state, patience, wait = float("inf"), None, 25, 0
    for epoch in range(300):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 32):
            b = perm[i:i + 32]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[b]), ytr[b])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xva), yva).item()
        if val_loss < best_val:
            best_val, wait = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                print(f"early stop at epoch {epoch + 1}")
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(Xte).numpy() * y_std + y_mean + base[te]
    truth = y_raw[te]

    rows = []
    for k, h in enumerate([24, 48]):
        mae = mean_absolute_error(truth[:, k], pred[:, k])
        rmse = float(np.sqrt(mean_squared_error(truth[:, k], pred[:, k])))
        rows.append(("LSTM", f"{h}h", round(mae, 2), round(rmse, 2)))
    lstm = pd.DataFrame(rows, columns=["Model", "Horizon", "MAE", "RMSE"])

    base_res = pd.read_csv("data/processed/baseline_results.csv")
    allres = pd.concat([base_res, lstm]).sort_values(["Horizon", "MAE"])
    print("\nTest set results (lower is better):")
    print(allres.to_string(index=False))
    allres.to_csv("data/processed/all_results.csv", index=False)

    # Save predictions for the dashboard
    out = pd.DataFrame({"actual_24h": truth[:, 0], "pred_24h": pred[:, 0],
                        "actual_48h": truth[:, 1], "pred_48h": pred[:, 1]},
                       index=dates[te])
    out.index.name = "Date"
    out.to_csv("data/processed/lstm_predictions.csv")


if __name__ == "__main__":
    main()
