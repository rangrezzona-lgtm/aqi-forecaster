# Mumbai AQI Forecaster

Forecasting Mumbai's Air Quality Index (AQI) **24 and 48 hours ahead** using Random Forest and LSTM, compared against a simple persistence baseline.

## Problem
Air quality affects public health. Can we predict tomorrow's and the day after tomorrow's AQI in Mumbai from recent pollution readings, and does a deep learning model (LSTM) actually beat simpler approaches?

## Data
- **Source:** [Air Quality Data in India (2015-2020)](https://www.kaggle.com/) on Kaggle, file `city_day.csv` (daily readings, CPCB stations).
- **Mumbai coverage:** AQI values exist only from May 2018 to July 2020, giving about 762 usable days after cleaning.
- **Pollutants used:** PM2.5, PM10, NO2, SO2, CO, O3, plus AQI.

## Method
1. **Cleaning:** put every calendar day on the index, interpolate gaps of up to 3 days, drop longer gaps.
2. **Features:** AQI lags (1, 2, 3, 7 days), 7-day rolling mean and standard deviation, day of week, month (sin/cos), season.
3. **Targets:** AQI 1 day ahead (24h) and 2 days ahead (48h).
4. **Split:** chronological 70% train / 15% validation / 15% test. Never shuffled, to avoid leaking the future into training.
5. **Models:**
   - *Persistence:* tomorrow's AQI equals today's.
   - *Random Forest:* 200 trees on the engineered features.
   - *LSTM (PyTorch):* last 14 days as input. It predicts the **change from today's AQI**, not the raw AQI, with early stopping on the validation set.

## Results (test set, lower is better)

| Horizon | Model | MAE | RMSE |
|---|---|---|---|
| 24h | Persistence | 8.29 | 12.56 |
| 24h | **Random Forest** | **7.58** | **10.35** |
| 24h | LSTM | 8.32 | 12.13 |
| 48h | Persistence | 12.30 | 18.97 |
| 48h | Random Forest | 13.07 | **17.62** |
| 48h | LSTM | **12.17** | 17.91 |

## Key findings
- At 24h the Random Forest is the best model, about 9% lower MAE than persistence.
- At 48h the three models are close. The LSTM's MAE advantage over persistence (0.13) is too small to call a real win on 115 test days from a single run.
- The first LSTM version, which predicted raw AQI, scored MAE 14.58 at 24h, worse than persistence. Predicting the change from today's AQI fixed most of that.
- Most important features: PM10 for the 24h forecast, the 7-day average AQI for the 48h forecast.

## Limitations
- Only about two years of Mumbai data, which is small for a neural network.
- The test period (early 2020) overlaps the COVID-19 lockdown, when pollution dropped sharply. The models had not seen conditions like this in training.
- Results come from a single train/test split and a single LSTM run, so differences are not statistically tested.

## Project structure
```
src/
  data_prep.py     # load, clean, features, time split
  baselines.py     # persistence and Random Forest
  lstm_model.py    # LSTM forecaster
```
The raw data is not included. Download `city_day.csv` from Kaggle and place it in `data/raw/`.

## How to run
```bash
pip install pandas numpy scikit-learn torch
python src/data_prep.py
python src/baselines.py
python src/lstm_model.py
```

## Next steps
- Streamlit dashboard showing actual vs predicted AQI
- Live data from the OpenAQ API
- Evaluate with rolling-window cross-validation and more cities

## Author
Zona Sameer Rangrez
