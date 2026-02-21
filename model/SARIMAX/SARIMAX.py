"""
ARIMAX (ARIMA with eXogenous variables) forecasting model for PM2.5
Uses cleaned_features.csv with selected feature columns.
Outputs:
  - training_validation_loss.png  (train vs validation loss curve)
  - R-squared printed to console
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from itertools import product
# ──────────────────────────────────────────────────────────────
# 1. Load & prepare data
# ──────────────────────────────────────────────────────────────
df = pd.read_csv("cleaned_features.csv", parse_dates=["Date"])
df.sort_values("Date", inplace=True)
df.set_index("Date", inplace=True)

# Map user-friendly names → actual CSV column names
FEATURE_MAP = {
    "PM25_lag1":        "PM2.5_lag1",
    "dayofyear_cos":    "dayofyear_cos",
    "Pres":             "Pres.",
    "Humi":             "Humi.",
    "Vis":              "Vis.",
    "Wind_Speed":       "Wind Speed",
    "PM25_rollmean14":  "PM2.5_rollmean14",
    "dayofyear_sin":    "dayofyear_sin",
    "heatidx":          "heatidx",
    "Prec_rollsum14":   "Prec._rollsum14",
    "Wind_Dir_cos":     "Wind Dir_cos",
    "Wind_Dir_sin":     "Wind Dir_sin",
}

TARGET_COL = "PM2.5"
exog_cols = list(FEATURE_MAP.values())

# Keep only needed columns and drop rows with NaN
df = df[[TARGET_COL] + exog_cols].dropna()

# Reindex to a continuous daily range so the DatetimeIndex carries freq='D'.
# Gaps are forward-filled to keep the series regular for SARIMAX.
full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D")
df = df.reindex(full_range)
df = df.ffill().bfill()          # fill gaps (forward then backward for leading NaNs)
df.index.name = "Date"

# Log-transform PM2.5 to stabilise variance (shifted to avoid log(0))
df[TARGET_COL] = np.log1p(df[TARGET_COL])

y = df[TARGET_COL]
X = df[exog_cols]

print(f"Dataset shape after cleaning: {df.shape}")
print(f"Date range: {df.index.min()} → {df.index.max()}")

# ──────────────────────────────────────────────────────────────
# 2. Train / Validation split  (80 / 20 chronological)
# ──────────────────────────────────────────────────────────────
split_idx = int(len(df) * 0.8)

y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]

# Standardise exogenous features (fit on train only)
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
X_val   = pd.DataFrame(scaler.transform(X_val),       index=X_val.index,   columns=X_val.columns)

print(f"Train size: {len(y_train)}  |  Validation size: {len(y_val)}")

# ──────────────────────────────────────────────────────────────
# 3. Grid search for best ARIMAX order (p, d, q)
# ──────────────────────────────────────────────────────────────
p_range = range(0, 4)
d_range = range(0, 2)
q_range = range(0, 4)

# Seasonal orders to try  (P, D, Q, s)  – weekly cycle s=7
seasonal_choices = [
    (0, 0, 0, 0),          # no seasonality
    (1, 0, 0, 7),
    (0, 0, 1, 7),
    (1, 0, 1, 7),
    (1, 1, 0, 7),
    (0, 1, 1, 7),
    (1, 1, 1, 7),
]

best_aic = np.inf
best_order = (1, 0, 1)
best_seasonal = (0, 0, 0, 0)

total = len(list(product(p_range, d_range, q_range))) * len(seasonal_choices)
current = 0

print(f"\nSearching best (p,d,q)×(P,D,Q,s) via AIC  ({total} combos) …")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for p, d, q in product(p_range, d_range, q_range):
        for sorder in seasonal_choices:
            current += 1
            if current % 50 == 0:
                print(f"  … {current}/{total}", flush=True)
            try:
                model = SARIMAX(
                    y_train,
                    exog=X_train,
                    order=(p, d, q),
                    seasonal_order=sorder,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                res = model.fit(disp=False, maxiter=300)
                if res.aic < best_aic:
                    best_aic = res.aic
                    best_order = (p, d, q)
                    best_seasonal = sorder
            except Exception:
                continue

print(f"Best order: {best_order}  seasonal: {best_seasonal}  (AIC = {best_aic:.2f})")

# ──────────────────────────────────────────────────────────────
# 4. Fit final model with best order
# ──────────────────────────────────────────────────────────────
print("\nFitting final SARIMAX model …")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    final_model = SARIMAX(
        y_train,
        exog=X_train,
        order=best_order,
        seasonal_order=best_seasonal,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    results = final_model.fit(disp=False, maxiter=500)
print(results.summary().tables[0])

# ──────────────────────────────────────────────────────────────
# 5. In-sample (train) & out-of-sample (validation) predictions
# ──────────────────────────────────────────────────────────────
# Training predictions  (in log-space)
y_train_pred = results.fittedvalues

# Validation predictions — walk-forward ONE-STEP-AHEAD
# Each step conditions on all previous *actual* observations,
# so errors do NOT compound across the forecast horizon.
print("\nRunning walk-forward one-step-ahead validation …")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    val_pred_list = []
    history_y = y_train.copy()
    history_X = X_train.copy()
    for i in range(len(y_val)):
        # Re-fit model on all data seen so far
        mdl = SARIMAX(
            history_y,
            exog=history_X,
            order=best_order,
            seasonal_order=best_seasonal,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fit = mdl.fit(disp=False, maxiter=300)
        # Predict exactly 1 step ahead
        pred = fit.forecast(steps=1, exog=X_val.iloc[[i]])
        val_pred_list.append(pred.values[0])
        # Append actual observation to history
        history_y = pd.concat([history_y, y_val.iloc[[i]]])
        history_X = pd.concat([history_X, X_val.iloc[[i]]])
        if (i + 1) % 50 == 0:
            print(f"  … {i+1}/{len(y_val)} steps done", flush=True)

y_val_pred = pd.Series(val_pred_list, index=y_val.index)

# ──────────────────────────────────────────────────────────────
# 6. Convert predictions back from log-space to original scale
# ──────────────────────────────────────────────────────────────
y_train_orig      = np.expm1(y_train)
y_train_pred_orig = np.expm1(y_train_pred)
y_val_orig        = np.expm1(y_val)
y_val_pred_orig   = np.expm1(y_val_pred)

# ──────────────────────────────────────────────────────────────
# 7. Walk-forward cumulative loss curves  (MSE at each step)
# ──────────────────────────────────────────────────────────────
def cumulative_mse(y_true, y_pred):
    """Return array of cumulative MSE at each time step."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    sq_err = (y_true - y_pred) ** 2
    cum_mse = np.cumsum(sq_err) / np.arange(1, len(sq_err) + 1)
    return cum_mse

train_cum_mse = cumulative_mse(y_train_orig, y_train_pred_orig)
val_cum_mse   = cumulative_mse(y_val_orig, y_val_pred_orig)

# ──────────────────────────────────────────────────────────────
# 8. Plot training loss & validation loss
# ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# (a) Cumulative MSE over time
axes[0].plot(y_train_orig.index, train_cum_mse, label="Train Loss (cum. MSE)", color="royalblue")
axes[0].plot(y_val_orig.index,   val_cum_mse,   label="Validation Loss (cum. MSE)", color="crimson")
axes[0].set_xlabel("Date")
axes[0].set_ylabel("Cumulative MSE")
axes[0].set_title("Training vs Validation Loss (Cumulative MSE)")
axes[0].legend()
axes[0].grid(alpha=0.3)

# (b) Rolling-window MSE (window = 30 days)
window = min(30, len(y_val) // 2)
train_sq_err = pd.Series((np.asarray(y_train_orig) - np.asarray(y_train_pred_orig)) ** 2, index=y_train_orig.index)
val_sq_err   = pd.Series((np.asarray(y_val_orig)   - np.asarray(y_val_pred_orig)) ** 2,   index=y_val_orig.index)

axes[1].plot(train_sq_err.rolling(window).mean(), label=f"Train Loss (rolling {window}-day MSE)", color="royalblue")
axes[1].plot(val_sq_err.rolling(window).mean(),   label=f"Val Loss (rolling {window}-day MSE)",   color="crimson")
axes[1].set_xlabel("Date")
axes[1].set_ylabel(f"Rolling {window}-day MSE")
axes[1].set_title(f"Training vs Validation Loss (Rolling {window}-day MSE)")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("training_validation_loss.png", dpi=150)
plt.close()
print("\nLoss plot saved → training_validation_loss.png")

# ──────────────────────────────────────────────────────────────
# 9. R-squared
# ──────────────────────────────────────────────────────────────
r2_train = r2_score(y_train_orig, y_train_pred_orig)
r2_val   = r2_score(y_val_orig, y_val_pred_orig)

rmse_train = np.sqrt(mean_squared_error(y_train_orig, y_train_pred_orig))
rmse_val   = np.sqrt(mean_squared_error(y_val_orig, y_val_pred_orig))

print("\n" + "=" * 50)
print("          SARIMAX  MODEL  RESULTS")
print("=" * 50)
print(f"  Best order (p,d,q)     : {best_order}")
print(f"  Best seasonal (P,D,Q,s): {best_seasonal}")
print(f"  AIC                    : {best_aic:.2f}")
print(f"  Train  R²              : {r2_train:.4f}")
print(f"  Val    R²              : {r2_val:.4f}")
print(f"  Train  RMSE            : {rmse_train:.4f}")
print(f"  Val    RMSE            : {rmse_val:.4f}")
print("=" * 50)
