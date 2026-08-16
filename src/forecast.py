"""Forecast Usage_kWh with a strict time split and leakage-free features.

Two horizons are evaluated:
  * day-ahead (h=96 intervals): every lag is >= 96 steps old, so a forecast for
    all of tomorrow can be made at midnight today. This is the main setting.
  * 15-min-ahead (h=1): lags start at the previous interval.
Train = 2018-01-01 .. 2018-09-30, Test = 2018-10-01 .. 2018-12-31.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from data import FIG_DIR, OUT_DIR, load

TARGET = "Usage_kWh"
SPLIT_DATE = pd.Timestamp("2018-10-01")
WEEK = 96 * 7
LOAD_CODES = {"Light_Load": 0, "Medium_Load": 1, "Maximum_Load": 2}

# Columns measured in the same interval as the target (or derived from it).
LEAKY_COLUMNS = [
    "CO2(tCO2)",                               # computed from the same interval's kWh
    "Lagging_Current_Reactive.Power_kVarh",    # metered in the same interval
    "Leading_Current_Reactive_Power_kVarh",    # metered in the same interval
    "Lagging_Current_Power_Factor",            # = kWh / sqrt(kWh^2 + kVArh^2)
    "Leading_Current_Power_Factor",            # same, leading side
    TARGET,
]
CATEGORICAL = ["hour", "dow", "load_type"]


def time_split(df: pd.DataFrame, split_date=SPLIT_DATE):
    train = df[df["date"] < split_date]
    test = df[df["date"] >= split_date]
    return train, test


def build_features(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Calendar features, load type and lags that are at least `horizon` steps old.

    `df` must be the full, regular, time-sorted series so that shift(k) == k*15 min.
    """
    y = df[TARGET]
    X = pd.DataFrame(index=df.index)
    X["hour"] = df["date"].dt.hour
    X["minute_of_day"] = df["date"].dt.hour * 60 + df["date"].dt.minute
    X["dow"] = df["date"].dt.dayofweek
    X["is_weekend"] = (df["WeekStatus"] == "Weekend").astype(int)
    X["load_type"] = df["Load_Type"].map(LOAD_CODES)

    lags = sorted({horizon, horizon + 1, horizon + 2, 96, 96 * 2, WEEK} | ({1, 2, 4} if horizon == 1 else set()))
    for k in lags:
        if k >= horizon:
            X[f"lag_{k}"] = y.shift(k)
    # same slot on each of the previous 7 days, averaged (all lags >= 96)
    if horizon <= 96:
        X["same_slot_mean_7d"] = pd.concat([y.shift(96 * d) for d in range(1, 8)], axis=1).mean(axis=1)
    X[f"roll_mean_96_from_{horizon}"] = y.shift(horizon).rolling(96).mean()
    X[f"roll_max_96_from_{horizon}"] = y.shift(horizon).rolling(96).max()
    if horizon == 1:
        X["roll_mean_4_from_1"] = y.shift(1).rolling(4).mean()

    assert not set(X.columns) & set(LEAKY_COLUMNS), "leaky column in features"
    return X


def mape(y, p):
    m = y != 0
    return 100 * np.mean(np.abs((y[m] - p[m]) / y[m]))


def smape(y, p):
    d = np.abs(y) + np.abs(p)
    m = d != 0
    return 100 * np.mean(2 * np.abs(y[m] - p[m]) / d[m])


def wape(y, p):
    return 100 * np.sum(np.abs(y - p)) / np.sum(np.abs(y))


def metrics(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {
        "MAE": float(np.mean(np.abs(y - p))),
        "RMSE": float(np.sqrt(np.mean((y - p) ** 2))),
        "MAPE_%": float(mape(y, p)),
        "sMAPE_%": float(smape(y, p)),
        "WAPE_%": float(wape(y, p)),
    }


def make_linear():
    ct = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)], remainder="passthrough"
    )
    return make_pipeline(ct, LinearRegression())


def make_hgb(columns):
    cat_mask = [c in CATEGORICAL for c in columns]
    return HistGradientBoostingRegressor(
        max_iter=500, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40,
        l2_regularization=1.0, categorical_features=cat_mask, random_state=0,
    )


def run(df: pd.DataFrame, horizon: int, importance=False):
    X = build_features(df, horizon)
    valid = X.notna().all(axis=1)
    train_idx = df.index[(df["date"] < SPLIT_DATE) & valid]
    test_idx = df.index[(df["date"] >= SPLIT_DATE) & valid]
    assert df.loc[train_idx, "date"].max() < df.loc[test_idx, "date"].min()

    Xtr, ytr = X.loc[train_idx], df.loc[train_idx, TARGET]
    Xte, yte = X.loc[test_idx], df.loc[test_idx, TARGET]

    preds = {"Seasonal naive (same slot last week)": df[TARGET].shift(WEEK).loc[test_idx]}
    lin = make_linear().fit(Xtr, ytr)
    preds["Linear regression"] = pd.Series(lin.predict(Xte), index=test_idx)
    hgb = make_hgb(list(X.columns)).fit(Xtr, ytr)
    preds["HistGradientBoosting"] = pd.Series(np.clip(hgb.predict(Xte), 0, None), index=test_idx)

    table = pd.DataFrame({name: metrics(yte, p) for name, p in preds.items()}).T
    result = {"table": table, "preds": preds, "y": yte, "dates": df.loc[test_idx, "date"],
              "n_train": len(train_idx), "n_test": len(test_idx), "features": list(X.columns)}
    if importance:
        pi = permutation_importance(hgb, Xte, yte, n_repeats=5, random_state=0,
                                    scoring="neg_mean_absolute_error")
        result["importance"] = pd.DataFrame(
            {"MAE_increase_kWh": pi.importances_mean, "std": pi.importances_std}, index=X.columns
        ).sort_values("MAE_increase_kWh", ascending=False)
    return result


def main():
    OUT_DIR.mkdir(exist_ok=True); FIG_DIR.mkdir(exist_ok=True)
    df = load()
    train, test = time_split(df)
    print(f"Train: {train['date'].min()} -> {train['date'].max()} ({len(train)} rows)")
    print(f"Test : {test['date'].min()} -> {test['date'].max()} ({len(test)} rows)")
    print(f"Excluded (leaky) columns: {LEAKY_COLUMNS[:-1]}")
    pd.set_option("display.width", 200)

    all_tables = {}
    for horizon, label in [(96, "day_ahead"), (1, "15min_ahead")]:
        r = run(df, horizon, importance=(horizon == 96))
        print(f"\n=== Horizon: {label} (h={horizon}) | train rows used={r['n_train']}, test rows={r['n_test']} ===")
        print("Features:", r["features"])
        print(r["table"].round(3).to_string())
        all_tables[label] = r["table"]
        r["table"].round(4).to_csv(OUT_DIR / f"metrics_{label}.csv")

        y = r["y"]
        low = y < 10
        print(f"Test intervals with actual < 10 kWh: {low.mean() * 100:.1f}%  "
              f"(MAPE on these = {mape(y[low].values, r['preds']['HistGradientBoosting'][low].values):.1f}%, "
              f"on >=10 kWh = {mape(y[~low].values, r['preds']['HistGradientBoosting'][~low].values):.1f}%)")

        if horizon == 96:
            out = pd.DataFrame({"date": r["dates"], "actual": y})
            for name, p in r["preds"].items():
                out[name] = p
            out.to_csv(OUT_DIR / "forecast_test_day_ahead.csv", index=False)
            imp = r["importance"]
            print("\nPermutation importance (HGB, test set, increase in MAE in kWh):")
            print(imp.round(3).to_string())
            imp.round(4).to_csv(OUT_DIR / "permutation_importance.csv")

            fig, ax = plt.subplots(figsize=(8, 4.5))
            imp["MAE_increase_kWh"][::-1].plot.barh(ax=ax, xerr=imp["std"][::-1], color="slateblue")
            ax.set(xlabel="Increase in test MAE when shuffled (kWh)", title="Permutation importance (day-ahead HGB)")
            fig.tight_layout(); fig.savefig(FIG_DIR / "permutation_importance.png", dpi=120); plt.close(fig)

            wk = out[(out["date"] >= "2018-10-08") & (out["date"] < "2018-10-15")]
            fig, ax = plt.subplots(figsize=(11, 4))
            ax.plot(wk["date"], wk["actual"], label="Actual", lw=1.2, color="black")
            ax.plot(wk["date"], wk["HistGradientBoosting"], label="HGB day-ahead", lw=1)
            ax.plot(wk["date"], wk["Seasonal naive (same slot last week)"], label="Seasonal naive", lw=1, alpha=.6)
            ax.set(ylabel="kWh per 15 min", title="Test week 8-14 Oct 2018: forecast vs actual")
            ax.legend(); ax.grid(alpha=.3)
            fig.tight_layout(); fig.savefig(FIG_DIR / "forecast_vs_actual.png", dpi=120); plt.close(fig)

    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump({k: v.round(4).to_dict(orient="index") for k, v in all_tables.items()}, f, indent=2)


if __name__ == "__main__":
    main()