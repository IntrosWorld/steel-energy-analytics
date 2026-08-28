import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import forecast as fc  # noqa: E402
import waste  # noqa: E402
from data import load  # noqa: E402


@pytest.fixture(scope="module")
def df():
    return load()


def test_index_is_regular_15min(df):
    assert len(df) == 35040
    assert df["date"].is_unique
    assert (df["date"].diff().dropna() == pd.Timedelta(minutes=15)).all()


def test_time_split_does_not_overlap(df):
    train, test = fc.time_split(df)
    assert len(train) + len(test) == len(df)
    assert train["date"].max() < test["date"].min()
    assert not set(train["date"]) & set(test["date"])


@pytest.mark.parametrize("horizon", [1, 96])
def test_no_leaky_columns_in_features(df, horizon):
    X = fc.build_features(df, horizon)
    assert not set(X.columns) & set(fc.LEAKY_COLUMNS)
    for col in X.columns:
        if col.startswith("lag_"):
            assert int(col.split("_")[1]) >= horizon


@pytest.mark.parametrize("horizon", [1, 96])
def test_features_do_not_see_recent_target(df, horizon):
    """Changing y at rows >= t - horizon + 1 must not change features at row t."""
    t = 20000
    X1 = fc.build_features(df, horizon)
    d2 = df.copy()
    d2.loc[t - horizon + 1:, fc.TARGET] = 9999.0
    for c in fc.LEAKY_COLUMNS[:-1]:
        d2[c] = -1.0
    X2 = fc.build_features(d2, horizon)
    pd.testing.assert_frame_equal(X1.loc[:t], X2.loc[:t])


def test_metrics_known_values():
    y = np.array([10.0, 20.0, 0.0, 40.0])
    p = np.array([12.0, 18.0, 1.0, 40.0])
    m = fc.metrics(y, p)
    assert m["MAE"] == pytest.approx(5 / 4)
    assert m["RMSE"] == pytest.approx(np.sqrt(9 / 4))
    assert m["MAPE_%"] == pytest.approx(100 * (0.2 + 0.1 + 0) / 3)   # zero actual excluded
    assert m["WAPE_%"] == pytest.approx(100 * 5 / 70)
    assert m["sMAPE_%"] == pytest.approx(100 * (4 / 22 + 4 / 38 + 2 + 0) / 4)


def test_idle_figures_add_up(df):
    r = waste.idle_analysis(df)
    total = df["Usage_kWh"].sum()
    assert r["total_kWh"] == pytest.approx(total)
    assert r["baseline_kWh"] + r["above_baseline_kWh"] == pytest.approx(total)
    assert r["idle_kWh"] + r["active_kWh"] == pytest.approx(total)
    assert sum(r["energy_by_load_type_kWh"].values()) == pytest.approx(total)
    assert 0 <= r["idle_excess_over_floor_kWh"] <= r["idle_kWh"]


def test_pf_estimate_is_consistent(df):
    r = waste.pf_analysis(df)
    assert r["lagging_kVArh_in_low_pf"] <= r["lagging_kVArh_total"]
    assert 0 <= r["kVArh_to_compensate"] <= r["lagging_kVArh_in_low_pf"]
    assert r["kVAh_after"] <= r["kVAh_before"]
    # after correction every low-PF interval should sit at the target PF
    p = df["Usage_kWh"]; q = df[waste.LAG_KVARH]
    low = (df[waste.LAG_PF] < 90) & (p > 0)
    q_new = np.minimum(q, p * np.tan(np.arccos(0.95)))[low]
    pf_new = p[low] / np.hypot(p[low], q_new)
    assert (pf_new >= 0.95 - 1e-9).all()


def test_peak_shift_bounded(df):
    r = waste.peak_analysis(df)
    total = df["Usage_kWh"].sum()
    for c in r["caps"].values():
        assert 0 < c["kWh_to_shift"] < total
        assert c["kWh_to_shift_in_Maximum_Load"] <= c["kWh_to_shift"]