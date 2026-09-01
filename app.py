"""Streamlit dashboard. Run `python src/forecast.py` first, then `streamlit run app.py`."""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

import waste  # noqa: E402
from data import OUT_DIR, load  # noqa: E402

st.set_page_config(page_title="Steel Plant Energy Analytics", layout="wide")
st.title("Steel plant energy: forecasting and wastage analysis")
st.caption("UCI Steel Industry Energy Consumption dataset (DAEWOO Steel, 2018, 15-min data)")


@st.cache_data
def get_data():
    df = load()
    return df, waste.idle_analysis(df), waste.pf_analysis(df), waste.peak_analysis(df)


df, idle, pf, peak = get_data()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Annual energy", f"{idle['total_kWh'] / 1000:,.0f} MWh")
c2.metric("Max demand", f"{peak['max_kW']:,.0f} kW")
c3.metric("Intervals with PF < 0.90", f"{pf['low_pf_share_%']:.1f}%")
c4.metric("Always-on floor", f"{idle['floor_kW']:.1f} kW")

tab1, tab2, tab3, tab4 = st.tabs(["Forecast", "Waste breakdown", "Power factor", "Recommendations"])

with tab1:
    fpath = OUT_DIR / "forecast_test_day_ahead.csv"
    if not fpath.exists():
        st.warning("Run `python src/forecast.py` to generate forecasts.")
    else:
        f = pd.read_csv(fpath, parse_dates=["date"])
        st.subheader("Day-ahead forecast vs actual (test period Oct-Dec 2018)")
        st.dataframe(pd.read_csv(OUT_DIR / "metrics_day_ahead.csv", index_col=0).round(2))
        dmin, dmax = f["date"].min().date(), f["date"].max().date()
        start = st.date_input("Week starting", value=pd.Timestamp("2018-10-08").date(), min_value=dmin, max_value=dmax)
        wk = f[(f["date"] >= pd.Timestamp(start)) & (f["date"] < pd.Timestamp(start) + pd.Timedelta(days=7))]
        models = st.multiselect("Models", [c for c in f.columns if c not in ("date", "actual")],
                                default=["HistGradientBoosting"])
        st.line_chart(wk.set_index("date")[["actual"] + models])
        st.subheader("Permutation importance (test set)")
        st.bar_chart(pd.read_csv(OUT_DIR / "permutation_importance.csv", index_col=0)["MAE_increase_kWh"])
        st.subheader("15-min-ahead model")
        st.dataframe(pd.read_csv(OUT_DIR / "metrics_15min_ahead.csv", index_col=0).round(2))

with tab2:
    st.subheader("Energy split (MWh/yr)")
    parts = pd.Series({
        "Always-on baseline": idle["baseline_kWh"],
        "Idle above floor": idle["idle_excess_over_floor_kWh"],
        "Production above floor": idle["above_baseline_kWh"] - idle["idle_excess_over_floor_kWh"],
    }) / 1000
    st.bar_chart(parts)
    st.write(f"Idle intervals (< {waste.IDLE_THRESHOLD_KWH:g} kWh): {idle['idle_intervals_share_%']:.1f}% of time, "
             f"{idle['idle_share_%']:.1f}% of energy.")
    col1, col2 = st.columns(2)
    df["hour"] = df["date"].dt.hour
    col1.write("Mean kWh per 15 min by hour")
    col1.bar_chart(df.pivot_table(index="hour", columns="WeekStatus", values="Usage_kWh", aggfunc="mean"))
    col2.write("Energy by load type (MWh)")
    col2.bar_chart(pd.Series(idle["energy_by_load_type_kWh"]) / 1000)
    st.subheader("Peak demand")
    st.write(f"Maximum {peak['max_kW']:.0f} kW at {peak['max_kW_time']}.")
    st.dataframe(pd.DataFrame(peak["caps"]).T.round(1))
    st.bar_chart(pd.Series(peak["monthly_max_kW"], name="Monthly max kW"))

with tab3:
    st.subheader("Power factor (estimate)")
    target = st.slider("Target PF after correction", 0.90, 0.99, 0.95, 0.01)
    pf_t = waste.pf_analysis(df, pf_target=target)
    a, b, c = st.columns(3)
    a.metric("kVArh to compensate", f"{pf_t['kVArh_to_compensate']:,.0f}")
    b.metric("kVAh reduction", f"{pf_t['kVAh_reduction']:,.0f}")
    c.metric("Bank size (P95)", f"{pf_t['bank_kVAr_p95']:.0f} kVAr")
    st.caption("Assumes real power unchanged, fixed voltage; I²R loss ∝ kVA². Not a measured saving.")
    st.write("Lagging PF (%) histogram")
    st.bar_chart(pd.cut(df[waste.LAG_PF], bins=range(0, 105, 5)).value_counts().sort_index().rename(str))
    st.write("Leading kVArh by hour (over-compensation when load drops)")
    st.bar_chart(pd.Series(pf["leading_kVArh_by_hour"]))
    st.scatter_chart(df.sample(5000, random_state=0), x=waste.LAG_KVARH, y=waste.LAG_PF)

with tab4:
    st.subheader("Ranked recommendations")
    rec = waste.recommendations(idle, pf, peak)
    rec["estimate"] = rec["estimate"].round(0)
    st.dataframe(rec, use_container_width=True)