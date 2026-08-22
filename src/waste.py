"""Wastage and optimisation analysis: idle load, power factor, peak demand, recommendations.

Every figure is computed from the 15-min data. Where an engineering assumption is
needed (target PF, demand cap) it is a named constant below and printed with the result.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import FIG_DIR, OUT_DIR, load

KWH = "Usage_kWh"
LAG_KVARH = "Lagging_Current_Reactive.Power_kVarh"
LEAD_KVARH = "Leading_Current_Reactive_Power_kVarh"
LAG_PF = "Lagging_Current_Power_Factor"

IDLE_THRESHOLD_KWH = 10.0   # gap in the bimodal distribution: idle intervals sit at 2.5-8 kWh, production at 20+
NIGHT_HOURS = range(0, 7)   # 00:00-06:59, used to estimate the always-on floor
PF_LIMIT = 0.90             # "low PF" threshold (lagging)
PF_TARGET = 0.95            # capacitor-bank correction target
INTERVAL_H = 0.25           # 15 min in hours


# ---------------------------------------------------------------- idle / baseline
def idle_analysis(df: pd.DataFrame) -> dict:
    x = df[KWH]
    total = x.sum()
    floor = df.loc[df["date"].dt.hour.isin(NIGHT_HOURS), KWH].median()
    base = np.minimum(x, floor)
    idle = x < IDLE_THRESHOLD_KWH
    idle_excess = (x - floor).clip(lower=0)[idle]
    by_type = x.groupby(df["Load_Type"]).sum()
    return {
        "total_kWh": total,
        "floor_kWh_per_15min": floor,
        "floor_kW": floor / INTERVAL_H,
        "baseline_kWh": base.sum(),
        "above_baseline_kWh": (x - base).sum(),
        "baseline_share_%": 100 * base.sum() / total,
        "idle_intervals": int(idle.sum()),
        "idle_intervals_share_%": 100 * idle.mean(),
        "idle_kWh": x[idle].sum(),
        "active_kWh": x[~idle].sum(),
        "idle_share_%": 100 * x[idle].sum() / total,
        "idle_excess_over_floor_kWh": idle_excess.sum(),
        "light_load_kWh": by_type.get("Light_Load", 0.0),
        "light_load_share_%": 100 * by_type.get("Light_Load", 0.0) / total,
        "energy_by_load_type_kWh": by_type.to_dict(),
        "weekend_idle_kWh": x[idle & (df["WeekStatus"] == "Weekend")].sum(),
        "sunday_kWh": x[df["Day_of_week"] == "Sunday"].sum(),
    }


# ---------------------------------------------------------------- power factor
def pf_analysis(df: pd.DataFrame, pf_limit=PF_LIMIT, pf_target=PF_TARGET) -> dict:
    """Estimate of capacitor correction for intervals below `pf_limit` lagging.

    Assumptions: the metered kWh (real power) is unchanged by correction; the bank
    supplies Q_c = P*(tan(phi1) - tan(phi2)) per interval; I^2R loss scales with kVA^2
    at constant voltage. Actual kWh loss saving needs the feeder/transformer
    resistance, which the dataset does not contain, so it is reported as a relative figure.
    """
    p = df[KWH]
    q = df[LAG_KVARH]
    low = (df[LAG_PF] < pf_limit * 100) & (p > 0)
    tan_t = np.tan(np.arccos(pf_target))
    q_target = p * tan_t
    q_removed = (q - q_target).clip(lower=0)[low]
    kvah_before = np.hypot(p, q)[low]
    kvah_after = np.hypot(p[low], q[low] - q_removed)
    kvar_needed = q_removed / INTERVAL_H
    low_active = low & (p >= IDLE_THRESHOLD_KWH)
    return {
        "pf_limit": pf_limit, "pf_target": pf_target,
        "low_pf_intervals": int(low.sum()),
        "low_pf_share_%": 100 * low.mean(),
        "low_pf_idle_intervals": int((low & (p < IDLE_THRESHOLD_KWH)).sum()),
        "low_pf_active_intervals": int(low_active.sum()),
        "kWh_in_low_pf": p[low].sum(),
        "lagging_kVArh_in_low_pf": q[low].sum(),
        "lagging_kVArh_total": q.sum(),
        "kVArh_to_compensate": q_removed.sum(),
        "kVArh_to_compensate_active_only": (q - q_target).clip(lower=0)[low_active].sum(),
        "kVAh_before": kvah_before.sum(),
        "kVAh_after": kvah_after.sum(),
        "kVAh_reduction": kvah_before.sum() - kvah_after.sum(),
        "I2R_loss_reduction_in_low_pf_intervals_%": 100 * (1 - (kvah_after ** 2).sum() / (kvah_before ** 2).sum()),
        "bank_kVAr_p95": float(kvar_needed.quantile(0.95)),
        "bank_kVAr_max": float(kvar_needed.max()),
        "leading_kVArh_total": df[LEAD_KVARH].sum(),
        "leading_kVArh_by_hour": df.groupby(df["date"].dt.hour)[LEAD_KVARH].sum().to_dict(),
        "leading_kVArh_by_load_type": df.groupby("Load_Type")[LEAD_KVARH].sum().to_dict(),
    }


# ---------------------------------------------------------------- peak demand
def peak_analysis(df: pd.DataFrame, cap_quantiles=(0.99, 0.95)) -> dict:
    kw = df[KWH] / INTERVAL_H
    day = df["date"].dt.date
    res = {
        "max_kW": float(kw.max()),
        "max_kW_time": str(df.loc[kw.idxmax(), "date"]),
        "monthly_max_kW": kw.groupby(df["date"].dt.month).max().to_dict(),
        "top100_by_hour": df.loc[kw.nlargest(100).index, "date"].dt.hour.value_counts().sort_index().to_dict(),
        "top100_by_load_type": df.loc[kw.nlargest(100).index, "Load_Type"].value_counts().to_dict(),
        "maximum_load_kWh": df.loc[df["Load_Type"] == "Maximum_Load", KWH].sum(),
        "caps": {},
    }
    for qt in cap_quantiles:
        cap = float(kw.quantile(qt))
        excess = (kw - cap).clip(lower=0) * INTERVAL_H           # kWh above the cap
        # headroom in same-day Light_Load (off-peak) intervals to absorb the moved energy
        head = ((cap - kw).clip(lower=0) * INTERVAL_H).where(df["Load_Type"] == "Light_Load", 0)
        daily = pd.DataFrame({"excess": excess.groupby(day).sum(), "head": head.groupby(day).sum()})
        res["caps"][f"P{int(qt * 100)}"] = {
            "cap_kW": cap,
            "intervals_above_cap": int((kw > cap).sum()),
            "kWh_to_shift": float(excess.sum()),
            "kWh_to_shift_in_Maximum_Load": float(excess[df["Load_Type"] == "Maximum_Load"].sum()),
            "peak_reduction_kW": float(kw.max() - cap),
            "days_with_excess": int((daily["excess"] > 0).sum()),
            "days_where_offpeak_headroom_suffices": int(((daily["excess"] > 0) & (daily["head"] >= daily["excess"])).sum()),
        }
    return res


# ---------------------------------------------------------------- recommendations
def recommendations(idle, pf, peak) -> pd.DataFrame:
    p95 = peak["caps"]["P95"]
    rows = [
        {
            "recommendation": "Trim idle-period consumption to the night-time floor (switch off auxiliaries/"
                              "lighting/compressors left running outside production)",
            "type": "energy saved",
            "estimate": idle["idle_excess_over_floor_kWh"], "unit": "kWh/yr",
            "method": f"Sum over idle intervals (<{IDLE_THRESHOLD_KWH:g} kWh) of consumption above the "
                      f"night median floor ({idle['floor_kWh_per_15min']:.2f} kWh/15 min)",
        },
        {
            "recommendation": "Audit the always-on base load; every 10% cut in the floor saves this much",
            "type": "energy saved (sensitivity)",
            "estimate": 0.10 * idle["baseline_kWh"], "unit": "kWh/yr per 10% cut",
            "method": f"10% x baseline energy sum(min(kWh, floor)) = 0.10 x {idle['baseline_kWh']:.0f}",
        },
        {
            "recommendation": f"Shift peak-period load so 15-min demand stays below {p95['cap_kW']:.0f} kW "
                              f"(P95); cuts maximum demand by {p95['peak_reduction_kW']:.0f} kW",
            "type": "energy shifted (demand charge)",
            "estimate": p95["kWh_to_shift"], "unit": "kWh/yr moved",
            "method": f"Sum of energy above the cap; same-day off-peak headroom suffices on "
                      f"{p95['days_where_offpeak_headroom_suffices']}/{p95['days_with_excess']} days",
        },
        {
            "recommendation": f"Capacitor bank / APFC to raise lagging PF from <{PF_LIMIT} to {PF_TARGET} "
                              f"(bank ~{pf['bank_kVAr_p95']:.0f} kVAr covers 95% of low-PF intervals)",
            "type": "apparent energy reduced (kVAh billing, I^2R loss)",
            "estimate": pf["kVAh_reduction"], "unit": "kVAh/yr",
            "method": f"Q_c = P(tan phi1 - tan phi2) per low-PF interval; {pf['kVArh_to_compensate']:.0f} kVArh "
                      f"compensated; I^2R loss in those intervals down "
                      f"{pf['I2R_loss_reduction_in_low_pf_intervals_%']:.1f}% (estimate)",
        },
        {
            "recommendation": "Switch capacitor stages off when load drops (lunch 12:00 and end of shift 21:00) "
                              "to stop over-compensation",
            "type": "leading reactive energy avoided",
            "estimate": pf["leading_kVArh_total"], "unit": "kVArh/yr",
            "method": "Upper bound = total leading kVArh metered; largest at 12:00 and 21:00 when the load falls",
        },
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figures + main
def plots(df, idle, pf, peak):
    FIG_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    parts = {"Always-on baseline": idle["baseline_kWh"],
             "Idle above floor": idle["idle_excess_over_floor_kWh"],
             "Production above floor": idle["above_baseline_kWh"] - idle["idle_excess_over_floor_kWh"]}
    ax.bar(parts.keys(), [v / 1000 for v in parts.values()], color=["grey", "tomato", "steelblue"])
    ax.set(ylabel="MWh / year", title="Where the energy goes")
    fig.tight_layout(); fig.savefig(FIG_DIR / "waste_breakdown.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    s = pd.Series(pf["leading_kVArh_by_hour"])
    s.plot.bar(ax=ax, color="purple")
    ax.set(xlabel="Hour", ylabel="Leading kVArh / year", title="Leading reactive energy by hour (over-compensation)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "leading_kvarh_by_hour.png", dpi=120); plt.close(fig)

    kw = (df[KWH] / INTERVAL_H).sort_values(ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(kw.index / len(kw) * 100, kw, color="black")
    for name, c in peak["caps"].items():
        ax.axhline(c["cap_kW"], ls="--", label=f"{name} cap {c['cap_kW']:.0f} kW")
    ax.set(xlabel="% of intervals", ylabel="Demand (kW)", title="Load duration curve")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIG_DIR / "load_duration.png", dpi=120); plt.close(fig)


def _fmt(d):
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()}


def main():
    df = load()
    idle, pf, peak = idle_analysis(df), pf_analysis(df), peak_analysis(df)
    print("== Idle / baseline =="); [print(f"  {k}: {v}") for k, v in _fmt(idle).items()]
    print("\n== Power factor (ESTIMATE) =="); [print(f"  {k}: {v}") for k, v in _fmt(pf).items()]
    print("\n== Peak demand =="); [print(f"  {k}: {v}") for k, v in peak.items()]
    rec = recommendations(idle, pf, peak)
    pd.set_option("display.max_colwidth", 200); pd.set_option("display.width", 250)
    print("\n== Recommendations ==")
    for i, r in rec.iterrows():
        print(f"{i + 1}. {r['recommendation']}\n   -> {r['estimate']:,.0f} {r['unit']} [{r['type']}]\n   method: {r['method']}")
    OUT_DIR.mkdir(exist_ok=True)
    rec.to_csv(OUT_DIR / "recommendations.csv", index=False)
    with open(OUT_DIR / "waste_summary.json", "w") as f:
        json.dump({"idle": idle, "pf": pf, "peak": peak}, f, indent=2, default=float)
    plots(df, idle, pf, peak)


if __name__ == "__main__":
    main()