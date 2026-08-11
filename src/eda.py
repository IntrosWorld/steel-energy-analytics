"""Exploratory analysis. Saves figures to figures/ and prints the numbers used in the README."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data import FIG_DIR, load

LAG_KVARH = "Lagging_Current_Reactive.Power_kVarh"
LAG_PF = "Lagging_Current_Power_Factor"


def main():
    FIG_DIR.mkdir(exist_ok=True)
    df = load()
    df["hour"] = df["date"].dt.hour
    df["month"] = df["date"].dt.month
    total = df["Usage_kWh"].sum()

    # --- by hour ---
    hourly = df.groupby("hour")["Usage_kWh"].mean()
    print("Mean kWh per 15-min interval by hour of day:")
    print(hourly.round(2).to_string())
    fig, ax = plt.subplots(figsize=(8, 4))
    for ws, g in df.groupby("WeekStatus"):
        g.groupby("hour")["Usage_kWh"].mean().plot(ax=ax, marker="o", label=ws)
    ax.set(xlabel="Hour of day (interval start)", ylabel="Mean kWh per 15 min",
           title="Load profile: weekday vs weekend")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIG_DIR / "hourly_profile.png", dpi=120); plt.close(fig)

    # --- weekday / weekend, day of week ---
    ws = df.groupby("WeekStatus")["Usage_kWh"].agg(["mean", "sum", "count"])
    ws["share_%"] = 100 * ws["sum"] / total
    print("\nBy WeekStatus:"); print(ws.round(2).to_string())
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow = df.groupby("Day_of_week")["Usage_kWh"].mean().reindex(order)
    print("\nMean kWh/interval by day of week:"); print(dow.round(2).to_string())
    fig, ax = plt.subplots(figsize=(7, 4))
    dow.plot.bar(ax=ax, color="steelblue")
    ax.set(ylabel="Mean kWh per 15 min", title="Consumption by day of week", xlabel="")
    fig.tight_layout(); fig.savefig(FIG_DIR / "day_of_week.png", dpi=120); plt.close(fig)

    # --- load type ---
    lt = df.groupby("Load_Type")["Usage_kWh"].agg(["mean", "sum", "count"])
    lt["share_energy_%"] = 100 * lt["sum"] / total
    lt["share_time_%"] = 100 * lt["count"] / len(df)
    print("\nBy Load_Type:"); print(lt.round(2).to_string())
    print("\nLoad_Type by hour (most common label):")
    print(df.groupby("hour")["Load_Type"].agg(lambda s: s.mode()[0]).to_string())
    fig, ax = plt.subplots(figsize=(7, 4))
    df.boxplot(column="Usage_kWh", by="Load_Type", ax=ax)
    ax.set(ylabel="kWh per 15 min", title="Consumption by load type"); fig.suptitle("")
    fig.tight_layout(); fig.savefig(FIG_DIR / "load_type_box.png", dpi=120); plt.close(fig)

    # --- monthly ---
    monthly = df.groupby("month")["Usage_kWh"].sum() / 1000
    print("\nMonthly energy (MWh):"); print(monthly.round(1).to_string())
    fig, ax = plt.subplots(figsize=(7, 4))
    monthly.plot.bar(ax=ax, color="darkorange")
    ax.set(ylabel="MWh", xlabel="Month", title="Monthly energy consumption")
    fig.tight_layout(); fig.savefig(FIG_DIR / "monthly.png", dpi=120); plt.close(fig)

    # --- power factor ---
    active = df[df["Usage_kWh"] > 0]
    pf_calc = 100 * active["Usage_kWh"] / np.hypot(active["Usage_kWh"], active[LAG_KVARH])
    err = (pf_calc - active[LAG_PF]).abs()
    print(f"\nPF check: |PF_calc - PF_reported| median={err.median():.3f}, 99th pct={err.quantile(.99):.3f} (percentage points)")
    print("Lagging PF (%) distribution:"); print(df[LAG_PF].describe().round(2).to_string())
    print(f"Share of intervals with lagging PF < 90%: {100 * (df[LAG_PF] < 90).mean():.2f}%")
    print(f"Rows with Usage_kWh == 0: {(df['Usage_kWh'] == 0).sum()}")
    print("Median lagging PF by load type:"); print(df.groupby("Load_Type")[LAG_PF].median().to_string())
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df[LAG_PF], bins=50, color="seagreen")
    ax.axvline(90, color="red", ls="--", label="0.90 threshold")
    ax.set(xlabel="Lagging power factor (%)", ylabel="Intervals", title="Lagging power factor distribution")
    ax.legend(); fig.tight_layout(); fig.savefig(FIG_DIR / "pf_hist.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sc = ax.scatter(df[LAG_KVARH], df[LAG_PF], c=df["Usage_kWh"], s=3, cmap="viridis", alpha=.5)
    fig.colorbar(sc, label="Usage kWh")
    ax.set(xlabel="Lagging reactive energy (kVArh / 15 min)", ylabel="Lagging PF (%)",
           title="Reactive power vs power factor")
    fig.tight_layout(); fig.savefig(FIG_DIR / "reactive_vs_pf.png", dpi=120); plt.close(fig)
    print(f"\nCorr(lagging kVArh, lagging PF) = {df[LAG_KVARH].corr(df[LAG_PF]):.3f}")
    print(f"Corr(lagging kVArh, Usage_kWh) = {df[LAG_KVARH].corr(df['Usage_kWh']):.3f}")
    print(f"\nFigures written to {FIG_DIR}")


if __name__ == "__main__":
    main()