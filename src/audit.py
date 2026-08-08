"""Print a data audit: schema, size, date range, missing values, duplicate timestamps."""
import pandas as pd

from data import load, load_raw


def main():
    raw = load_raw()
    print("== Columns and dtypes (raw CSV) ==")
    print(raw.dtypes.to_string())
    print(f"\nRows: {len(raw)}   Columns: {raw.shape[1]}")
    print("\nFirst rows:")
    print(raw.head(3).to_string())

    print(f"\nRaw first/last timestamp: {raw['date'].iloc[0]} / {raw['date'].iloc[-1]}")
    print("Raw row after '01/01/2018 23:45':", raw.loc[95, ["date", "NSM", "Day_of_week"]].to_dict())
    df = load()
    print("(timestamps converted to interval START; midnight rows moved to next day)")
    print(f"\nDate range: {df['date'].min()}  ->  {df['date'].max()}")
    print("\n== Missing values per column ==")
    print(df.isna().sum().to_string())
    dup = df["date"].duplicated().sum()
    print(f"\nDuplicate timestamps: {dup}")
    step = df["date"].diff().value_counts().head(5)
    print("\nMost common time steps:")
    print(step.to_string())
    full = pd.date_range(df["date"].min(), df["date"].max(), freq="15min")
    print(f"\nExpected 15-min slots in range: {len(full)}; missing slots: {len(full.difference(df['date']))}")
    print("\n== Categorical values ==")
    for c in ["WeekStatus", "Day_of_week", "Load_Type"]:
        print(c, df[c].value_counts().to_dict())
    print("\n== Numeric summary ==")
    print(df.describe().T.round(3).to_string())
    # Check that CO2 is a deterministic function of kWh
    print("\nCO2(tCO2) distinct values:", sorted(df["CO2(tCO2)"].unique()))
    nz = df[df["CO2(tCO2)"] > 0]
    print(f"Implied emission factor on CO2>0 rows (tCO2/MWh): median="
          f"{(nz['CO2(tCO2)'] / nz['Usage_kWh'] * 1000).median():.3f}  (CO2 is rounded to 0.01 t)")
    print(f"Corr(CO2, Usage_kWh) = {df['CO2(tCO2)'].corr(df['Usage_kWh']):.4f}")
    end = df["date"] + pd.Timedelta(minutes=15)
    print("\nNSM == seconds since midnight of interval END:",
          bool((df["NSM"] == end.dt.hour * 3600 + end.dt.minute * 60).all()))
    pf_ok = df["Lagging_Current_Power_Factor"].between(0, 100).all()
    print("Power factors are stored in percent (0-100):", bool(pf_ok))
    print("Total energy in year (kWh):", round(df["Usage_kWh"].sum(), 2))


if __name__ == "__main__":
    main()