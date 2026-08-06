"""Load the UCI Steel Industry Energy Consumption dataset (id 851) and cache it locally."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_CSV = DATA_DIR / "steel_industry_raw.csv"
FIG_DIR = ROOT / "figures"
OUT_DIR = ROOT / "outputs"


def download() -> pd.DataFrame:
    from ucimlrepo import fetch_ucirepo

    ds = fetch_ucirepo(id=851)
    df = ds.data.original.copy()
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(RAW_CSV, index=False)
    return df


def load_raw() -> pd.DataFrame:
    if RAW_CSV.exists():
        return pd.read_csv(RAW_CSV)
    return download()


def load() -> pd.DataFrame:
    """Return a cleaned frame whose `date` is the START of each 15-min interval.

    The raw timestamps mark the END of the interval, and the midnight row is
    written with the date of the day that is ending (e.g. '01/01/2018 00:00'
    follows '01/01/2018 23:45' and has NSM=0). We move those rows to the next
    day and then shift everything back 15 min, giving a clean regular index
    2018-01-01 00:00 .. 2018-12-31 23:45. The rest of the day's calendar
    columns (Day_of_week, WeekStatus) stay consistent with that interval.
    """
    df = load_raw().copy()
    end = pd.to_datetime(df["date"], format="%d/%m/%Y %H:%M")
    end = end + pd.to_timedelta((df["NSM"] == 0).astype(int), unit="D")
    df["date"] = end - pd.Timedelta(minutes=15)
    return df.sort_values("date").reset_index(drop=True)