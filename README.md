# Steel Plant Energy Analytics: Forecasting and Wastage Analysis

A portfolio project that uses 15-minute electricity data from a real steel plant to cover two problem statements:

- **SAIL #12, Energy Consumption Forecasting & Optimisation:** forecast consumption, relate it to operating parameters (load type, time of day, weekday/weekend, power factor) and recommend ways to reduce energy use.
- **SAIL #13, Plant-wise Gas/Energy Consumption Analysis:** find wastage points (idle load, low power factor, over-compensation, demand peaks) and estimate how much consumption could be cut.

Every number below was printed by the scripts in `src/`. The full logs are in `outputs/*.txt`.

---

## 1. Data

**Source:** Sathishkumar V E, Changsun Shin, Yongyun Cho. *Steel Industry Energy Consumption*. UCI Machine Learning Repository, id 851 (2021), DOI [10.24432/C52G8C](https://doi.org/10.24432/C52G8C).
Data from DAEWOO Steel Co. Ltd, Gwangyang, South Korea (via KEPCO), 2018.
Intro paper: *Efficient energy consumption prediction model for a data analytic-enabled industry building in a smart city*, Building Research & Information 49(1), 127–143, 2021.
**Licence:** Creative Commons Attribution 4.0 International (CC BY 4.0).

`src/data.py` downloads the data with `ucimlrepo` and caches it in `data/`.

### Audit (`python src/audit.py` → `outputs/audit.txt`)

| Item | Result |
|---|---|
| Rows × columns | 35,040 × 11 |
| Date range (interval start, after fix below) | 2018-01-01 00:00 → 2018-12-31 23:45 |
| Missing values | 0 in every column |
| Duplicate timestamps | 0 |
| Missing 15-min slots | 0 (35,040 expected, 35,040 present; every step is exactly 15 min) |
| Total energy | 959,636.71 kWh |

| Column | dtype | Notes |
|---|---|---|
| `date` | object | `dd/mm/yyyy HH:MM`, marks the **end** of the interval |
| `Usage_kWh` | float64 | target; 0 – 157.18, mean 27.39 |
| `Lagging_Current_Reactive.Power_kVarh` | float64 | 0 – 96.91 |
| `Leading_Current_Reactive_Power_kVarh` | float64 | 0 – 27.76 |
| `CO2(tCO2)` | float64 | only 8 distinct values (0.00 … 0.07), corr with kWh = 0.9882 |
| `Lagging_Current_Power_Factor` | float64 | in **percent** (0–100) |
| `Leading_Current_Power_Factor` | float64 | in percent |
| `NSM` | int64 | seconds since midnight of the interval end |
| `WeekStatus` | object | Weekday 25,056 / Weekend 9,984 |
| `Day_of_week` | object | Monday 5,088, other days 4,992 each |
| `Load_Type` | object | Light 18,072 / Medium 9,696 / Maximum 7,272 |

**Timestamp quirk:** the raw file starts at `01/01/2018 00:15`. The midnight row after `01/01/2018 23:45` is written as `01/01/2018 00:00` (NSM = 0) even though it is the end of that day. Sorting the raw timestamps as written would therefore misplace 365 rows. `data.load()` moves midnight rows to the next day and shifts every timestamp back 15 minutes, so each row is labelled with the **start** of its interval. The test suite checks that the resulting index is regular.

**Sanity checks:**
- The reported lagging PF equals `kWh / sqrt(kWh² + kVArh²)`: median absolute error 0.002 percentage points, 99th percentile 0.005.
- CO₂ implies about 0.461 tCO₂/MWh on non-zero rows, so it is just a rounded multiple of kWh.

## 2. EDA (`python src/eda.py` → `figures/`, `outputs/eda.txt`)

- **Two-shift daily profile:** mean consumption per 15 minutes is about 4.2–5.2 kWh from 01:00 to 07:59, then 51–58 kWh from 08:00 to 11:59. It drops to 7.1 kWh at the 12:00 lunch break, is 51–56 kWh from 13:00 to 16:59 and 33.6–38.1 kWh from 17:00 to 20:59, then falls to about 8–10 kWh.
- **Weekdays vs weekends:** weekdays average 33.62 kWh per interval, weekends 11.73. Sunday averages 7.55. Weekdays use 87.79% of the energy.
- **`Load_Type` is a time-of-day (tariff-style) label, not a measurement:**

  | Load type | Share of time | Share of energy | Mean kWh |
  |---|---|---|---|
  | Light | 51.58% | 16.24% | 8.63 |
  | Medium | 27.67% | 38.84% | 38.45 |
  | Maximum | 20.75% | 44.91% | 59.27 |

- **Monthly energy:** highest in January (126.2 MWh), lowest in September (57.9 MWh).
- **Power factor:**
  - The median lagging PF is 87.96%, and **54.81% of intervals are below 90%**.
  - Median PF by load type: Light 66.3%, Medium 96.8%, Maximum 91.7%. Poor PF is mostly a light-load (idle) problem, where magnetising current dominates.
- **Reactive energy:** lagging kVArh correlates strongly with kWh (0.896) but only weakly with PF (0.145). High reactive energy comes with high load, where PF is fine; low PF happens at small loads.

Figures: `hourly_profile.png`, `day_of_week.png`, `load_type_box.png`, `monthly.png`, `pf_hist.png`, `reactive_vs_pf.png`.

## 3. Forecasting (`python src/forecast.py`)

### Setup
- **Time split only:**
  - Train: 2018-01-01 → 2018-09-30 (26,208 rows; 25,536 remain after dropping the first week, which has no lag-672 value).
  - Test: 2018-10-01 → 2018-12-31 (8,832 rows).
- **Two horizons:**
  - **Day-ahead (main):** every lag is at least 96 intervals (24 h) old, so all of tomorrow can be forecast at midnight.
  - **15-min-ahead:** lags start at the previous interval.
- **Features:**
  - Calendar: hour, minute of day, day of week, weekend flag.
  - `Load_Type`: known in advance because it follows the clock.
  - Lagged `Usage_kWh`:
    - day-ahead: lags 96, 97, 98, 192 and 672; the mean of the same slot over the previous 7 days; a 96-interval rolling mean and max ending 24 h ago
    - 15-min-ahead: adds lags 1–4 and a short rolling mean
- **Models:**
  - Seasonal naive: the same slot last week (lag 672).
  - Linear regression: one-hot hour, day of week and load type.
  - `HistGradientBoostingRegressor`: 500 iterations, learning rate 0.05, with categorical support.

### Leakage decisions: excluded columns
| Column | Why it is excluded |
|---|---|
| `CO2(tCO2)` | Computed from the same interval's kWh (corr 0.988). Using it amounts to using the target. |
| `Lagging_Current_Reactive.Power_kVarh` | Metered over the same 15 minutes as the target and tracks it closely (corr 0.896). It is not known before the interval ends. |
| `Leading_Current_Reactive_Power_kVarh` | Same reason. |
| `Lagging_Current_Power_Factor` | Derived from the same interval's kWh and kVArh. |
| `Leading_Current_Power_Factor` | Same reason. |
| `NSM`, `WeekStatus`, `Day_of_week` | Not leaky. Replaced by equivalent features taken from the corrected timestamp. |

`tests/test_pipeline.py` checks two things:
- None of these columns reach the model.
- Overwriting the target within the forecast horizon, and every leaky column, leaves earlier feature rows unchanged.

### Results (test set, Oct–Dec 2018, 8,832 intervals)

**Day-ahead (h = 96)**

| Model | MAE (kWh) | RMSE (kWh) | MAPE % | sMAPE % | WAPE % |
|---|---|---|---|---|---|
| Seasonal naive (same slot last week) | 12.942 | 24.771 | 95.197 | 37.874 | 49.630 |
| Linear regression | 12.616 | 18.575 | 164.586 | 82.057 | 48.379 |
| **HistGradientBoosting** | **10.113** | **17.988** | 76.483 | 42.546 | **38.780** |

**15-min-ahead (h = 1)**

| Model | MAE (kWh) | RMSE (kWh) | MAPE % | sMAPE % | WAPE % |
|---|---|---|---|---|---|
| Seasonal naive (same slot last week) | 12.942 | 24.771 | 95.197 | 37.874 | 49.630 |
| Linear regression | 6.444 | 11.008 | 61.878 | 45.775 | 24.710 |
| **HistGradientBoosting** | **4.218** | **8.785** | **16.830** | **13.518** | **16.174** |

### Reading the results honestly
- **Day-ahead gain over the seasonal baseline:** real but moderate. MAE falls by 21.9% (12.94 → 10.11 kWh), RMSE by 27.4% and WAPE from 49.6% to 38.8%. Without knowing the next day's production schedule, a day-ahead model cannot tell whether a given shift will run or how hard (see the Oct 12–14 misses in `figures/forecast_vs_actual.png`).
- **MAPE is misleading here:**
  - 61.5% of test intervals are below 10 kWh (idle), so small absolute errors on 3 kWh loads become huge percentages.
  - Day-ahead HGB MAPE is 103.8% on intervals below 10 kWh but 32.8% on intervals of 10 kWh or more.
  - The one interval with exactly 0 kWh is excluded from MAPE.
  - **WAPE (total absolute error ÷ total energy) is the headline metric**, with sMAPE shown alongside.
- **Linear regression:** its high MAPE and sMAPE come from predicting negative values in 14.6% of test intervals, with a median of 6.85 kWh when the actual is below 10. A straight-line model cannot switch sharply between idle and production.
- **15-min-ahead:** much better (WAPE 16.2%) because the previous interval shows whether the line is running. It is useful for real-time demand control, less so for planning.

### Permutation importance (day-ahead HGB, test set, increase in MAE in kWh, 5 repeats)
| Feature | ΔMAE | Feature | ΔMAE |
|---|---|---|---|
| load_type | 7.265 | minute_of_day | 0.509 |
| lag_96 (same time yesterday) | 3.962 | roll_mean_96_from_96 | 0.475 |
| same_slot_mean_7d | 2.876 | roll_max_96_from_96 | 0.409 |
| lag_672 (same time last week) | 1.888 | lag_192 | 0.366 |
| dow | 1.822 | is_weekend | 0.077 |
| hour | 1.774 | lag_98 | 0.059 |
| lag_97 | 0.803 | | |

The shift schedule (load type, hour, day) and yesterday's consumption carry most of the signal.

## 4. Wastage and optimisation (`python src/waste.py`)

### Idle and baseline load
- **Always-on floor:** the median night-time (00:00–06:59) consumption is **3.17 kWh per 15 min, about 12.68 kW**. It runs all year:
  - baseline energy = Σ min(kWh, floor) = **108,795 kWh (11.34% of the total)**
  - energy above the floor = 850,841 kWh
- **Idle intervals (< 10 kWh):** 20,899 intervals (**59.6% of the time**), using **77,343 kWh (8.06% of the energy)**. Weekend idle energy is 28,465 kWh.
- **Idle excess:** energy above the floor inside idle intervals is **13,375 kWh**.
- **Light_Load periods:** 155,893 kWh (16.24%). This includes production that starts at 08:00, which is still a light-load period, so it overstates waste.

### Low power factor (ESTIMATE)
**Assumptions:**
- Real power is unchanged by correction.
- The bank supplies Q_c = P·(tan φ₁ − tan φ₂) in each interval.
- Voltage is constant, so I²R loss scales with kVA².
- Line and transformer resistance are not in the data, so the absolute kWh loss saving cannot be computed. The results below cover kVArh, kVAh and the *relative* loss.

**Low-PF intervals (lagging PF < 0.90):**
- 19,206 intervals (54.81%), carrying 397,554 kWh and 279,401 kVArh (out of 456,760 lagging kVArh in total).
- 13,703 of them are idle intervals and 5,503 are production intervals.

**Correcting them to PF 0.95:**
- Reactive energy to compensate: **148,731 kVArh** (100,452 kVArh in production intervals only).
- Apparent energy: 493,033 → 418,478 kVAh, a **74,555 kVAh reduction**. This matters under kVAh-based tariffs, which many Indian state utilities use for HT consumers.
- I²R loss in those intervals: **down 19.9%**.
- Bank size: **~100 kVAr** covers 95% of the low-PF intervals; the worst interval needs 221 kVAr. An automatic PF controller (APFC) with stages suits this better than one fixed bank.

**Over-compensation:** **135,638 kVArh of leading reactive energy** was metered. It peaks at **12:00 (26,102 kVArh) and 21:00 (25,341 kVArh)**, exactly when the load drops for lunch and at the end of the shift. This suggests capacitors stay connected when the inductive load goes away.

### Peak demand
- **Maximum demand:** 628.72 kW, on 2018-11-22 at 09:30. Monthly maxima range from 486.7 kW (July) to 628.7 kW (November).
- **Top 100 demand intervals:**
  - by load type: 48 Maximum_Load, 42 Medium_Load, 10 Light_Load
  - by time: spread over 08:00–11:59 and 13:00–16:59

| Demand cap | Cap (kW) | Intervals above | kWh to shift/yr | …of which in Maximum_Load | Peak cut (kW) | Days affected (all with enough same-day off-peak headroom) |
|---|---|---|---|---|---|---|
| P99 | 490.8 | 351 | 2,903 | 1,353 | 137.9 | 86 |
| P95 | 396.2 | 1,751 | 25,344 | 12,220 | 232.6 | 181 |

This energy is **moved, not saved**. The benefit is lower maximum-demand charges and less stress on the transformer. The headroom check only compares energy; it ignores process sequencing, so the P95 case is an upper bound.

### Ranked recommendations (`outputs/recommendations.csv`)
| # | Recommendation | Estimate | Type | Method |
|---|---|---|---|---|
| 1 | Cut idle-period consumption to the night-time floor (auxiliaries, lighting and compressors left on outside production) | **13,375 kWh/yr** | energy saved | Σ over idle intervals of (kWh − 3.17) |
| 2 | Audit the always-on 12.7 kW base load | **10,880 kWh/yr per 10% cut** | energy saved (sensitivity) | 0.10 × 108,795 kWh baseline |
| 3 | Shift peak-period load to keep demand ≤ 396 kW (P95) | **25,344 kWh/yr moved**, −233 kW max demand (P99 cap: 2,903 kWh, −138 kW) | demand charge | Σ energy above the cap |
| 4 | Staged capacitor bank / APFC to raise PF from below 0.90 to 0.95 (~100 kVAr) | **74,555 kVAh/yr** less apparent energy; I²R loss −19.9% in those intervals | kVAh billing / losses (estimate) | Q_c = P(tan φ₁ − tan φ₂) |
| 5 | Switch capacitor stages off when load drops (12:00, 21:00) | **up to 135,638 kVArh/yr** leading energy avoided | over-compensation | total metered leading kVArh |

Ranking: items 1 and 2 are true kWh savings. Items 3 to 5 reduce cost and electrical stress but do not directly reduce kWh.

## 5. Dashboard
`streamlit run app.py` has four tabs:
- **Forecast:** forecast vs actual for any test week, the metrics tables and permutation importance.
- **Waste breakdown:** the energy split, hourly profile, load-type energy and peak-demand caps.
- **Power factor:** a slider for the target PF that recomputes the kVAr estimate, plus the PF histogram, leading kVArh by hour and a reactive-vs-PF scatter.
- **Recommendations:** the ranked recommendations table.

## 6. Tests
`python -m pytest -q` → **10 passed**. The tests check that:
- the index is regular and 15 minutes apart
- the train/test split never overlaps
- no leaky column reaches the model, and no lag is shorter than the forecast horizon
- overwriting the future target or the leaky columns leaves earlier features unchanged
- MAE, RMSE, MAPE, sMAPE and WAPE match hand-computed values
- the idle/baseline/load-type energy splits add up to the total
- the PF correction reaches 0.95 in every low-PF interval and never increases kVAh
- the peak-shift figures stay within the total

## 7. Limitations
- **One plant, one year (2018):** no year-over-year seasonality, and the 3-month test period (Oct–Dec) is lower-load than training (61.5% idle intervals).
- **No production data:** there is no tonnage, so **no true specific energy consumption (kWh/t)** can be computed. Idle energy and the base load are the closest proxies. With tonnage, SEC per shift or product would be the main KPI for SAIL #12.
- **Electricity only:** SAIL #13 also covers gas (BF, CO and LD gas), which this dataset does not include.
- **No sub-metering:** the data is a single plant-level meter, so waste can be linked to *time periods* but not to specific equipment.
- **PF savings are estimates:** there is no network impedance or tariff data, so they are reported in kVArh, kVAh and relative I²R terms only.
- **`Load_Type` is inferred to be a clock-based label:** this is based on its pattern by hour. If it were set from measured load, it would be leaky for forecasting.
- **Rounding:** CO₂ is rounded to 0.01 t and cannot support a meaningful per-interval emissions analysis.

## 8. How to run
```bash
pip install -r requirements.txt
python src/audit.py      # downloads data on first run, prints the audit
python src/eda.py        # figures/
python src/forecast.py   # metrics + predictions in outputs/
python src/waste.py      # waste summary + recommendations
python -m pytest -q
streamlit run app.py
```
Tested with Python 3.13.2 on Windows.

## Layout
```
src/data.py       download, cache, timestamp fix
src/audit.py      data audit
src/eda.py        exploratory figures
src/forecast.py   features, split, models, metrics, importance
src/waste.py      idle, PF, peak analysis, recommendations
app.py            Streamlit dashboard
tests/            pytest suite
figures/          saved PNGs
outputs/          printed logs, metrics, predictions, recommendations
```