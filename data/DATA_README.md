# Data card

## Files

| file | rows | columns | size |
|---|---|---|---|
| `sow_features_minimal.csv` | 24,066 | 17 | ~5.2 MB |
| `sha256sums.txt` | 1 | 2 | < 1 KB |

`sha256sums.txt` contains a single line `<hex>  sow_features_minimal.csv`
matching the SHA-256 of the shipped CSV. Verify after download with
`sha256sum -c sha256sums.txt` (Linux/Mac) or
`Get-FileHash sow_features_minimal.csv -Algorithm SHA256` (Windows).

## Source and scope

The CSV is the publishable subset of an internal 5-minute feature table
covering 14 pregnant sows housed in four indirect-calorimetry chambers
during May-September 2025. Each row is one 5-minute window with the window
right-edge timestamp; HP is the heat production reading from the chamber
gas-exchange controller for that window.

Quality gating was applied upstream:

- `hr_valid == 1` (ECG segment passed R-peak QA);
- `move_coverage >= 0.5` (at least 150 of 300 seconds of IMU data are
  non-missing in the window).

These quality flags are not retained as columns because after filtering they
are constant or near-constant.

A physiological-range trim was then applied to the target:
`11 <= hp_kcal_5min <= 70` kcal / 5 min. The lower bound removes 18 windows
near zero plus a small cluster below 11 kcal that correspond to chamber
transients (door opens for feeding or cleaning, gas-analyser stabilisation
after period change); the upper bound removes 5 windows above 70 kcal that
lie above the typical HP range observed in this cohort under steady-state
recording. The trim drops 73 of 24,139 rows (0.302%).

## Anonymisation

| original | shipped | notes |
|---|---|---|
| chamber + cohort id | `animal_id` | renumbered `S01..S14` by first appearance |
| absolute `datetime` | `time_offset_s` | integer seconds from start of (animal, period) block |
| `period_no_in_cohort` | `period_in_animal` | renumbered per animal, starting at 1 |

Chambers, dates, and original animal labels are not retained. Calendar order
within each animal is preserved.

## Columns

| name | dtype | unit | description |
|---|---|---|---|
| `animal_id` | string | - | anonymised sow id, `S01..S14` |
| `period_in_animal` | int | - | within-animal experimental period index, 1-based |
| `day_in_period` | int | - | day within the period, 1-based |
| `time_offset_s` | int | seconds | offset from the first window of (animal, period) |
| `hr_mean` | float | bpm | mean heart rate in the 5-min window |
| `hr_sd` | float | bpm | standard deviation of instantaneous heart rate |
| `rmssd` | float | ms | root mean square of successive RR differences |
| `pnn50` | float | % | proportion of NN intervals differing by more than 50 ms |
| `gyro_mean` | float | rad/s | mean of per-second mean gyroscope magnitude |
| `angle_sd` | float | rad | mean of per-second std of trunk angle |
| `dz_sd` | float | rad/s^2 | mean of per-second std of vertical angular acceleration |
| `active_fraction` | float | 0-1 | fraction of seconds classified as active |
| `frac_rest` | float | 0-1 | fraction of seconds classified as rest |
| `weight_kg` | float | kg | per-day sow body weight |
| `hour_of_day` | int | 0-23 | hour-of-day for the window right-edge |
| `is_fasting_day` | int | 0/1 | 1 on the last day of each experimental period |
| `hp_kcal_5min` | float | kcal/5 min | heat production measured by indirect calorimetry |

Body weight is per-day, not per-window; the same value repeats for all
windows on a given (animal, date).

## How to load

```python
import pandas as pd
df = pd.read_csv("data/sow_features_minimal.csv")
print(df.shape)                    # (24066, 17)
print(df["animal_id"].nunique())   # 14
```

## License

The CSV and this data card are released under CC BY 4.0; see top-level
`DATA_LICENSE`. Cite the Zenodo archive when redistributing.
