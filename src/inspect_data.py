from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/raw/day_ahead_prices.csv")

df = pd.read_csv(DATA_PATH)

#convert from string to dt
df["start_time_utc"] = pd.to_datetime(df["start_time_utc"])
df["start_time_sweden"] = pd.to_datetime(df["start_time_sweden"])

#then sort in order:
df = df.sort_values("start_time_utc").reset_index(drop=True)

print("Shape:")
print(df.shape)

print("\nDate range:")
print(df["start_time_utc"].min())
print(df["start_time_utc"].max())

print("\nFirst five rows after sorting:")
print(df.head())

print("\nLast five rows after sorting:")
print(df.tail())

print("\nDuplicate UTC timestamps:")
print(df["start_time_utc"].duplicated().sum())

print("\nDuplicate Swedish timestamps:")
print(df["start_time_sweden"].duplicated().sum())

# Calculate the time difference between consecutive observations
time_diff = df["start_time_utc"].diff()

print("\nMost common time differences:")
print(time_diff.value_counts().head())

print("\nGaps larger than one hour:")
print(df.loc[time_diff > pd.Timedelta(hours=1), ["start_time_utc", "price"]].head(20))

# Build the complete hourly UTC timeline that should exist
expected_timestamps = pd.date_range(
    start=df["start_time_utc"].min(),
    end=df["start_time_utc"].max(),
    freq="h",
)

actual_timestamps = pd.DatetimeIndex(df["start_time_utc"])

missing_timestamps = expected_timestamps.difference(actual_timestamps)

print("\nExpected number of hourly observations:")
print(len(expected_timestamps))

print("\nActual number of observations:")
print(len(df))

print("\nNumber of missing hourly timestamps:")
print(len(missing_timestamps))

print("\nFirst 20 missing timestamps:")
print(missing_timestamps[:20])

print("\nAll observed time-gap sizes:")
print(time_diff.value_counts().sort_index())

# Find timestamps that are not exactly on the hour
off_hour_mask = (
    (df["start_time_utc"].dt.minute != 0)
    | (df["start_time_utc"].dt.second != 0)
)

off_hour_rows = df.loc[off_hour_mask]

print("\nObservations not aligned to a full hour:")
print(off_hour_rows)

if not off_hour_rows.empty:
    for idx in off_hour_rows.index:
        start = max(0, idx - 3)
        end = min(len(df), idx + 4)

        print("\nRows surrounding off-hour observation:")
        print(
            df.loc[
                start:end,
                ["start_time_utc", "start_time_sweden", "price"],
            ]
        )