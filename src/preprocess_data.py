from pathlib import Path

import pandas as pd


RAW_DATA_PATH = Path("data/raw/day_ahead_prices.csv")
PROCESSED_DATA_PATH = Path("data/processed/day_ahead_prices_clean.csv")


def load_raw_data():
    df = pd.read_csv(RAW_DATA_PATH)

    df["start_time_utc"] = pd.to_datetime(df["start_time_utc"])
    df["start_time_sweden"] = pd.to_datetime(df["start_time_sweden"])

    return df


def clean_data(df):
    # Sort observations chronologically
    df = df.sort_values("start_time_utc").reset_index(drop=True)

    # Keep only observations aligned to a full UTC hour
    

    df = df.loc[(df["start_time_utc"].dt.minute == 0)
            & (df["start_time_utc"].dt.second == 0)].copy()

    # Create a complete hourly UTC timeline
    full_timeline = pd.date_range(
        start=df["start_time_utc"].min(),
        end=df["start_time_utc"].max(),
        freq="h",
    )

    # Reindex onto the complete timeline
    df = (
        df.set_index("start_time_utc")
        .reindex(full_timeline)
        .rename_axis("start_time_utc")
        .reset_index()
    )

    return df


def main():
    df = load_raw_data()
    clean_df = clean_data(df)

    assert clean_df["start_time_utc"].is_unique

    time_diffs = clean_df["start_time_utc"].diff().dropna()

    assert (time_diffs == pd.Timedelta(hours=1)).all()
    assert (clean_df["start_time_utc"].dt.minute == 0).all()
    assert (clean_df["start_time_utc"].dt.second == 0).all()

    clean_df.to_csv(PROCESSED_DATA_PATH, index=False)

    print("\nShape:")
    print(clean_df.shape)

    print("\nMissing prices:")
    print(clean_df["price"].isna().sum())

    print("\nDate range:")
    print(clean_df["start_time_utc"].min())
    print(clean_df["start_time_utc"].max())


if __name__ == "__main__":
    main()