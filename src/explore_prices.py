from pathlib import Path

import pandas as pd

df = pd.read_csv("data/processed/day_ahead_prices_clean.csv")

print("First five rows:")
print(df.head())

print("\nPrice statistics:")
print(df["price"].describe())

#plot clean data
import matplotlib.pyplot as plt

df["start_time_utc"] = pd.to_datetime(df["start_time_utc"])

""" plt.figure(figsize=(12, 5))
plt.plot(df["start_time_utc"], df["price"])
plt.xlabel("time")
plt.ylabel("price")
plt.title("clean data")
plt.tight_layout()
plt.show() """

#plot day mean
df = df.set_index("start_time_utc")
daily_price = df["price"].resample("D").mean()

""" plt.figure(figsize=(12, 5))
plt.plot(daily_price.index, daily_price)
plt.xlabel("time")
plt.ylabel("price")
plt.title("clean data")
plt.tight_layout()
plt.show() """

print("\nDaily price statistics:")
print(daily_price.describe())
