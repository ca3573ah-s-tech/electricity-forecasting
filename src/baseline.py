import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv("data/processed/day_ahead_prices_clean.csv")

##Baseline p_t = p_(t-24)

df["start_time_utc"] = pd.to_datetime(df["start_time_utc"])
df = df.set_index("start_time_utc")

df["forecast"] = df["price"].shift(24)

print(df.loc["2022-01-02 13:00:00", ["price", "forecast"]])
print(df.loc["2022-01-01 13:00:00", "price"])



#######Test period 2025
#MAE
evaluation_df = df[["price", "forecast"]].dropna()

test_df = evaluation_df.loc["2025-01-01":].copy()
test_df["error"] = test_df["price"] - test_df["forecast"]
test_df["error_abs"] = np.abs(test_df["error"] )

mae = test_df["error_abs"].mean()

#RMSE

test_df["error_sq"] = test_df["error"]**2
rmse2 = test_df["error_sq"].mean()
rmse = np.sqrt(rmse2)

print("Test MAE:", mae)
print("Test RMSE:", rmse)




