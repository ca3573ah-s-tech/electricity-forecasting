import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

def prepare_data():
    df = pd.read_csv("data/processed/day_ahead_prices_clean.csv")

    df["start_time_utc"] = pd.to_datetime(df["start_time_utc"])

    df["forecast_24"] = df["price"].shift(24)
    df["forecast_48"] = df["price"].shift(48)
    df["forecast_168"] = df["price"].shift(168)

    df["hour_of_day"] = df["start_time_utc"].dt.hour
    df["day_of_week"] = df["start_time_utc"].dt.dayofweek

    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)

    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    evaluation_df = df[
        [
            "start_time_utc",
            "price",
            "forecast_24",
            "forecast_48",
            "forecast_168",
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
        ]
    ].dropna()

    train_df = evaluation_df[
        evaluation_df["start_time_utc"] < "2024-01-01"
    ]

    validation_df = evaluation_df[
        (evaluation_df["start_time_utc"] >= "2024-01-01")
        & (evaluation_df["start_time_utc"] < "2025-01-01")
    ]

    test_df = evaluation_df[
        evaluation_df["start_time_utc"] >= "2025-01-01"
    ]

    return train_df, validation_df, test_df

train_df, validation_df, test_df = prepare_data()

features = [
    "forecast_24",
    "forecast_48",
    "forecast_168",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]

X_train = train_df[features]
y_train = train_df["price"]

X_validation = validation_df[features]
y_validation = validation_df["price"]


def evaluate_predictions(y_true, y_pred):
    errors = y_true - y_pred

    mae = np.abs(errors).mean()
    rmse = np.sqrt((errors ** 2).mean())

    return mae, rmse


linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

linear_pred = linear_model.predict(X_validation)

linear_mae, linear_rmse = evaluate_predictions(
    y_validation,
    linear_pred
)

""" print("\nLinear regression:")
print("MAE:", linear_mae)
print("RMSE:", linear_rmse) """

gb_model = GradientBoostingRegressor(
    random_state=42
)

gb_model.fit(X_train, y_train)

gb_pred = gb_model.predict(X_validation)

gb_mae, gb_rmse = evaluate_predictions(
    y_validation,
    gb_pred
)

""" print("\nGradient Boosting:")
print("MAE:", gb_mae)
print("RMSE:", gb_rmse) """

##try on test set now!

train_validation_df = pd.concat(
    [train_df, validation_df]
)

X_train_final = train_validation_df[features]
y_train_final = train_validation_df["price"]

X_test = test_df[features]
y_test = test_df["price"]

final_model = GradientBoostingRegressor(
    random_state=42
)

final_model.fit(X_train_final, y_train_final)

test_pred = final_model.predict(X_test)

test_mae, test_rmse = evaluate_predictions(
    y_test,
    test_pred
)

print("\nFinal test results:")
print("MAE:", test_mae)
print("RMSE:", test_rmse)

##Final test results:
#MAE: 21.320210848158293
#RMSE: 31.36645546656435

#visuals

test_results = test_df[
    ["start_time_utc", "price"]
].copy()

test_results["prediction"] = test_pred

import matplotlib.pyplot as plt

plot_data = test_results[
    (test_results["start_time_utc"] >= "2025-01-01")
    & (test_results["start_time_utc"] < "2025-01-15")
]

plt.figure(figsize=(12, 5))

plt.plot(
    plot_data["start_time_utc"],
    plot_data["price"],
    label="Actual"
)

plt.plot(
    plot_data["start_time_utc"],
    plot_data["prediction"],
    label="Forecast"
)

plt.xlabel("Date")
plt.ylabel("Price (EUR/MWh)")
plt.title("SE3 Electricity Price Forecast - January 2025")
plt.legend()
plt.tight_layout()

plt.tight_layout()
plt.savefig(
    "results/forecast_january_2025.png",
    dpi=150
)
plt.show()

plt.figure(figsize=(6, 6))

plt.scatter(
    test_results["price"],
    test_results["prediction"],
    alpha=0.3
)

min_price = min(
    test_results["price"].min(),
    test_results["prediction"].min()
)

max_price = max(
    test_results["price"].max(),
    test_results["prediction"].max()
)

plt.plot(
    [min_price, max_price],
    [min_price, max_price],
    linestyle="--"
)

plt.xlabel("Actual price (EUR/MWh)")
plt.ylabel("Predicted price (EUR/MWh)")
plt.title("Actual vs Predicted Prices - 2025 Test Set")

plt.tight_layout()
plt.savefig(
    "results/actual_vs_predicted.png",
    dpi=150
)

plt.show()

