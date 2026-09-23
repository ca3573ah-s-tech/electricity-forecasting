import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import stochastic_model

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression


def prepare_data():
    df = pd.read_csv(
        "data/processed/day_ahead_prices_clean.csv"
    )

    df["start_time_utc"] = pd.to_datetime(
        df["start_time_utc"]
    )

    # Historical price features
    df["forecast_24"] = df["price"].shift(24)
    df["forecast_48"] = df["price"].shift(48)
    df["forecast_168"] = df["price"].shift(168)

    # Calendar features
    df["hour_of_day"] = df["start_time_utc"].dt.hour
    df["day_of_week"] = df["start_time_utc"].dt.dayofweek

    df["hour_sin"] = np.sin(
        2 * np.pi * df["hour_of_day"] / 24
    )
    df["hour_cos"] = np.cos(
        2 * np.pi * df["hour_of_day"] / 24
    )

    df["dow_sin"] = np.sin(
        2 * np.pi * df["day_of_week"] / 7
    )
    df["dow_cos"] = np.cos(
        2 * np.pi * df["day_of_week"] / 7
    )

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


def evaluate_predictions(y_true, y_pred):
    errors = y_true - y_pred

    mae = np.abs(errors).mean()
    rmse = np.sqrt((errors ** 2).mean())

    return mae, rmse


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


# Model comparison on validation data

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

linear_pred = linear_model.predict(X_validation)

linear_mae, linear_rmse = evaluate_predictions(
    y_validation,
    linear_pred,
)

gb_model = GradientBoostingRegressor(
    random_state=42
)
gb_model.fit(X_train, y_train)

gb_pred = gb_model.predict(X_validation)

gb_mae, gb_rmse = evaluate_predictions(
    y_validation,
    gb_pred,
)

print("\nValidation results:")
print(
    f"Linear Regression - "
    f"MAE: {linear_mae:.2f}, RMSE: {linear_rmse:.2f}"
)
print(
    f"Gradient Boosting - "
    f"MAE: {gb_mae:.2f}, RMSE: {gb_rmse:.2f}"
)


# Fit OU model to validation residuals

validation_residuals = y_validation - gb_pred
residuals = np.asarray(validation_residuals)

kappa, mu, sigma = (
    stochastic_model.estimate_ou_parameters(
        residuals,
        dt=1.0,
    )
)

half_life = np.log(2) / kappa

print("\nOU residual model:")
print(f"kappa: {kappa:.4f}")
print(f"mu: {mu:.4f}")
print(f"sigma: {sigma:.4f}")
print(f"half-life: {half_life:.2f} hours")


# Final Gradient Boosting model

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

final_model.fit(
    X_train_final,
    y_train_final,
)

test_pred = final_model.predict(X_test)

test_mae, test_rmse = evaluate_predictions(
    y_test,
    test_pred,
)

print("\nFinal test results:")
print(f"MAE: {test_mae:.2f}")
print(f"RMSE: {test_rmse:.2f}")

np.random.seed(42)

# Find valid, non-overlapping 24-hour test windows

block_size = 24

test_times = (
    test_df["start_time_utc"]
    .reset_index(drop=True)
)

valid_blocks = []

start = 0

while start + block_size <= len(test_times):
    end = start + block_size

    block_times = test_times.iloc[start:end]

    block_diffs = block_times.diff().dropna()

    block_is_contiguous = (
        block_diffs == pd.Timedelta(hours=1)
    ).all()

    if start == 0:
        previous_time = validation_df[
            "start_time_utc"
        ].iloc[-1]
    else:
        previous_time = test_times.iloc[start - 1]

    has_previous_hour = (
        block_times.iloc[0] - previous_time
        == pd.Timedelta(hours=1)
    )

    if block_is_contiguous and has_previous_hour:
        valid_blocks.append((start, end))
        start += block_size
    else:
        start += 1


print("\nValid rolling windows:")
print("Valid 24-hour blocks:", len(valid_blocks))
print("Hours covered:", len(valid_blocks) * 24)


# Rolling OU evaluation

n_paths = 1000
n_test = len(test_pred)

ou_median = np.full(n_test, np.nan)
ou_lower = np.full(n_test, np.nan)
ou_upper = np.full(n_test, np.nan)

for start, end in valid_blocks:
    horizon = end - start

    if start == 0:
        x0 = validation_residuals.iloc[-1]
    else:
        x0 = y_test.iloc[start - 1] - test_pred[start - 1]

    residual_paths = stochastic_model.simulate_ou_paths(
        x0=x0,
        mu=mu,
        kappa=kappa,
        sigma=sigma,
        n_steps=horizon + 1,
        n_paths=n_paths,
    )[1:, :]

    ml_forecast = test_pred[start:end].reshape(-1, 1)

    price_paths = ml_forecast + residual_paths

    ou_lower[start:end] = np.percentile(
        price_paths,
        10,
        axis=1,
    )

    ou_median[start:end] = np.percentile(
        price_paths,
        50,
        axis=1,
    )

    ou_upper[start:end] = np.percentile(
        price_paths,
        90,
        axis=1,
    )

valid_mask = ~np.isnan(ou_median)

actual_valid = y_test.to_numpy()[valid_mask]
gb_valid = test_pred[valid_mask]

ou_median_valid = ou_median[valid_mask]
ou_lower_valid = ou_lower[valid_mask]
ou_upper_valid = ou_upper[valid_mask]

gb_mae_rolling, gb_rmse_rolling = evaluate_predictions(
    actual_valid,
    gb_valid,
)

ou_mae_rolling, ou_rmse_rolling = evaluate_predictions(
    actual_valid,
    ou_median_valid,
)

inside_interval = (
    (actual_valid >= ou_lower_valid)
    & (actual_valid <= ou_upper_valid)
)

coverage = inside_interval.mean()

average_width = (
    ou_upper_valid - ou_lower_valid
).mean()

below_interval = (
    actual_valid < ou_lower_valid
).mean()

above_interval = (
    actual_valid > ou_upper_valid
).mean()

print("\nRolling 24-hour evaluation - valid 2025 windows")
print("Hours evaluated:", valid_mask.sum())

print("\nGradient Boosting:")
print(f"MAE: {gb_mae_rolling:.2f}")
print(f"RMSE: {gb_rmse_rolling:.2f}")

print("\nGradient Boosting + OU median:")
print(f"MAE: {ou_mae_rolling:.2f}")
print(f"RMSE: {ou_rmse_rolling:.2f}")

print("\nOU 80% prediction interval:")
print(f"Coverage: {coverage:.3f}")
print(f"Average width: {average_width:.2f} EUR/MWh")
print(f"Below interval: {below_interval:.3f}")
print(f"Above interval: {above_interval:.3f}")

# Monte Carlo uncertainty using OU residual model

np.random.seed(42)

horizon = 48
n_paths = 1000

x0 = validation_residuals.iloc[-1]

residual_paths = stochastic_model.simulate_ou_paths(
    x0=x0,
    mu=mu,
    kappa=kappa,
    sigma=sigma,
    n_steps=horizon + 1,
    n_paths=n_paths,
)[1:, :]

ml_forecast = test_pred[:horizon].reshape(-1, 1)

price_paths = ml_forecast + residual_paths

lower = np.percentile(
    price_paths,
    10,
    axis=1,
)

median = np.percentile(
    price_paths,
    50,
    axis=1,
)

upper = np.percentile(
    price_paths,
    90,
    axis=1,
)


# Plot stochastic forecast interval

forecast_times = test_df[
    "start_time_utc"
].iloc[:horizon]

actual = y_test.iloc[:horizon].to_numpy()
gb_forecast = test_pred[:horizon]

plt.figure(figsize=(12, 6))

plt.plot(
    forecast_times,
    actual,
    label="Actual",
    linewidth=2,
)

plt.plot(
    forecast_times,
    gb_forecast,
    label="Gradient Boosting",
    linestyle="--",
)

plt.plot(
    forecast_times,
    median,
    label="GB + OU median",
)

plt.fill_between(
    forecast_times,
    lower,
    upper,
    alpha=0.2,
    label="80% Monte Carlo interval",
)

plt.xlabel("Time")
plt.ylabel("Price (EUR/MWh)")
plt.title(
    "48-hour Test Window with OU Uncertainty"
)
plt.legend()
plt.tight_layout()

plt.savefig(
    "results/stochastic_forecast_48h.png",
    dpi=150,
)

plt.show()