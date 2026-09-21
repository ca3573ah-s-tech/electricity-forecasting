import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


df = pd.read_csv("data/processed/day_ahead_prices_clean.csv")

##Baseline p_t = p_(t-24)
##ML mot din baseline på MAE ≈ 23.20 och RMSE ≈ 36.19.

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
        "hour_of_day",
        "day_of_week",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
    ]
].dropna()

""" print(evaluation_df.head())
print(evaluation_df.tail())
print(evaluation_df.shape) """


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

""" print("Train:", train_df.shape)
print("Validation:", validation_df.shape)
print("Test:", test_df.shape) """

features = [
    "forecast_24",
    "forecast_48",
    "forecast_168",
]

X_train = train_df[features]
y_train = train_df["price"]

X_validation = validation_df[features]
y_validation = validation_df["price"]

X_test = test_df[features]
y_test = test_df["price"]

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

model = LinearRegression()
model.fit(X_train, y_train)
""" y_pred = model.predict(X_test)
errors = y_test - y_pred

##MAE

print("MAE: ",np.abs(errors).mean())

##RMSE

e = errors**2
mean_e = e.mean()
print("RMSE: ", np.sqrt(mean_e))

##base: MAE:  22.432241604100795
##      RMSE:  32.68066546991834

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_) """

""" validation_pred = model.predict(X_validation)

linear_errors = y_validation - validation_pred

linear_mae = np.abs(linear_errors).mean()
linear_rmse = np.sqrt((linear_errors ** 2).mean())


#baseline

baseline_pred = validation_df["forecast_24"]

baseline_errors = y_validation - baseline_pred

baseline_mae = np.abs(baseline_errors).mean()
baseline_rmse = np.sqrt((baseline_errors ** 2).mean()) """


#compare

""" print("\nValidation results for 2024")

print("\n24h baseline:")
print("MAE:", baseline_mae)
print("RMSE:", baseline_rmse)

print("\nLinear regression:")
print("MAE:", linear_mae)
print("RMSE:", linear_rmse) """


##feature engineering, 
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

model = LinearRegression()
model.fit(X_train, y_train)

validation_pred = model.predict(X_validation)

""" linear_errors = y_validation - validation_pred

linear_mae = np.abs(linear_errors).mean()
linear_rmse = np.sqrt((linear_errors ** 2).mean())
print("now with cyclic functions")
print("\nLinear regression:")
print("MAE:", linear_mae)
print("RMSE:", linear_rmse)
 """
##now with cyclic functions
#Linear regression:
#MAE: 18.9676030342152
#RMSE: 30.992052475629702
#slightly better than without cyclic features, still worse MAE than the baseline

validation_results = validation_df.copy()

validation_results["prediction"] = validation_pred
validation_results["error"] = (
    validation_results["price"] - validation_results["prediction"]
)

validation_results["absolute_error"] = np.abs(
    validation_results["error"]
)

largest_errors = validation_results.nlargest(
    10,
    "absolute_error"
)

""" print(
    largest_errors[
        [
            "start_time_utc",
            "price",
            "prediction",
            "absolute_error",
        ]
    ]
) """

##problems with quick fluctiations, 
# probably since model does not include any information about weather etc

##Now non-linear model:

from sklearn.ensemble import GradientBoostingRegressor
gb_model = GradientBoostingRegressor(
    random_state=42
)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_validation)

gb_errors = gb_pred - y_validation
nonlinear_mae = np.abs(gb_errors).mean()
nonlinear_rmse = np.sqrt((gb_errors ** 2).mean())
print("\nnonlinear regression:")
print("MAE:", nonlinear_mae)
print("RMSE:", nonlinear_rmse)

##improves MAE 

train_pred = gb_model.predict(X_train)

train_errors = y_train - train_pred

train_mae = np.abs(train_errors).mean()
train_rmse = np.sqrt((train_errors ** 2).mean())

print("\nGradient Boosting training results:")
print("MAE:", train_mae)
print("RMSE:", train_rmse)

print("\nGradient Boosting validation results:")
print("MAE:", nonlinear_mae)
print("RMSE:", nonlinear_rmse)

""" Gradient Boosting training results:
MAE: 25.466086184546178
RMSE: 42.77720403139915

Gradient Boosting validation results:
MAE: 18.14164918838669
RMSE: 30.322094365275273 
Probably these results since training includes 2022 where the market hade more extremes"""