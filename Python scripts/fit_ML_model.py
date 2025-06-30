#==============================================================================
# fit_ML_model.py
# Author: Adam D. Rennhoff
# Purpose: Load simulated data, tune XGBoost and RF models, and save best model
# Input: ML4Econ_data.csv
# Output: xgb_model.json
#==============================================================================

# ------ Load necessary libraries ------
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# ------ Load data ------
df = pd.read_csv("ML4Econ_data.csv", index_col=0)
X = df.drop(columns=["y"])
y = df["y"]

# ------ Split data ------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#==============================================================================
# XGBoost
#==============================================================================

# ------ Define XGBoost model ------
xgb_model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)

# ------ Set hyperparameter grid ------
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.8, 0.9, 1.0]
}

# ------ Grid search ------
grid = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring="neg_mean_squared_error",
    cv=5,
    verbose=1,
    n_jobs=-1)

grid.fit(X_train, y_train)

# ------ Print best hyperparameters ------
print("Best hyperparameters found:")
print(grid.best_params_)

# best_params_ = {
#     'n_estimators': 300,
#     'max_depth': 3,
#     'learning_rate': 0.1,
#     'subsample': 0.8
# }

# ------ Evaluate model performance ------
best_model_cv = grid.best_estimator_
y_pred = best_model_cv.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"XGBoost Test MSE: {mse:.4f}")

# ------ Refit final model on full dataset ------
final_model = xgb.XGBRegressor(**grid.best_params_)
final_model.fit(X, y)

# ------ Save model as JSON ------
final_model.get_booster().save_model("xgb_model.json")

# ------ Comparable linear regression performance ------
# Fit OLS
ols_model = LinearRegression()
ols_model.fit(X_train, y_train)

# Predict on test set
y_pred_ols = ols_model.predict(X_test)

# Compute test MSE
mse_ols = mean_squared_error(y_test, y_pred_ols)
print(f"Linear Regression Test MSE: {mse_ols:.4f}")

#==============================================================================
# Random Forest (higher MSE)
#==============================================================================

# ------ Define Random Forest model ------
rf_model = RandomForestRegressor(random_state=42)

# ------ Set hyperparameter grid ------
rf_param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 5, 10],
    "max_features": ["sqrt", "log2", None]
}

# ------ Grid search ------
rf_grid = GridSearchCV(
    estimator=rf_model,
    param_grid=rf_param_grid,
    scoring="neg_mean_squared_error",
    cv=5,
    verbose=1,
    n_jobs=-1
)

rf_grid.fit(X_train, y_train)

# ------ Evaluate model performance ------
print("Best RF hyperparameters found:")
print(rf_grid.best_params_)

best_rf_model = rf_grid.best_estimator_
rf_y_pred = best_rf_model.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_y_pred)
print(f"Random Forest Test MSE: {rf_mse:.4f}")
