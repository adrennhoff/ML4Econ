import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.inspection import permutation_importance
import shap

#==============================================================================
# Standard feature importance based on Gain
#==============================================================================
# Load trained model
booster = xgb.Booster()
booster.load_model("xgb_model.json")

# Get feature importance (by gain)
importance = booster.get_score(importance_type='gain')

# Convert to DataFrame and sort
importance_df = pd.DataFrame({
    'Feature': list(importance.keys()),
    'Importance': list(importance.values())
}).sort_values(by='Importance', ascending=False)

# Plot
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel("Gain")
#plt.title("XGBoost Feature Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

plt.savefig('../Plots/gain_feature_imp.png', dpi=300)
plt.close()

#==============================================================================
# Permutation importance (no confidence intervals)
#==============================================================================
# 1. Load data
df = pd.read_csv("ML4Econ_data.csv", index_col=0)
X = df.drop(columns=["y"])
y = df["y"]

# 2. Load best model 
best_model = xgb.XGBRegressor()
best_model.load_model("xgb_model.json")

# Refit scikit-learn wrapper around loaded booster (needed for sklearn API to work)
best_model = xgb.XGBRegressor(objective="reg:squarederror")
best_model.load_model("xgb_model.json")

# Compute permutation importance
result = permutation_importance(best_model, X, y, n_repeats=30, random_state=42, scoring='neg_mean_squared_error')

# Organize into DataFrame
perm_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': result.importances_mean
}).sort_values(by='Importance', ascending=False)

# Plot
plt.figure(figsize=(10, 6))
plt.barh(perm_df['Feature'], perm_df['Importance'])
plt.xlabel(r"Decrease in Model Performance ($\Delta$MSE)")
#plt.title("XGBoost Permutation Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

plt.savefig('../Plots/permutation_imp.png', dpi=300)
plt.close()

#==============================================================================
# Permutation importance confidence intervals
#==============================================================================

# Run permutation importance again if needed
result = permutation_importance(best_model, X, y, n_repeats=30, random_state=42, scoring='neg_mean_squared_error')

# Get statistics
features = X.columns
means = result.importances_mean
lower_bounds = np.percentile(result.importances, 2.5, axis=1)
upper_bounds = np.percentile(result.importances, 97.5, axis=1)

# Assemble DataFrame
perm_ci_df = pd.DataFrame({
    'Feature': features,
    'Mean Importance': means,
    '2.5%': lower_bounds,
    '97.5%': upper_bounds
}).sort_values(by='Mean Importance', ascending=False)

# Display nicely
pd.set_option("display.float_format", "{:.4f}".format)
display(perm_ci_df)

# Plot instead of table

plt.figure(figsize=(10, 6))
plt.errorbar(
    x=perm_ci_df['Mean Importance'],
    y=perm_ci_df['Feature'],
    xerr=[
        perm_ci_df['Mean Importance'] - perm_ci_df['2.5%'],
        perm_ci_df['97.5%'] - perm_ci_df['Mean Importance']
    ],
    fmt='o',
    capsize=4,
    color='steelblue'
)
plt.gca().invert_yaxis() 
plt.xlabel(r"Decrease in Model Performance ($\Delta$MSE)")
#plt.title("Permutation Importance with 95% Confidence Intervals")
plt.tight_layout()
plt.show()

plt.savefig('../Plots/permutation_CI.png', dpi=300)
plt.close()

#==============================================================================
# SHAP Importance (mean absolute SHAP value)
#==============================================================================
# Explain the model with SHAP (uses TreeExplainer for XGBoost)
explainer = shap.Explainer(best_model, X)
shap_values = explainer(X)

# Calculate mean absolute SHAP values
shap_importance = np.abs(shap_values.values).mean(axis=0)
shap_df = pd.DataFrame({
    'Feature': X.columns,
    'Mean |SHAP|': shap_importance
}).sort_values(by='Mean |SHAP|', ascending=False)

# Plot
plt.figure(figsize=(10, 6))
plt.barh(shap_df['Feature'], shap_df['Mean |SHAP|'])
plt.xlabel("Mean |SHAP value|")
#plt.title("SHAP-Based Feature Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

plt.savefig('../Plots/SHAP_imp.png', dpi=300)
plt.close()