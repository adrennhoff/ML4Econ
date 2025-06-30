#==============================================================================
# Supplemental plots for XGBoost model analysis
#==============================================================================
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import shap

# Load data
df = pd.read_csv("ML4Econ_data.csv", index_col=0)
X = df.drop(columns=["y"])
y = df["y"]

# Load best model 
best_model = xgb.XGBRegressor(objective="reg:squarederror")
best_model.load_model("xgb_model.json")

#==============================================================================
# SHAP Beeswarm Plot
#==============================================================================
# Create SHAP explainer and calculate SHAP values
explainer = shap.Explainer(best_model, X)
shap_values = explainer(X)

# Create beeswarm plot
plt.figure(figsize=(10, 8))
shap.plots.beeswarm(shap_values, max_display=20, show=False)
plt.savefig('../Plots/shap_beeswarm.png', dpi=300)
plt.show()
plt.close()

#==============================================================================
# SHAP Waterfall Plots (side-by-side for observations 1 and 2)
#==============================================================================

# Create plots separately then display side by side using subplots
fig = plt.figure(figsize=(16, 8))
gs = gridspec.GridSpec(1, 2, figure=fig)

# Plot observation 1
plt.subplot(gs[0, 0])
shap.plots.waterfall(shap_values[0], max_display=15, show=False)
plt.title("Observation 1", fontsize=14, pad=20)

# Plot observation 2  
plt.subplot(gs[0, 1])
shap.plots.waterfall(shap_values[1], max_display=15, show=False)
plt.title("Observation 2", fontsize=14, pad=20)

plt.tight_layout()
#plt.savefig('../Plots/shap_waterfall_sidebyside.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

#==============================================================================
# SHAP Waterfall Plots for Observations 1 and 2
#==============================================================================
# Observation 1
shap.plots.waterfall(shap_values[0], max_display=10, show=False)
plt.title("Observation 1", fontsize=14)
plt.show()
plt.close()

# Observation 2
shap.plots.waterfall(shap_values[1], max_display=10, show=False)
plt.title("Observation 2", fontsize=14)
plt.show()
plt.close()

#==============================================================================
# SHAP Force Plots for Observations 1 and 2
# Note: appearance is best is exported as HTML file
#==============================================================================
# Observation 1
shap.save_html("../Plots/force_plot_obs1.html", shap.plots.force(shap_values[0]))

# Observation 2
shap.save_html("../Plots/force_plot_obs2.html", shap.plots.force(shap_values[1]))
