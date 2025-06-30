# ------ Load necessary libraries ------
import pandas as pd
import numpy as np
import shap
import xgboost as xgb
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PyALE import ale
from sklearn.utils import resample

# ------ Load and define data ------
df = pd.read_csv("ML4Econ_data.csv", index_col=0)
X = df.drop(columns=["y"])
y = df["y"]

# ------ Load XGBoost model ------
# Create model instance with objective
best_model = xgb.XGBRegressor(objective="reg:squarederror")
# Load trained model from .json file
best_model.load_model("xgb_model.json")


#==============================================================================
# PDP plot for x2
#==============================================================================
# x2 which is convex
PartialDependenceDisplay.from_estimator(
    best_model,
    X,
    features=['x2'],
    kind='average',
    grid_resolution=200,
    n_jobs=-1
)

plt.tight_layout()
plt.show()

plt.savefig('../Plots/PDP_x2.png', dpi=300)
plt.close()

#==============================================================================
# PDP plot for x3
#==============================================================================
# x3 which is convex (decreasing) with diminishing marginal returns
PartialDependenceDisplay.from_estimator(
    best_model,
    X,
    features=['x3'],
    kind='average',
    grid_resolution=250,
    n_jobs=-1
)

plt.tight_layout()
plt.show()

plt.savefig('../Plots/PDP_x3.png', dpi=300)
plt.close()

#==============================================================================
# PDP plot for x7
#==============================================================================
# x7 which plateaus for x7 > 1
PartialDependenceDisplay.from_estimator(
    best_model,
    X,
    features=['x7'],
    kind='average',
    grid_resolution=250,
    n_jobs=-1
)

plt.tight_layout()
plt.show()

plt.savefig('../Plots/PDP_x7.png', dpi=300)
plt.close()

#==============================================================================
# PDP plot for x8
#==============================================================================
# x8 which is upward sloping linear
PartialDependenceDisplay.from_estimator(
    best_model,
    X,
    features=['x8'],
    kind='average',
    grid_resolution=250,
    n_jobs=-1
)

plt.tight_layout()
plt.show()

plt.savefig('../Plots/PDP_x8.png', dpi=300)
plt.close()

#==============================================================================
# Combined PDP plot
#==============================================================================
fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)

# File paths
image_files = ['../Plots/PDP_x2.png', '../Plots/PDP_x3.png', '../Plots/PDP_x7.png', '../Plots/PDP_x8.png']

# Titles (optional)
titles = ["(a) x2", "(b) x3", "(c) x7", "(d) x8"]

for ax, img_path, title in zip(axes.flat, image_files, titles):
    img = mpimg.imread(img_path)
    ax.imshow(img)
    ax.axis('off')  # Hide axes
    ax.set_title(title, loc='left')


plt.savefig("../Plots/combined_PDP_plot.png", dpi=300)
plt.close()

#==============================================================================
# ICE plot for x2
#==============================================================================
PartialDependenceDisplay.from_estimator(
    best_model,
    X,
    features=['x2'],
    kind='individual', 
    grid_resolution=250,
    n_jobs=-1
)

plt.tight_layout()
plt.show()

plt.savefig('../Plots/ICE_x2.png', dpi=300)
plt.close()

#==============================================================================
# ICE plot for x5
#==============================================================================
PartialDependenceDisplay.from_estimator(
    best_model,
    X,
    features=['x5'],
    kind='individual', 
    grid_resolution=250,
    n_jobs=-1
)

plt.tight_layout()
plt.show()

plt.savefig('../Plots/ICE_x5.png', dpi=300)
plt.close()

#==============================================================================
# ALE plot for x2
#==============================================================================

# ALE for x2 (centered)
ale_result = ale(
    X=X,
    model=best_model,
    feature=["x2"],
    include_CI=True,    # confidence intervals off for now
    grid_size=50        # number of bins
)


#==============================================================================
# Bootstrapped Average SHAP values in bins
#==============================================================================

best_params = {
    'n_estimators': 300,
    'max_depth': 3,
    'learning_rate': 0.1,
    'subsample': 0.8
}

def bootstrap_shap_bins(X, y, feature_name, params, n_boot=100, n_bins=50):
    bin_edges = np.linspace(X[feature_name].min(), X[feature_name].max(), n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    result_df = pd.DataFrame(index=bin_centers)

    for i in range(n_boot):
        X_resampled, y_resampled = resample(X, y, replace=True, random_state=42 + i)
        model = xgb.XGBRegressor(**params)
        model.fit(X_resampled, y_resampled)

        explainer = shap.Explainer(model)
        shap_values = explainer(X)
        shap_feature_vals = shap_values[:, feature_name].values

        binned = pd.cut(X[feature_name], bins=bin_edges, include_lowest=True)
        grouped = pd.DataFrame({'bin': binned, 'shap': shap_feature_vals})
        bin_means = grouped.groupby('bin', observed=False).mean(numeric_only=True).reindex(binned.cat.categories)
        result_df[f'rep_{i}'] = bin_means['shap'].values

    return result_df


def compute_shap_bin_summary(shap_bin_df):
    means = shap_bin_df.mean(axis=1)
    lower = shap_bin_df.quantile(0.025, axis=1)
    upper = shap_bin_df.quantile(0.975, axis=1)

    summary_df = pd.DataFrame({
        'Bin_Center': shap_bin_df.index,
        'SHAP_Mean': means,
        'Lower_CI': lower,
        'Upper_CI': upper
    })
    return summary_df


def plot_shap_bin_averages(summary_df, feature_name):
    """
    Plot SHAP values averaged within bins of a feature, with bootstrapped 95% CIs.
    Returns the figure object so it can be saved externally.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(summary_df['Bin_Center'], summary_df['SHAP_Mean'], label='SHAP (bin average)')
    ax.fill_between(summary_df['Bin_Center'], summary_df['Lower_CI'], summary_df['Upper_CI'], 
                    alpha=0.3, label='95% CI')
    ax.axhline(0, color='gray', linestyle='--')
    ax.set_xlabel(feature_name)
    ax.set_ylabel("Average SHAP Value (within bin)")
    ax.legend()
    fig.tight_layout()
    return fig  # no plt.show()

# Example function calls: x2
# Step 1: Generate bootstrapped SHAP bin values
shap_bin_df = bootstrap_shap_bins(X, y, feature_name="x2", params=best_params, n_boot=100)
shap_bin_df_clean = shap_bin_df.dropna(how='all')

# Step 2: Summarize across replications
summary_df = compute_shap_bin_summary(shap_bin_df_clean)

# Step 3: Plot (without title)
fig = plot_shap_bin_averages(summary_df, "x2")
fig.savefig('../Plots/SHAP_bins_x2.png', dpi=300)
fig.show()

plt.close(fig)

# Function calls: x8
# Step 1: Generate bootstrapped SHAP bin values
shap_bin_df = bootstrap_shap_bins(X, y, feature_name="x8", params=best_params, n_boot=100)
shap_bin_df_clean = shap_bin_df.dropna(how='all')

# Step 2: Summarize across replications
summary_df = compute_shap_bin_summary(shap_bin_df_clean)

# Step 3: Plot (without title)
fig = plot_shap_bin_averages(summary_df, "x8")
fig.savefig('../Plots/SHAP_bins_x8.png', dpi=300)
fig.show()

plt.close(fig)

# Function calls: x15
# Step 1: Generate bootstrapped SHAP bin values
shap_bin_df = bootstrap_shap_bins(X, y, feature_name="x15", params=best_params, n_boot=100)
shap_bin_df_clean = shap_bin_df.dropna(how='all')

# Step 2: Summarize across replications
summary_df = compute_shap_bin_summary(shap_bin_df_clean)

# Step 3: Plot (without title)
fig = plot_shap_bin_averages(summary_df, "x15")
fig.savefig('../Plots/SHAP_bins_x15.png', dpi=300)
fig.show()

plt.close(fig)