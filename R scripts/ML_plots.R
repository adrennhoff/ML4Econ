#==============================================================================
# ML_plots.R
# Author: Adam D. Rennhoff
# Purpose: Generate feature importance plots using built-in, permutation, and SHAP methods
# Input: xgb_model.json, ML4Econ_data.csv
# Output: Various plots
#==============================================================================

# ------ Load necessary libraries ------
library(xgboost)
library(ggplot2)
library(dplyr)
library(fastshap)
library(tidyr)
library(pdp)

#==============================================================================
# Load data and model
#==============================================================================

# ------ Load data ------
df <- read.csv("ML4Econ_data.csv", row.names = 1)
X <- df %>% select(-y)
y <- df$y

# ------ Load trained XGBoost model ------
xgb_model <- xgb.load("xgb_model.json")

#==============================================================================
# 1. Built-in Feature Importance (Gain)
#==============================================================================

# ------ Get feature importance ------
importance <- xgb.importance(model = xgb_model)

# ------ Plot built-in importance ------
gain_plot <- ggplot(importance, aes(x = reorder(Feature, Gain), y = Gain)) +
  geom_col(fill = "steelblue", alpha = 0.7) +
  coord_flip() +
  labs(
    x = "Feature",
    y = "Gain",
    title = "XGBoost Built-in Feature Importance"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

print(gain_plot)
#ggsave("gain_feature_imp.png", gain_plot, width = 10, height = 6, dpi = 300)

#==============================================================================
# 2. Permutation Importance
#==============================================================================

# ------ Permutation importance function ------
permutation_importance <- function(model, X, y, n_repeats = 30) {
  # Baseline performance
  baseline_pred <- predict(model, as.matrix(X))
  baseline_mse <- mean((y - baseline_pred)^2)
  
  # Initialize results
  n_features <- ncol(X)
  importance_scores <- matrix(0, nrow = n_repeats, ncol = n_features)
  colnames(importance_scores) <- colnames(X)
  
  # Set seed for reproducibility
  set.seed(42)
  
  # For each feature
  for (i in 1:n_features) {
    feature_name <- colnames(X)[i]
    
    # For each repeat
    for (rep in 1:n_repeats) {
      # Create permuted data
      X_perm <- X
      X_perm[, i] <- sample(X_perm[, i])
      
      # Get predictions with permuted feature
      perm_pred <- predict(model, as.matrix(X_perm))
      perm_mse <- mean((y - perm_pred)^2)
      
      # Store importance (increase in MSE)
      importance_scores[rep, i] <- perm_mse - baseline_mse
    }
  }
  
  # Calculate mean importance
  mean_importance <- colMeans(importance_scores)
  
  return(list(
    importances_mean = mean_importance,
    importances = importance_scores
  ))
}

# ------ Calculate permutation importance ------
cat("Computing permutation importance...\n")
perm_result <- permutation_importance(xgb_model, X, y, n_repeats = 30)

# ------ Create DataFrame ------
perm_df <- data.frame(
  Feature = names(perm_result$importances_mean),
  Importance = perm_result$importances_mean
) %>%
  arrange(desc(Importance))

# ------ Plot permutation importance ------
perm_plot <- ggplot(perm_df, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_col(fill = "darkgreen", alpha = 0.7) +
  coord_flip() +
  labs(
    x = "Feature",
    y = expression(paste("Decrease in Model Performance (", Delta, "MSE)")),
    title = "XGBoost Permutation Importance"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

print(perm_plot)
#ggsave("permutation_imp.png", perm_plot, width = 10, height = 6, dpi = 300)

#==============================================================================
# 3. SHAP Feature Importance
#==============================================================================

# SHAP values
cat("Computing SHAP values...\n")

# Create prediction wrapper function
pfun <- function(object, newdata) {
  predict(object, as.matrix(newdata))
}

# Calculate SHAP values for full dataset (matching Python)
shap_values <- explain(
  object = xgb_model,
  X = as.matrix(X),  # Use full dataset like Python
  pred_wrapper = pfun,
  nsim = 100
)

# Calculate mean absolute SHAP values
shap_importance <- colMeans(abs(shap_values))

# Create DataFrame (matching Python structure)
shap_df <- data.frame(
  Feature = names(shap_importance),
  'Mean |SHAP|' = shap_importance
) %>%
  arrange(desc(Mean..SHAP.))

# Plot
shap_plot <- ggplot(shap_df, aes(x = reorder(Feature, Mean..SHAP.), y = Mean..SHAP.)) +
  geom_col(fill = "darkorange", alpha = 0.7) +
  coord_flip() +
  labs(
    x = "Feature",
    y = "Mean |SHAP value|",
    title = "SHAP-Based Feature Importance"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

print(shap_plot)
#ggsave("SHAP_imp.png", shap_plot, width = 10, height = 6, dpi = 300)

#==============================================================================
# 4. PDP plot for x2
#==============================================================================

# Create prediction function for pdp (exactly two arguments)
pred_fun <- function(object, newdata) {
  predict(object, as.matrix(newdata))
}

# Generate PDP using explicit package reference
pdp_x2 <- pdp::partial(
  object = xgb_model,
  pred.var = "x2",
  train = X,
  pred.fun = pred_fun,
  grid.resolution = 250,
  ice = FALSE,
  plot = FALSE
)

# Since pdp is ignoring ice = FALSE, manually convert ICE to PDP
pdp_x2_final <- pdp_x2 %>%
  group_by(x2) %>%
  summarise(yhat = mean(yhat), .groups = 'drop')

# Plot the actual PDP
pdp_plot_x2 <- ggplot(pdp_x2_final, aes(x = x2, y = yhat)) +
  geom_line(color = "steelblue", linewidth = 1) +
  labs(
    x = "x2",
    y = "Partial dependence",
    title = "Partial Dependence Plot"
  ) +
  theme_minimal()

print(pdp_plot_x2)
#ggsave("pdp_plot_x2.png", pdp_plot_x2, width = 6.4, height = 4.8, dpi = 300)

#==============================================================================
# 5. ICE plot for x2
#==============================================================================

# Plot directly
pdp::partial(
  object = xgb_model,
  pred.var = "x2",
  train = X,
  pred.fun = pred_fun,
  grid.resolution = 250,
  ice = TRUE,
  plot = TRUE
)

#==============================================================================
# 6. Bootstrapped Average SHAP values in bins
#==============================================================================

# Best parameters (matching Python)
best_params <- list(
  max_depth = 3,
  eta = 0.1,
  subsample = 0.8,
  objective = "reg:squarederror"
)

n_rounds <- 300

bootstrap_shap_bins <- function(X, y, feature_name, params, n_boot = 100, n_bins = 50) {
  # Create bin edges and centers
  feature_vals <- X[[feature_name]]
  bin_edges <- seq(min(feature_vals), max(feature_vals), length.out = n_bins + 1)
  bin_centers <- 0.5 * (bin_edges[-length(bin_edges)] + bin_edges[-1])
  
  # Initialize result matrix
  result_df <- data.frame(bin_center = bin_centers)
  
  cat("Starting bootstrap iterations...\n")
  
  for (i in 1:n_boot) {
    if (i %% 10 == 0) cat(sprintf("Bootstrap iteration %d/%d\n", i, n_boot))
    
    # Bootstrap resample
    set.seed(42 + i)
    boot_indices <- sample(nrow(X), nrow(X), replace = TRUE)
    X_resampled <- X[boot_indices, ]
    y_resampled <- y[boot_indices]
    
    # Fit XGBoost model
    dtrain <- xgb.DMatrix(data = as.matrix(X_resampled), label = y_resampled)
    model <- xgb.train(
      params = best_params,
      data = dtrain,
      nrounds = n_rounds,
      verbose = 0
    )
    
    # Create prediction wrapper for fastshap
    pfun <- function(object, newdata) {
      predict(object, as.matrix(newdata))
    }
    
    # Calculate SHAP values on original X
    shap_values <- explain(
      object = model,
      X = as.matrix(X),
      pred_wrapper = pfun,
      nsim = 100
    )
    
    # Extract SHAP values for the specific feature
    feature_col_idx <- which(colnames(X) == feature_name)
    shap_feature_vals <- shap_values[, feature_col_idx]
    
    # Create bins and calculate means
    binned <- cut(feature_vals, breaks = bin_edges, include.lowest = TRUE)
    
    # Group SHAP values by bins and calculate means
    bin_data <- data.frame(
      bin = binned,
      shap = shap_feature_vals
    )
    
    bin_means <- bin_data %>%
      group_by(bin, .drop = FALSE) %>%
      summarise(mean_shap = mean(shap, na.rm = TRUE), .groups = 'drop') %>%
      pull(mean_shap)
    
    # Store results
    result_df[[paste0("rep_", i)]] <- bin_means
  }
  
  # Set row names to bin centers for easier processing
  rownames(result_df) <- bin_centers
  
  return(result_df)
}

compute_shap_bin_summary <- function(shap_bin_df) {
  # Extract bin centers
  bin_centers <- shap_bin_df$bin_center
  
  # Calculate summary statistics across bootstrap replications
  shap_data <- shap_bin_df %>% select(-bin_center)
  
  means <- rowMeans(shap_data, na.rm = TRUE)
  lower <- apply(shap_data, 1, function(x) quantile(x, 0.025, na.rm = TRUE))
  upper <- apply(shap_data, 1, function(x) quantile(x, 0.975, na.rm = TRUE))
  
  summary_df <- data.frame(
    Bin_Center = bin_centers,
    SHAP_Mean = means,
    Lower_CI = lower,
    Upper_CI = upper
  )
  
  return(summary_df)
}

plot_shap_bin_averages <- function(summary_df, feature_name) {
  # Create plot with confidence intervals
  p <- ggplot(summary_df, aes(x = Bin_Center)) +
    geom_ribbon(aes(ymin = Lower_CI, ymax = Upper_CI), 
                alpha = 0.3, fill = "steelblue") +
    geom_line(aes(y = SHAP_Mean), color = "steelblue", linewidth = 1) +
    geom_hline(yintercept = 0, color = "gray", linetype = "dashed") +
    labs(
      x = feature_name,
      y = "Average SHAP Value (within bin)"
    ) +
    theme_minimal() +
    theme(
      legend.position = "none"  # Remove legend since we're not using it
    )
  
  return(p)
}

#==============================================================================
# Example function calls: x2
#==============================================================================

# Step 1: Generate bootstrapped SHAP bin values
shap_bin_df <- bootstrap_shap_bins(X, y, feature_name = "x2", params = best_params, n_boot = 100)
# Remove any rows that are all NA
shap_bin_df_clean <- shap_bin_df[complete.cases(shap_bin_df), ]

# Step 2: Summarize across replications
summary_df <- compute_shap_bin_summary(shap_bin_df_clean)

# Step 3: Plot
shap_plot <- plot_shap_bin_averages(summary_df, "x2")
print(shap_plot)

# Save plot
#ggsave("SHAP_bins_x2.png", shap_plot, width = 8, height = 5, dpi = 300)
