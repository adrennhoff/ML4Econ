#==============================================================================
# fit_ML_in_R.R
# Author: Adam D. Rennhoff
# Purpose: Load simulated data, tune XGBoost and Random Forest models, save best models
# Input: ML4Econ_data.csv
# Output: xgb_model.json (same name as Python script)
#==============================================================================

# ------ Load necessary libraries ------
library(tidyverse)
library(xgboost)
library(randomForest)
library(caret)
library(MLmetrics)

# Set seed for reproducibility
set.seed(42)

# ------ Load data ------
df <- read.csv("ML4Econ_data.csv", row.names = 1)
X <- df %>% select(-y)
y <- df$y

# ------ Split data ------
# Create train/test split (80/20)
train_indices <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[train_indices, ]
X_test <- X[-train_indices, ]
y_train <- y[train_indices]
y_test <- y[-train_indices]

#==============================================================================
# XGBoost
#==============================================================================

# ------ Prepare data for XGBoost ------
# Convert to DMatrix format
dtrain <- xgb.DMatrix(data = as.matrix(X_train), label = y_train)
dtest <- xgb.DMatrix(data = as.matrix(X_test), label = y_test)

# ------ Set hyperparameter grid ------
xgb_grid <- expand.grid(
  nrounds = c(100, 200, 300),
  max_depth = c(3, 4, 5, 6),
  eta = c(0.01, 0.05, 0.1),
  subsample = c(0.8, 0.9, 1.0),
  colsample_bytree = 1,
  min_child_weight = 1,
  gamma = 0
)

# ------ Set up train control for cross-validation ------
train_control <- trainControl(
  method = "cv",
  number = 5,
  verboseIter = TRUE,
  allowParallel = TRUE
)

# ------ Grid search for XGBoost ------
cat("Tuning XGBoost hyperparameters...\n")
xgb_tune <- train(
  x = X_train,
  y = y_train,
  method = "xgbTree",
  trControl = train_control,
  tuneGrid = xgb_grid,
  metric = "RMSE",
  objective = "reg:squarederror"
)

# ------ Print best hyperparameters ------
cat("Best XGBoost hyperparameters found:\n")
print(xgb_tune$bestTune)

# ------ Evaluate XGBoost performance ------
xgb_pred <- predict(xgb_tune, X_test)
xgb_mse <- MSE(y_test, xgb_pred)
cat(sprintf("XGBoost Test MSE: %.4f\n", xgb_mse))

# ------ Refit final XGBoost model on full dataset ------
best_params <- as.list(xgb_tune$bestTune)
best_params$data <- xgb.DMatrix(data = as.matrix(X), label = y)
best_params$objective <- "reg:squarederror"
best_params$verbose <- 0

final_xgb_model <- do.call(xgboost, best_params)

# ------ Save XGBoost model ------
#xgb.save(final_xgb_model, "xgb_model.json")

#==============================================================================
# Random Forest
#==============================================================================

# ------ Set hyperparameter grid for Random Forest ------
rf_grid <- expand.grid(
  ntree = c(100, 200, 300),           
  maxnodes = c(-1, 31, 1023),         
  mtry = c(sqrt(ncol(X_train)), log2(ncol(X_train)), ncol(X_train))
)

# ------ Manual grid search for Random Forest ------
# ------ Mimics Python's GridSearchCV ------
# ------ Could also use caret train() with tuneGrid -----
cat("Tuning Random Forest hyperparameters...\n")

# Initialize variables to store best results
best_rf_mse <- Inf
best_rf_params <- NULL
best_rf_model <- NULL

# Perform manual grid search with cross-validation
for (i in 1:nrow(rf_grid)) {
  cat(sprintf("Testing RF configuration %d/%d\n", i, nrow(rf_grid)))
  
  # Get current parameters
  current_mtry <- rf_grid$mtry[i]
  current_ntree <- rf_grid$ntree[i]
  current_nodesize <- rf_grid$nodesize[i]
  
  # Perform 5-fold cross-validation
  cv_mse <- numeric(5)
  folds <- createFolds(y_train, k = 5, list = TRUE)
  
  for (fold in 1:5) {
    # Get fold indices
    fold_indices <- folds[[fold]]
    
    # Create fold train/validation sets
    X_fold_train <- X_train[-fold_indices, ]
    X_fold_val <- X_train[fold_indices, ]
    y_fold_train <- y_train[-fold_indices]
    y_fold_val <- y_train[fold_indices]
    
    # Fit Random Forest
    rf_fold <- randomForest(
      x = X_fold_train,
      y = y_fold_train,
      ntree = current_ntree,
      mtry = current_mtry,
      nodesize = current_nodesize
    )
    
    # Predict and calculate MSE
    rf_fold_pred <- predict(rf_fold, X_fold_val)
    cv_mse[fold] <- MSE(y_fold_val, rf_fold_pred)
  }
  
  # Calculate mean CV MSE
  mean_cv_mse <- mean(cv_mse)
  
  # Update best model if current is better
  if (mean_cv_mse < best_rf_mse) {
    best_rf_mse <- mean_cv_mse
    best_rf_params <- list(
      mtry = current_mtry,
      ntree = current_ntree,
      nodesize = current_nodesize
    )
  }
}

# ------ Print best Random Forest hyperparameters ------
cat("Best Random Forest hyperparameters found:\n")
print(best_rf_params)

# ------ Fit best Random Forest model ------
best_rf_model <- randomForest(
  x = X_train,
  y = y_train,
  ntree = best_rf_params$ntree,
  mtry = best_rf_params$mtry,
  nodesize = best_rf_params$nodesize
)

# ------ Evaluate Random Forest performance ------
rf_pred <- predict(best_rf_model, X_test)
rf_mse <- MSE(y_test, rf_pred)
cat(sprintf("Random Forest Test MSE: %.4f\n", rf_mse))
