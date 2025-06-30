#==============================================================================
# simulate_data.py
# Author: Adam D. Rennhoff
# Purpose: Simulate features and target variable for use in guide
# Output: ML4Econ_data.csv file
#==============================================================================

# ------ Load necessary libraries ------  
import numpy as np
import pandas as pd

# ------ Set simulation parameters ------
np.random.seed(42)
n = 1000    # number of observations
p = 15    # number of features/variables (X)

#=========================================================
# Step 1: simulate indendent features
#=========================================================
X = np.random.normal(size=(n,p))
columns = [f'x{i+1}' for i in range(p)]
df = pd.DataFrame(X, columns=columns)

#========================================================= 
# Step 2: Create a non-linear function of a few variables
# Use a few features with non-linear effects
#=========================================================
signal = (
    1.2 * np.sin(df['x1']) +
    df['x2']**2 +
    np.exp(-df['x3']) +
    df['x4'] * df['x5'] +    # interaction term
    1.5 * np.log(np.abs(df['x6']) + 1) + 
    2 * df['x8'] +
    df['x5'] * df['x9'] +
    -1.75 * df['x9'] +
    1.5 * np.minimum(1, df['x7'])
)

# ------ Add noise ------
epsilon = np.random.normal(scale=0.5, size=n)
df['y'] = signal + epsilon

# ------ Move 'y' to the first column for appearance ------
cols = ['y'] + [col for col in df.columns if col != 'y']
df_final = df[cols]

print(df_final.shape)
print(df_final.columns)

#=========================================================
# Step 3: Export as CSV file
#=========================================================
df_final.to_csv('ML4Econ_data.csv')

