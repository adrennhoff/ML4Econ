# Interpreting Machine Learning Models: A Practical Guide for Applied Economists

**Version:** July 14, 2025

This Github repo contains supplemental materials for my paper _Interpreting Machine Learning Models: A Practical Guide for Applied Economists_. The most recent version of the paper can be found in this [Github repo](Machine_Learning_for_Economists.pdf) or on my [SSRN research page](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5350967).

## Brief Outline of Materials

In order to encourage the wider adoption of machine learning techniques, I am providing a variety of resources in this Github repo. The simulated data used to fit all models and produce the resulting plots can be found in [ML4Econ_data.csv](ML4Econ_data.csv).

My code (see below) fits an XGBoost model. This model has been saved as a `.JSON` file, allowing you to work with the exact model fit on my computer. The model is saved as [xgb_model.json](xgb_model.json).

### Python Scripts

The [data](ML4Econ_data.csv) used in this guide was simulated using the [simulate_data.py](<Python scripts/simulate_data.py>). The folder contains several other script files:

- [fit_ML_model.py](<Python scripts/fit_ML_model.py>) : tunes and fits several different models; saving the [final chosen model](xgb_model.json)

- [feature_importance.py](<Python scripts/feature_importance.py>) : contains the code to replicate Section 3 of the paper on feature importance

- [directional_influence.py](<Python scripts/directional_influence.py>) : contains the code to replicate Section 4 of the paper on directional influence and magnitude

- [supplemental_plots.py](<Python scripts/supplemental_plots.py>) : contains supplemental code to produce beeswarm and waterfall plots, as shown in the Appendix

### R Scripts

As noted above, the data simulation, model fitting, and output plots are all produced using Python code. As many applied economists are more familiar with working in R, I have also included equivalent R code.

_Note_: there may be minor differences in appearance between the plots in the paper and those produced by R. All results are qualitatively similar, however.

- [fit_ML_in_R.R](<R scripts/fit_ML_in_R.R>) : contains the code to tune and fit the random forest and XGBoost models; final model can be saved in the same `.JSON` format as the model fit in Python

- [ML_plots.R](<R scripts/ML_plots.R>) : an abbreviated version of the plots presented in the paper

##### Questions? Suggestions?

Please feel free to [contact me](mailto:Adam.Rennhoff@mtsu.edu) with any questions or suggestions you have.