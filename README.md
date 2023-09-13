# ats780
HW assignments for ATS780: Machine Learning for the Atmospheric Sciences taught by Elizabeth Barnes

## File description

### HW1 folder

RF_cong_dist_pred.py

This is the main workflow file for HW1. Here, an RF model attempts to predict whether a Congressional District will vote Democrat or Republican based on multiple predictors, including race, gender, income, education, national polling, and local polling. The goal is to predict which party will control the House of Representatives in a given election (majority >217 seats).

data_prep.py

This workflow focuses on preparing the data for the RF. The data is first loaded, split based on features/label, and then split based on testing and training data.
