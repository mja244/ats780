# ats780
HW assignments for ATS780: Machine Learning for the Atmospheric Sciences taught by Elizabeth Barnes

## File description

### HW1 folder

RF_cong_dist_pred.py

This is the main workflow file for HW1. Here, an RF model attempts to predict whether a Congressional District will vote Democrat or Republican based on multiple predictors, including race, gender, income, education, national polling, and local polling. The goal is to predict which party will control the House of Representatives in a given election (majority >217 seats).

data_prep.py

This workflow focuses on preparing the data for the RF. The data is first loaded, split based on features/label, and then split based on testing and training data.

### HW2 folder

cnn_GF_error.py

Main workflow file for HW2. Here, a CNN trains on SST from the MPI-ESM Grand Ensemble to predict the error in the GF precipitation response compared to the MPI-ESM Grand Ensemble output. Separately, the precipitation GF is convolved with the MPI-ESM GE SST, which gives a precipitation prediction.  This is then subtracted from the actual MPI-ESM GE output to get the error. This error is used as the predictor and the SST pattern is the predictand.

Note that most processing of the data was done on Levante (German supercomputer).
