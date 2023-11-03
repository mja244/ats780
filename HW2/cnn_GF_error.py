# Convolutional neural network that trains on SST patterns
# from the MPI-GE (100 ens members) to predict the error in
# the GF response to the coupled model output

import numpy as np
import seaborn as sb
import xarray as xr
import tensorflow as tf
import tensorflow.keras as keras
import sklearn
import matplotlib.pyplot as plt

# Load data (previously processed on German supercomputer Levante)
# 'response' is the precipitation response predicted from the GF (calculated on Levante)
# 'output' is the precipitation output predicted by the MPI GE
# 'predictor' is the SST to be inputted to the model
# 'predictand' is the error of the response compared to output

response = np.load('data/pr_hist_rcp85_response.npy') # 100 ens members, 250 years
output = np.load('data/pr_hist_rcp85_mpige.npy') 

predictor = np.load('data/rcp85_tos.npy') # 100 ens, 250 yrs, 96 lat, 192 lon
predictand = response - output

# Split data into training, validation, and testing
# split along ensemble members (60 for training, 20 for validation, 20 for testing)

Xtrain = predictor[0:60,:,:,:]
Xval = predictor[60:80,:,:,:]
Xtest = predictor[80:,:,:,:]

Ytrain = predictand[0:60,:]
Yval = predictand[60:80,:]
Ytest = predictand[80:,:]

print('Shapes:')
print('  Xtrain: ', Xtrain.shape)
print('  Xval: ', Xval.shape)
print('  Xtest: ', Xtest.shape)

print('  Ytrain: ', Ytrain.shape)
print('  Yval: ', Yval.shape)
print('  Ytest: ', Ytest.shape)

