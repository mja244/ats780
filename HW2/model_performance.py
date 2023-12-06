# How well did the model perform?

import numpy as np
import seaborn as sb
import xarray as xr
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
from scipy.signal import detrend
from settings import settings

tf.keras.backend.clear_session()
tf.keras.utils.set_random_seed(settings['random_seed'])

#--------------load model and data-----------------------------------
## load model history
with open('training_history_small.pkl', 'rb') as file:
	history = pickle.load(file)

model = tf.keras.models.load_model('small_cnn_model')

Xtrain = np.load('data/Xtrain_pr.npy')
Xval = np.load('data/Xval_pr.npy')
#Xtest = np.load('data/Xtest_pr.npy')

Ytrain = np.load('data/Ytrain_pr.npy')
Yval = np.load('data/Yval_pr.npy')
#Ytest = np.load('data/Ytest_pr.npy')



#-----------plot loss---------------------------------
fig, axs = plt.subplots()

axs.plot(history["loss"], label="training")
axs.plot(history["val_loss"], label="validation")
axs.set_xlabel("Epoch")
axs.set_ylabel("Loss")
axs.legend()

plt.show()


#------------plot predictions--------------------------
errTrain = model.predict(Xtrain)
errVal = model.predict(Xval)
#errTest = model.predict(Xtest)
#
#np.save('data/errTrain', errTrain)
#np.save('data/errVal', errVal)
#np.save('data/errTest', errTest)
#
#errTrain = np.load('data/errTrain.npy')
#errVal = np.load('data/errVal.npy')

fig, ax = plt.subplots()

ax.plot(np.arange(0, 50), errTrain[0:50])
ax.plot(np.arange(0, 50), Ytrain[0:50])

plt.show()
