# Convolutional neural network that trains on SST patterns
# from the MPI-GE (100 ens members) to predict the
# the precipitation response to the coupled model output

import numpy as np
import seaborn as sb
import xarray as xr
import tensorflow as tf
import tensorflow.keras as keras
import sklearn
import matplotlib.pyplot as plt
import pickle
from scipy.signal import detrend
from settings import settings

# Load data (previously processed on German supercomputer Levante)
# 'predictand' is the precipitation output predicted by the MPI GE
# 'predictor' is the SST to be inputted to the model

predictand = np.load('data/pr_hist_rcp85_mpige.npy') # 100 ens, 250 yrs

predictor = np.load('data/rcp85_tos.npy') # 100 ens, 250 yrs, 96 lat, 192 lon

# change nan (land area) to zeros
predictor = np.nan_to_num(predictor)
predictand = np.nan_to_num(predictand)

# detrend data
predictor = detrend(predictor, axis=1)
predictand = detrend(predictand, axis=1)

# Combine ensemble members and years
predictor = predictor.reshape(25000, 96, 192)
predictand = predictand.reshape(25000)


# Split data into training, validation, and testing
# split along ensemble members (40 for training, 15 for validation, 15 for testing)

Xtrain = predictor[0:3750,:,:,np.newaxis] # 0-15k, 0-10k
Xval = predictor[3750:5000,:,:,np.newaxis] # 15k-20k, 
Xtest = predictor[5000:6250,:,:,np.newaxis] # 20k-25k

Ytrain = predictand[0:3750]
Yval = predictand[3750:5000]
Ytest = predictand[5000:6250]
#Ytrain = np.full(100,1)
#Yval = np.full(50,1)

print('Shapes:')
print('  Xtrain: ', Xtrain.shape)
print('  Xval: ', Xval.shape)
print('  Xtest: ', Xtest.shape)

print('  Ytrain: ', Ytrain.shape)
print('  Yval: ', Yval.shape)
#print('  Ytest: ', Ytest.shape)

Xmean = np.nanmean(predictor, axis=0)
Xstd = np.std(predictor, axis=0)
Ymean = np.nanmean(predictand)
Ystd = np.std(predictand)

Xmean = Xmean[np.newaxis,:,:,np.newaxis]
Xstd = Xstd[np.newaxis,:,:,np.newaxis]

Xtrain = np.nan_to_num((Xtrain - Xmean) / Xstd)
Xval = np.nan_to_num((Xval - Xmean) / Xstd)
Xtest = np.nan_to_num((Xtest - Xmean) / Xstd)

Ytrain = np.nan_to_num((Ytrain - Ymean) / Ystd)
Yval = np.nan_to_num((Yval - Ymean) / Ystd)
#Ytest = np.nan_to_num((Ytest - Ymean) / Ystd)

np.save('data/Xtrain_pr', Xtrain)
np.save('data/Xval_pr', Xval)
np.save('data/Xtest_pr', Xtest)
np.save('data/Ytrain_pr', Ytrain)
np.save('data/Yval_pr', Yval)
#np.save('data/Ytest', Ytest)





##----------------Build the model---------------------------------
#
#def build_model(Xtrain, Ytrain, settings):
#	# create input layer
#	input_layer = tf.keras.layers.Input(shape=Xtrain.shape[1:])
#
#	## create normalization layer
#	#normalizer = tf.keras.layers.Normalization(axis=(1,))
#	#normalizer.adapt(Xtrain)
#	#layers = normalizer(input_layer)
#
#	# use this if you don't normalize
#	layers = tf.keras.layers.Layer()(input_layer)
#
#	# convolutional layers (repeat 3 times: conv, conv, pooling)
#	for k_size, activation in zip(settings['kernels'], settings['kernel_act']):
#		layers = tf.keras.layers.Conv2D(
#			filters=1, 
#			kernel_size=k_size, 
#			strides=2,
#			activation=activation, 
#			padding='same',
#			#input_shape=[96, 192, 1],
#		)(layers)
#
#		#layers = tf.keras.layers.Conv2D(
#		#	filters=1, 
#		#	kernel_size=k_size, 
#		#	strides=1,
#		#	activation=activation, 
#		#	padding='same',
#		#	input_shape=[96, 192, 1],
#		#)(layers)
#
#		# pooling layer (not necessary?)
#		layers = tf.keras.layers.MaxPool2D(pool_size=2, padding='same')(layers)
#
#	conv_shape = layers.shape
#
#	# dense layers (3 layers including output layer of 1)
#	layers = tf.keras.layers.Flatten()(layers)
#	for hidden, activation in zip(settings['hiddens'], settings['act_fun']):
#		layers = tf.keras.layers.Dense(
#			units=hidden,
#			activation=activation,
#			kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0, l2=0),
#			bias_initializer=tf.keras.initializers.RandomNormal(seed=settings['random_seed']),
#			kernel_initializer=tf.keras.initializers.RandomNormal(seed=settings['random_seed']),
#		)(layers)
#
#	# output layer
#	output_layer = tf.keras.layers.Dense(
#		units=1,
#		bias_initializer=tf.keras.initializers.RandomNormal(seed=settings['random_seed'] + 1),
#		kernel_initializer=tf.keras.initializers.RandomNormal(seed=settings['random_seed'] + 2),
#		)(layers)
#
#	# construct model
#	model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
#	model.summary()
#
#	return model
#
##------------------------Compile model----------------------------
#
#def compile_model(model, settings):
#	model.compile(
#		optimizer=tf.keras.optimizers.Adam(learning_rate=settings['learning_rate']),
#		loss=settings['loss'],
#		metrics=settings['loss'],
#			#tf.keras.metrics.MSE(),
#		#],
#	)
#	return model
#
#
#
##---------------------------Settings------------------------------
#
#tf.keras.backend.clear_session()
#tf.keras.utils.set_random_seed(settings['random_seed'])
#
#model = build_model(Xtrain, Ytrain, settings)
#model = compile_model(model, settings)
#
#model.save('small_cnn_model', save_format='tf')

print('hi')
model = tf.keras.models.load_model('small_cnn_model.keras')

loss, acc = model.evaluate(Xval, Yval, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))


##----------------------Train the model!----------------------------
#
#history = model.fit(
#	Xtrain,
#	Ytrain,
#	epochs=settings['n_epochs'],
#	batch_size=settings['batch_size'],
#	shuffle=True,
#	validation_data=[Xval, Yval],	
#	verbose=1,
#	)

#with open('training_history_small.pkl', 'wb') as file:
#	pickle.dump(history.history, file)

##-----------plot loss---------------------------------
#fig, axs = plt.subplots()
#
#axs.plot(history.history["loss"], label="training")
#axs.plot(history.history["val_loss"], label="validation")
#axs.set_xlabel("Epoch")
#axs.set_ylabel("Loss")
#axs.legend()
#
#plt.show()


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

ax.plot(np.arange(0, 150), errVal[0:150])
ax.plot(np.arange(0, 150), Yval[0:150])

ax.set_xlabel('Year')
ax.set_ylabel('SUWS precip (mm/day normalized)')

plt.show()
plt.close()


#-------------plot truth vs what model predicts------------
errTest = model.predict(Xtest)

fig, ax = plt.subplots()

ax.scatter(errTest, Ytest)

ax.set_xlabel('CNN prediction')
ax.set_ylabel('AOGCM (actual)')

ax.set_xlim((-2.5, 2.5))
ax.set_ylim((-2.5, 2.5))

plt.show()
plt.close()
