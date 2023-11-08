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

# change nan (land area) to zeros
predictor = np.nan_to_num(predictor)
predictand = np.nan_to_num(predictand)

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

print(Xtrain[0,100,45,:])


#----------------Build the model---------------------------------

def build_model(Xtrain, Ytrain, settings):
	# create input layer
	input_layer = tf.keras.layers.Input(shape=Xtrain.shape[1:])

	# create normalization layer
	normalizer = tf.keras.layers.Normalization(axis=(1,))
	normalizer.adapt(Xtrain)
	layers = normalizer(input_layer)

	## what does this do?
	#layers = Layer()(input_layer)

	# convolutional layers (repeat 3 times: conv, conv, pooling)
	for k_size, activation in zip(settings['kernels'], settings['kernel_act']):
		layers = tf.keras.layers.Conv2D(
			filters=32, 
			kernel_size=k_size, 
			strides=1,
			activation=activation, 
			padding='same',
		)(layers)

		layers = tf.keras.layers.Conv2D(
			filters=32, 
			kernel_size=k_size, 
			strides=1,
			activation=activation, 
			padding='same',
		)(layers)

		# pooling layer (not necessary?)
		layers = tf.keras.layers.MaxPool2D(pool_size=2, padding='same')(layers)

	conv_shape = layers.shape

	# dense layers (3 layers including output layer of 1)
	layers = tf.keras.layers.Flatten()(layers)
	for hidden, activation in zip(settings['hiddens'], settings['act_fun']):
		layers = tf.keras.layers.Dense(
			units=hidden,
			activation=activation,
			kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0, l2=0),
			bias_initializer=tf.keras.initializers.RandomNormal(seed=settings['random_seed']),
			kernel_initializer=tf.keras.initializers.RandomNormal(seed=settings['random_seed']),
		)(layers)

	# output layer
	output_layer = tf.keras.layers.Dense(
		units=1,
		bias_initializer=tf.keras.initializers.RandomNormal(seed=settings['random_seed'] + 1),
		kernel_initializer=tf.keras.initializers.RandomNormal(seed=settings['random_seed'] + 2),
		)(layers)

	# construct model
	model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
	model.summary()

	return model

#------------------------Compile model----------------------------

def compile_model(model, settings):
	model.compile(
		optimizer=tf.keras.optimizers.Adam(learning_rate=settings['learning_rate']),
		loss=settings['loss'],
		metrics=settings['loss'],
			#tf.keras.metrics.MSE(),
		#],
	)
	return model



#---------------------------Settings------------------------------

settings = {
	'network_type': 'cnn',
	'kernel_size': 5,
	'kernels': [32, 32, 32],
	'kernel_act': ['relu', 'relu', 'relu'],
	'hiddens': [16, 8],
	'act_fun': ['relu', 'relu'],
	'learning_rate': 0.000005,
	'batch_size': 32,
	'rng_seed': None,
	'rng_seed_list': [123, ],
	'n_epochs': 1,#25_000,
	'patience': 10,
	'loss': 'mse',
	'random_seed': 33,
}	

tf.keras.backend.clear_session()
tf.keras.utils.set_random_seed(settings['random_seed'])

model = build_model(Xtrain, Ytrain, settings)
model = compile_model(model, settings)


#----------------------Train the model!----------------------------

