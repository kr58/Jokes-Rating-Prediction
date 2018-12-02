import numpy as np
import pickle
import os
import math
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.layers import Input, Dense, Lambda
from keras.models import Model, load_model
from keras import objectives
from keras import backend as K

def RMSE(error, num):
    return np.sqrt(error / num)

def NMAE(error_mae, num):
    return (error_mae / num)/20

def VAE_NMAE(batch_size, original_dim, encoding_dim1, encoding_dim2, latent_dim, nb_epochs, epsilon_std):

	def vae_loss(x,x_bar):
	    reconst_loss=original_dim*objectives.binary_crossentropy(x, x_bar)
	    kl_loss= -0.5*K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
	    return kl_loss + reconst_loss

	def sampling(args):
	    _mean,_log_var=args
	    epsilon=K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=epsilon_std)
	    return _mean+K.exp(_log_var/2)*epsilon
	    
	input_img = Input(shape=(original_dim,))
	
	encoded1 = Dense(encoding_dim1, activation='sigmoid')(input_img)
	encoded2 = Dense(encoding_dim2, activation='tanh')(encoded1)
	z_mean = Dense(latent_dim)(encoded2)
	z_log_var = Dense(latent_dim)(encoded2)

	z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

	h_decoder2 = Dense(encoding_dim2, activation='tanh')
	h_decoder1 = Dense(encoding_dim1, activation='sigmoid')
	x_bar = Dense(original_dim,activation='softmax')
	h_decoded2 = h_decoder2(z)
	h_decoded1 = h_decoder1(h_decoded2)
	x_decoded = x_bar(h_decoded1)

	vae = Model(input_img, x_decoded)

	vae.load_weights("Data/weights.hdf5")

	x_test_matrix = pickle.load(open("Data/test_data.file", "rb"))
	x_test_matrix = x_test_matrix.todense()
	x_test = np.squeeze(np.asarray(x_test_matrix))

	x_test_reconstructed = vae.predict(x_test, batch_size=batch_size)

	print("NMAE: ", mean_absolute_error(np.array(x_test), np.array(x_test_reconstructed))/20)
	print('RMSE: ', mean_squared_error(np.array(x_test), np.array(x_test_reconstructed)))
