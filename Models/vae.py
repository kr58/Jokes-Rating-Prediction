import numpy as np
import pickle
import os
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import objectives
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, Callback
from Scripts.evaluation import *

# def VAE(batch_size=20, original_dim=100, encoding_dim1=64, encoding_dim2 = 32, latent_dim=16, nb_epochs=50, epsilon_std=1.0):
	
batch_size=20 
original_dim=100
encoding_dim1=64 
encoding_dim2 = 32 
latent_dim=16 
nb_epochs=50 
epsilon_std=1.0	
def vae_loss(x, x_bar):
    reconst_loss=original_dim*objectives.binary_crossentropy(x, x_bar)
    kl_loss = -0.5*K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return kl_loss + reconst_loss

def sampling(args):
    _mean,_log_var=args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=epsilon_std)
    return _mean+K.exp(_log_var/2)*epsilon

input_img = Input(shape=(original_dim,))
encoded1 = Dense(encoding_dim1, activation='sigmoid')(input_img)
encoded2 = Dense(encoding_dim2, activation='tanh')(encoded1)
z_mean = Dense(latent_dim)(encoded2)
z_log_var = Dense(latent_dim)(encoded2)

z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

h_decoder2 = Dense(encoding_dim2, activation='tanh')
h_decoder1 = Dense(encoding_dim1, activation='sigmoid')
x_bar = Dense(original_dim, activation='softmax')
h_decoded2 = h_decoder2(z)
h_decoded1 = h_decoder1(h_decoded2)
x_decoded = x_bar(h_decoded1)

autoencoder = Model(input_img, x_decoded)
autoencoder.compile(optimizer='adadelta', loss=vae_loss)

x_train = pickle.load(open("Data/train_data.file", "rb"))
print("number of training users: ", x_train.shape[0])

x_val = pickle.load(open("Data/val_data.file", "rb"))
x_val = x_val.todense()

weightsPath = "Data/weights.hdf5"
checkpointer = ModelCheckpoint(filepath=weightsPath, verbose=1, save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
autoencoder.fit(x_train, x_train, epochs=nb_epochs, batch_size=256, shuffle=True, validation_data=(x_val, x_val), callbacks=[checkpointer, reduce_lr])
VAE_NMAE(batch_size, original_dim, encoding_dim1, encoding_dim2, latent_dim, nb_epochs, epsilon_std)