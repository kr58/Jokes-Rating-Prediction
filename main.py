import numpy as np
import argparse
from os import path
import tensorflow as tf
from Scripts.load_data import load_ratings, read_data

from Models.nnmf import NNMF
from Models.mf import MF
from Models.nrr import NRR
from Models.autorec import IAutoRec, UAutoRec

def parse_args(choices):
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', choices=choices, default='LRML')
	return parser.parse_args()

if __name__ == '__main__':

	choices = ['MF', 'NNMF', 'NRR', 'I-AutoRec', 'U-AutoRec', 'VAE']
	learning_rate = 1e-3
	batch_size = 256

	args = parse_args(choices)
	if args.model == "VAE":
		if not path.exists('Data/train_data.file'):
			read_data('Data/jester-modified.csv')
		from Models.vae import *
	else:
		train_data, test_data, n_user, n_item = load_ratings('Data/jester-modified.csv')
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		with tf.Session(config=config) as sess:
			if args.model == "MF":
				model = MF(sess, n_user, n_item, batch_size=batch_size)
			if args.model == "NNMF":
				model = NNMF(sess, n_user, n_item, learning_rate=learning_rate)
			if args.model == "NRR":
				model = NRR(sess, n_user, n_item)
			if args.model == "I-AutoRec":
				model = IAutoRec(sess, n_user, n_item)
			if args.model == "U-AutoRec":
				model = UAutoRec(sess, n_user, n_item)

			if args.model != 'VAE':
				model.build_network()
				model.execute(train_data, test_data)