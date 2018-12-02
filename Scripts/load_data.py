import pandas as pd
import numpy as np
import random
import pickle
import os, sys

from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix

def load_ratings(filename, header=['user_id', 'item_id', 'rating'], test_size=0.2, sep=","):
	df = pd.read_csv(filename, sep=sep, names=header, engine='python')

	n_users = df.user_id.unique().shape[0]
	n_items = df.item_id.unique().shape[0]

	train_data, test_data = train_test_split(df, test_size=test_size)
	train_data = pd.DataFrame(train_data)
	test_data = pd.DataFrame(test_data)

	train_row, train_col, train_rating = [], [], []
	for line in train_data.itertuples():
		train_row.append(int(line[1]))
		train_col.append(int(line[2]))
		train_rating.append(float(line[3]))
	train_matrix = csr_matrix((train_rating, (train_row, train_col)), shape=(n_users, n_items))

	test_row, test_col, test_rating = [], [], []
	for line in test_data.itertuples():
		test_row.append(int(line[1]))
		test_col.append(int(line[2]))
		test_rating.append(float(line[3]))
	test_matrix = csr_matrix((test_rating, (test_row, test_col)), shape=(n_users, n_items))

	print("Load data finished. Number of users:", n_users, "Number of items:", n_items)

	return train_matrix.todok(), test_matrix.todok(), n_users, n_items

def read_data(filename, header=['user_id', 'item_id', 'rating'], sep=","):
	raw_data = pd.read_csv(filename, sep=sep, names=header, engine='python')
	unique_user_ids = raw_data.user_id.unique()

	np.random.seed(1)
	np.random.shuffle(unique_user_ids)
	unique_movie_ids = raw_data.item_id.unique()

	number_of_users = len(unique_user_ids)
	number_of_movies = len(unique_movie_ids)
	print("total number of users: ", number_of_users)
	print("total number of movies: ", number_of_movies)

	val_user_ids, test_user_ids, train_user_ids, val_rows, val_cols, test_rows, test_cols, rows, cols = [], [], [], [], [], [], [], [], []
	for i in range(number_of_users):
		if i<2000:
			val_user_ids.append(unique_user_ids[i])
		elif i>=2000 and i<4000:
			test_user_ids.append(unique_user_ids[i])
		else:
			train_user_ids.append(unique_user_ids[i])			

	movie2id, user2id, test_user2id, val_user2id = {}, {}, {}, {}
	movie2id = dict((mid, i) for (i, mid) in enumerate(unique_movie_ids))
	user2id = dict((uid, i) for (i, uid) in enumerate(train_user_ids))

	print("creating training data....")
	for u_id in train_user_ids:
		m_ids = raw_data[(raw_data.user_id == u_id)]['item_id'].tolist()
		movie_indexes = [movie2id[m] for m in m_ids]
		rows.extend([user2id[u_id] for i in range(len(m_ids))])
		cols.extend(movie_indexes)

	train_data = csr_matrix((np.ones_like(rows),(np.array(rows), np.array(cols))), dtype='float64', shape=(len(train_user_ids), number_of_movies))

	pickle.dump(train_data, open("Data/train_data.file", "wb"))
	print("number of training users: ", len(train_user_ids))

	print("creating test data....")
	test_user2id = dict((uid, i) for (i, uid) in enumerate(test_user_ids))
	for u_id in test_user_ids:
		m_ids = raw_data[(raw_data.user_id == u_id)]['item_id'].tolist()
		movie_indexes = [movie2id[m] for m in m_ids]
		test_rows.extend([test_user2id[u_id] for i in range(len(m_ids))])
		test_cols.extend(movie_indexes)

	test_data = csr_matrix((np.ones_like(test_rows),(np.array(test_rows), np.array(test_cols))), dtype='float64', shape=(len(test_user_ids), number_of_movies))
	pickle.dump(test_data, open("Data/test_data.file", "wb"))
	print("number of test users: ", len(test_user_ids))

	print("creating validation data")
	val_user2id = dict((uid, i) for (i, uid) in enumerate(val_user_ids))
	for u_id in val_user_ids:
		m_ids = raw_data[(raw_data.user_id == u_id)]['item_id'].tolist()
		movie_indexes = [movie2id[m] for m in m_ids]
		val_rows.extend([val_user2id[u_id] for i in range(len(m_ids))])
		val_cols.extend(movie_indexes)

	val_data = csr_matrix((np.ones_like(val_rows),(np.array(val_rows), np.array(val_cols))), dtype='float64', shape=(len(val_user_ids), number_of_movies))
	pickle.dump(val_data, open("Data/val_data.file", "wb"))
	print("number of validation users: ", len(val_user_ids))