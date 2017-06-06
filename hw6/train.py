import numpy as np 
import keras
import random
from keras.callbacks import EarlyStopping, ModelCheckpoint	
import os
from keras.models import load_model
import sys

user_num = 0
movie_num = 0

def read_data(path, train):
	
	user = []
	movie = []
	rate = []
	
	global user_num, movie_num

	with open(path, 'r') as fp:
		fp.readline()
		for line in fp:
			arr = line.replace('\n', '').split(',')
			user.append([int(arr[1])])
			movie.append([int(arr[2])])
			if train == True:
				rate.append([float(arr[3])])

			user_num = max(user_num, int(arr[1]) + 1)
			movie_num = max(movie_num, int(arr[2]) + 1)
	
	if train == True:
		c = list(zip(user, movie, rate))
		random.shuffle(c)
		user, movie, rate = zip(*c)

	return np.array(user), np.array(movie), np.array(rate)


def output(out, path):
	with open(path, 'w') as fp:
		fp.write('TestDataID,Rating\n')
		N = out.shape[0]
		for i in range(N):
			fp.write('%d,%f\n' % (i + 1, out[i][0]))

def main():

	data_dir = sys.argv[1]
	ans_csv = sys.argv[2]

	print ('Reading data')
	#train_user, train_movie, train_out = read_data('train.csv', True)
	#print ('Training data: %d' % (train_user.shape[0]))

	test_user, test_movie, _ = read_data(data_dir + 'test.csv', False)
	print ('Testing data: %d' % (test_user.shape[0]))

	#print ('Total users: %d' % user_num)
	#print ('Total movies: %d' % movie_num)

	

	user_input = keras.layers.Input(shape=[1])
	user_vec = keras.layers.normalization.BatchNormalization()(user_input)
	user_vec = keras.layers.Flatten()(keras.layers.Embedding(user_num, 128)(user_input))
	user_vec = keras.layers.Dropout(0.5)(user_vec)
	user_vec = keras.layers.normalization.BatchNormalization()(user_vec)
	# user_vec = keras.layers.Dropout(0.3)(keras.layers.Dense(64, activation='relu')(user_vec))


	movie_input = keras.layers.Input(shape=[1])
	movie_vec = keras.layers.normalization.BatchNormalization()(movie_input)
	movie_vec = keras.layers.Flatten()(keras.layers.Embedding(movie_num, 128)(movie_input))
	movie_vec = keras.layers.Dropout(0.5)(movie_vec)
	movie_vec = keras.layers.normalization.BatchNormalization()(movie_vec)
	# movie_vec = keras.layers.Dropout(0.1)(keras.layers.Dense(32, activation='relu')(movie_vec))
	# movie_vec = keras.layers.Dropout(0.3)(keras.layers.Dense(64, activation='relu')(movie_vec))

	input_vecs = keras.layers.dot([movie_vec, user_vec], axes=-1)
	# input_vecs = keras.layers.concatenate([movie_vec, user_vec])
	# input_vecs = keras.layers.Add([user_vec, movie_vec])
	# nn = keras.layers.Dense(1, activation='relu')(input_vecs)


	model = keras.models.Model([user_input, movie_input], input_vecs)
	model.compile(loss='mean_squared_error', optimizer='adam')
	model.summary()

	# earlystopping = EarlyStopping(monitor='val_loss',patience = 10, verbose=1, mode='min')
	# checkpoint = ModelCheckpoint(monitor='val_loss',filepath='best.hdf5', verbose=1,save_best_only=True,save_weights_only=True,mode='min')
	# model.fit([train_user, train_movie], train_out, batch_size=8096, epochs=200) #, validation_split=0.1) #, callbacks=[earlystopping,checkpoint])
	# model.load_weights(os.path.join(os.path.dirname(__file__),'best.hdf5'))
	# model.save('my_model.h5')
	model = load_model(os.path.join(os.path.dirname(__file__),'my_model.h5'))

	out = model.predict([test_user, test_movie])
	print (out.shape)

	output(out, ans_csv)

	print ('OKOK')
if __name__=='__main__':
    main()