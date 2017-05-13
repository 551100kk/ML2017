import numpy as np
import math
import time
import random
import sys
#from PIL import Image

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D, Flatten, AveragePooling2D, Cropping2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.datasets import mnist
from keras.regularizers import l2
from keras import regularizers
from keras.models import load_model

train_in = np.append(np.load('train_in2.npy'), np.load('train_in2.npy'), axis=0)
train_out = np.append(np.load('train_label2.npy'), np.load('train_label2.npy'), axis=0)

arr = []

N = len(train_in)
for i in range(N):
	train_out[i][0] += 1
	if train_out[i][0] >= 10 and train_out[i][0]<=30:
		arr.append(i)

train_in = train_in[arr]
train_out = train_out[arr]


model = Sequential()

model.add(Dense(input_dim=80,output_dim=80))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(80))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(70))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(60))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(40))
model.add(Activation('relu'))
model.add(Dense(1))
#model.add(Activation('linear'))
model.summary()
model.compile(loss='mean_squared_logarithmic_error', optimizer='SGD')

for i in range(25):
	print ((i))
	model.fit(train_in, train_out, epochs=20, validation_split=0.1)
	model.save('my_model_msle_3060.h5')