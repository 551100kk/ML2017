# 85824
import numpy
import math
import time
import random
import sys
import os
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


def histeqq(im,nbr_bins=256):

   #get image histogram
   imhist,bins = numpy.histogram(im.flatten(),nbr_bins,normed=True)
   cdf = imhist.cumsum() #cumulative distribution function
   cdf = 255 * cdf / cdf[-1] #normalize

   #use linear interpolation of cdf to find new pixel values
   im2 = numpy.interp(im.flatten(),bins[:-1],cdf)

   return im2.reshape(im.shape), cdf

train_in = []
train_out = []
test_in = []
test_out = []

train_N = 28709
test_N = 7178

with open('train.csv', 'r') as fp:
    fp.readline()
    for i in range(train_N):
        a = fp.readline().replace('\n','').split(',')
        label = int(a[0])
        feature = a[1].split(' ')
        feature = [int(x) for x in feature]
        ans = [0 for i in range(7)]
        ans[label] = 1
        train_in.append(feature)
        train_out.append(ans)
        tmp = []
        for i in range(48):
            for j in range(48):
                tmp.append(feature[i*48+47-j])
        train_in.append(tmp)
        train_out.append(ans)

N = len(train_in)
for T in range(N):
    out = numpy.array(train_in[T])
    out = out.reshape(48,48)
    out, h = histeqq(out)
    train_in[T] = out.reshape(48*48)
    for i in range(48*48):
        train_in[T][i] /= 255

with open('test.csv', 'r') as fp:
    fp.readline()
    for i in range(test_N):
        a = fp.readline().replace('\n','').split(',')
        label = int(a[0])
        feature = a[1].split(' ')
        feature = [int(x) for x in feature]
        test_in.append(feature)

N = len(test_in)
for T in range(N):
    out = numpy.array(test_in[T])
    out = out.reshape(48,48)
    out, h = histeqq(out)
    test_in[T] = out.reshape(48*48)
    for i in range(48*48):
        test_in[T][i] /= 255

total = len(train_in)

train_in = numpy.array(train_in)
train_out = numpy.array(train_out)
test_in = numpy.array(test_in)

test_in=test_in.reshape(test_in.shape[0],48,48,1)
train_in=train_in.reshape(train_in.shape[0],48,48,1)

model = load_model(os.path.join(os.path.dirname(__file__),'my_model2.h5'))
out = model.predict(test_in)

for i in range(test_N):
    ma = 0
    ans = -1
    tmp = []
    for j in range(7):
        if out[i][j] > ma:
            ma = out[i][j]
            ans = j
        tmp.append(0)
    tmp[ans] = 1
    test_out.append(tmp)

test_out = numpy.array(test_out)

train_in = numpy.append(train_in, test_in, axis=0)
train_out = numpy.append(train_out, test_out, axis=0)

print (len(train_in))
print (len(test_in))

model.fit(train_in, train_out, epochs=10, batch_size=128, validation_split=0.1)

model.save('my_model_semi.h5')