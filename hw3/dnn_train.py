# 85824
import numpy
import math
import time
import random
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

train_csv = sys.argv[1]

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

vi = []
vo = []

train_N = 28709

with open(train_csv, 'r') as fp:
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

total = len(train_in)

vi = train_in[: int(total * 0.1)]
train_in = train_in[int(total * 0.1):]
vo = train_out[: int(total * 0.1)]
train_out = train_out[int(total * 0.1):]

train_in = numpy.array(train_in)
train_out = numpy.array(train_out)

vi = numpy.array(vi)
vo = numpy.array(vo)


model2 = Sequential()

model2.add(Dense(input_dim=48*48,output_dim=689))
model2.add(Dense(500))
model2.add(Activation('relu'))
model2.add(Dense(7))
model2.add(Activation('softmax'))
model2.summary()
model2.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])

with open("progress.txt", "w") as fp:
    for i in range(100):
        model2.fit(train_in, train_out, epochs=1, batch_size=128, validation_data=(vi, vo))
        score = model2.evaluate(train_in, train_out)
        a = score[1]
        score = model2.evaluate(vi, vo)
        b = score[1]
        fp.write("%f %f\n" % (a, b))
model2.save('my_model.h5')