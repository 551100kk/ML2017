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
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os

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

total = len(train_in)

vi = train_in[: int(total * 0.1)]
train_in = train_in[int(total * 0.1):]
vo = train_out[: int(total * 0.1)]
train_out = train_out[int(total * 0.1):]

train_in = numpy.array(train_in)
train_out = numpy.array(train_out)

vi = numpy.array(vi)
vo = numpy.array(vo)

ori = []
lab = []
vi=vi.reshape(vi.shape[0],48,48,1)

model = load_model(os.path.join(os.path.dirname(__file__),'my_model.h5'))
out = model.predict(vi)

N = len(vi)

for i in range(N):
    ma = 0
    ans = -1
    for j in range(7):
        if out[i][j] > ma:
            ma = out[i][j]
            ans = j
    lab.append(ans)

for i in range(N):
    ma = 0
    ans = -1
    for j in range(7):
        if vo[i][j] > ma:
            ma = vo[i][j]
            ans = j
    ori.append(ans)

lab = numpy.array(lab)
ori = numpy.array(ori)
print (ori)
print (lab)
#os._exit(0)
import itertools
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.jet):
    """
    This function prints and plots the confusion matrix.
    """
    cm = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = numpy.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


conf_mat = confusion_matrix(ori,lab)
plt.figure()
plot_confusion_matrix(conf_mat, classes=["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"])
plt.savefig('confuse.png')
os._exit(0)