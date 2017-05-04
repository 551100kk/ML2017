import numpy
import os
import sys
from keras.models import load_model


test_csv = sys.argv[1]
ans_csv = sys.argv[2]

def histeqq(im,nbr_bins=256):

   #get image histogram
   imhist,bins = numpy.histogram(im.flatten(),nbr_bins,normed=True)
   cdf = imhist.cumsum() #cumulative distribution function
   cdf = 255 * cdf / cdf[-1] #normalize
   #use linear interpolation of cdf to find new pixel values
   im2 = numpy.interp(im.flatten(),bins[:-1],cdf)

   return im2.reshape(im.shape), cdf

test_in = []
test_out = []

test_N = 7178

with open(test_csv, 'r') as fp:
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


test_in = numpy.array(test_in)
test_in=test_in.reshape(test_in.shape[0],48,48,1)

model = load_model(os.path.join(os.path.dirname(__file__),'my_model2.h5'))
out = model.predict(test_in)

with open(ans_csv, 'w') as fp:
    fp.write('id,label\n')
    for i in range(test_N):
        
        ma = 0
        ans = -1
        for j in range(7):
            if out[i][j] > ma:
                ma = out[i][j]
                ans = j
        fp.write('%d,%d\n' % (i, ans))

os._exit(0)