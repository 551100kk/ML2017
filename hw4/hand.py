from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import svd, eigh
import math
import scipy.misc
import os
from keras.models import load_model

# 481
arr = []

for i in range(481):
	im = Image.open('hand/hand.seq%d.png' % (i+ 1))
	img = np.array(im.convert('L')).astype('float32')
	img = scipy.misc.imresize(img, (48, 50))
	arr.append(img.flatten())
	print (i)


arr = np.array(arr)
val , U = eigh(np.cov(arr.T))

N = len(val)
print (N)

X = []
for x in range(80):
	X.append([val[N - x - 1]])
Q = np.array([X]).reshape((1,80))

model = load_model(os.path.join(os.path.dirname(__file__),'my_model.h5'))
ans = round(model.predict(Q)[0][0])+ 1

print (ans)

os._exit(0)