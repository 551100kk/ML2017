from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import svd, eigh
import math

arr = np.array([])

for i in range(10):
	for j in range(10):
		im = Image.open('faceExpressionDatabase/%c%02d.bmp' % (chr(i + ord('A')), j))
		img = np.array(im.convert('L')).flatten().astype('float32')
		arr = np.append(arr, img)

arr = arr.reshape(100, 64*64)
ori = np.array(arr)

mean = arr.mean(axis = 0)
arr = arr - mean


val , U = eigh(np.cov(arr.T))
U=U.T[::-1]

# problem 1.1
fig = plt.figure()
for i in range(9):
	eigface = U[i].reshape(64, 64)
	ax = fig.add_subplot(3, 3, i + 1)
	ax.imshow(eigface, cmap='gray')
	plt.xticks(np.array([]))
	plt.yticks(np.array([]))

fig.suptitle('Eigenface')
fig.savefig('eigenface.png')

plt.figure()
plt.imshow((U[:10].mean(axis=0)).reshape(64, 64), cmap='gray')
plt.suptitle('Avg Eigenface')
plt.xticks(np.array([]))
plt.yticks(np.array([]))
plt.savefig('avg_eigenface.png')

# problem 1.2
#origin
fig = plt.figure()
for i in range(10):
	for j in range(10):
		eigface = ori[i * 10 + j].reshape(64, 64)
		ax = fig.add_subplot(10, 10, i * 10 + j + 1)
		ax.imshow(eigface, cmap='gray')
		plt.xticks(np.array([]))
		plt.yticks(np.array([]))

fig.suptitle('Origin face')
fig.savefig('originface.png')

#reconstruct
re = arr.dot(U[:5].T).dot(U[:5])
fig = plt.figure()
for i in range(10):
	for j in range(10):
		eigface = (re[i * 10 + j] + mean).reshape(64, 64)
		ax = fig.add_subplot(10, 10, i * 10 + j + 1)
		ax.imshow(eigface, cmap='gray')
		plt.xticks(np.array([]))
		plt.yticks(np.array([]))

fig.suptitle('Reconstruct face')
fig.savefig('reconstruct.png')

# problem 1.3
for dim in range(1, 100):
	re = arr.dot(U[:dim].T).dot(U[:dim])
	
	loss = 0
	for i in range(100):
		loss += ((ori[i] - (re[i] + mean)) ** 2).mean()
	loss = math.sqrt(loss / 100) / 256
	print ('%d %f' % (dim, loss))

	if loss < 0.01:
		break