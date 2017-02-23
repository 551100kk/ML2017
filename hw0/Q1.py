import sys

png1 = sys.argv[1]
png2 = sys.argv[2]

A = []
with open(png1, 'r') as fp:
	A = fp.read().split('\n')
	while A[-1] == '':
		del A[-1]
	sz = len(A)
	for i in range(sz):
		A[i] = [int(val) for val in A[i].split(',')]

x = len(A)
y = len(A[0])

B = []
with open(png2, 'r') as fp:
	B = fp.read().split('\n')
	while B[-1] == '':
		del B[-1]
	sz = len(B)
	for i in range(sz):
		B[i] = [int(val) for val in B[i].split(',')]

z = len(B[0]);

C = [[0 for j in range(z)] for i in range(x)]
for i in range(x):
	for j in range(y):
		for k in range(z):
			C[i][k] += A[i][j] * B[j][k]

arr = []

for i in range(x):
	for j in range(z):
		arr.append(C[i][j])

arr.sort()
with open('ans_one.txt', 'w') as fp:
	for val in arr:
		fp.write(str(val) + '\n')