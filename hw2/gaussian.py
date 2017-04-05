import numpy
import math
import time
import random
import sys

train_X = sys.argv[1]
train_Y = sys.argv[2]
test_csv = sys.argv[3]
ans_csv = sys.argv[4]

train_in = []
train_out = []
vi = []
one = []
zero = []

def sig(z):
    return 1 / (1+numpy.exp(-z))
# calculate error
def acc(ans, out):
    cnt = 0
    size = len(ans)
    for i in range(size):
        if out[i][0]==round(ans[i]):
            cnt += 1
    return cnt / size

# get features
def get(data):
    data.append(data[0]*data[0]*data[0]/100000000000)
    return data

# read data ( total 5652 training data )
with open(train_X, 'r') as fp1, open(train_Y, 'r') as fp2:
    fp1.readline()
    for i in range(32561):
        a = fp1.readline().replace('\n','').split(',')
        b = fp2.readline().replace('\n','').split(',')
        a = get([float(x) for x in a])
        b = [int(x) for x in b]
        train_in.append(a)
        train_out.append(b)
        if b[0] == 1:
            one.append(a)
        else:
            zero.append(a)
        # print (a)
        # print (b)
with open(test_csv, 'r') as fp1:
    fp1.readline()
    for i in range(16281):
        a = fp1.readline().replace('\n','').split(',')
        a = get([float(x) for x in a])
        vi.append(a)

train_in = numpy.array(train_in)
train_out = numpy.array(train_out)
vi = numpy.array(vi)
one = numpy.array(one)
zero = numpy.array(zero)

FFF = len(train_in[0])
print (FFF)

N1 = len(one)
N2 = len(zero)
u1 = numpy.average(one.T, axis=1)
u2 = numpy.average(zero.T, axis=1)
E1 = numpy.cov(one.T)
E2 = numpy.cov(zero.T)
E = E1 * (1.0 * N1 / (N1 + N2)) + E2 * (1.0 * N2 / (N1 + N2))
EI = numpy.linalg.inv(E)

w = (u1 - u2).T.dot(EI)
b = -u1.T.dot(EI).dot(u1)/2+u2.T.dot(EI).dot(u2)/2+math.log(N1)-math.log(N2)

tmp = train_in.dot(w)+b
Fwb = sig(tmp)
error = acc(Fwb, train_out)
print (error)

out = sig(vi.dot(w)+b)
with open(ans_csv, 'w') as fp2:
    fp2.write('id,label\n')
    for i in range(16281):
        fp2.write('%d,%d\n' % (i+1,round(out[i])))