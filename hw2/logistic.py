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

def sig(z):
    return 1 / (1+numpy.exp(-z))
# calculate error
def acc(ans, out):
    cnt = 0
    size = len(ans)
    for i in range(size):
        if out[i][0]==round(ans[i][0]):
            cnt += 1
    return cnt / size

# get features
def get(data):
    new = []
    for i in range(0,6):
        if i == 2 or i == 1:
            continue
        for j in range(1,19):
            if j == 2:
                continue
            data.append(math.pow(data[i],0.5*j))
        data.append(math.log(1+data[i]))
    data.append(1)
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
with open(test_csv, 'r') as fp1:
    fp1.readline()
    for i in range(16281):
        a = fp1.readline().replace('\n','').split(',')
        a = get([float(x) for x in a])
        vi.append(a)

train_in = numpy.array(train_in)
train_out = numpy.array(train_out)
vi = numpy.array(vi)

FFF = len(train_in[0])
print (FFF)

mod = numpy.array([[0.0] for i in range(FFF)])
rate = numpy.array([[1] for i in range(FFF)])
grad_t = numpy.array([[0.0] for i in range(FFF)])

div = []
std = []
for i in range(FFF):
    div.append(numpy.average(train_in[:,[i]]))
    std.append(numpy.std(train_in[:,[i]], ddof=1))
    if std[i] == 0:
        std[i] = 1
for i in range(FFF-1):
    train_in[:,[i]] = (train_in[:,[i]] - div[i] ) / std[i] + 0.0
    vi[:,[i]] = (vi[:,[i]] - div[i] ) / std[i] + 0.0

best = 0
nice = 0
idd = 0
for T in range(1599):
    Fwb = sig(train_in.dot(mod))
    grad = numpy.transpose(train_in).dot(Fwb - train_out)
    grad_t += grad * grad
    mod -= rate * grad  / (grad_t ** 0.5)
    print (T)
    
out = sig(vi.dot(mod))
with open(ans_csv, 'w') as fp2:
    fp2.write('id,label\n')
    for i in range(16281):
        fp2.write('%d,%d\n' % (i+1,round(out[i][0])))