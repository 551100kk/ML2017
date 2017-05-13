import numpy as np
from numpy.linalg import matrix_rank, eigh

import math
import sys
def elu(arr):
    return np.where(arr > 0, arr, np.exp(arr) - 1)


def make_layer(in_size, out_size):
    w = np.random.normal(scale=0.5, size=(in_size, out_size))
    b = np.random.normal(scale=0.5, size=out_size)
    return (w, b)


def forward(inpd, layers):
    out = inpd
    for layer in layers:
        w, b = layer
        out = elu(out @ w + b)

    return out


def gen_data(dim, layer_dims, N):
    layers = []
    data = np.random.normal(size=(N, dim))

    nd = dim
    for d in layer_dims:
        layers.append(make_layer(nd, d))
        nd = d

    w, b = make_layer(nd, nd)
    gen_data = forward(data, layers)
    gen_data = gen_data @ w + b
    return gen_data

train_in = []
train_out = []
train_label = []

if __name__ == '__main__':
    # if we want to generate data with intrinsic dimension of 10
    
    for T in range(2000): 
        for i in range(60):
            dim = i + 1
            N = 10000
            layer_dims = [np.random.randint(60, 80), 100]
            data = gen_data(dim, layer_dims, N)
            val , mat = eigh(np.cov(data.transpose()))

            tmp_in = []
            tmp_out = []

            for x in range(80):
                tmp_in.append(val[99 - x])

            for x in range(60):
                tmp_out.append(0)
            tmp_out[i] = 1

            train_in.append(tmp_in)
            train_out.append(tmp_out)
            train_label.append([i])
                
            print ((T, i))
    
    train_in = np.array(train_in)
    train_out = np.array(train_out)
    train_label = np.array(train_label)
    
    np.save('train_in.npy', train_in)
    np.save('train_out.npy', train_out)
    np.save('train_label.npy', train_label)

    exit(0)