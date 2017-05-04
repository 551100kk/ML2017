from keras import backend as K
from keras.models import load_model
import os
import numpy
from matplotlib import pyplot as plt

#read
test_in = []
test_N = 101

with open('test.csv', 'r') as fp:
    fp.readline()
    for i in range(test_N):
        a = fp.readline().replace('\n','').split(',')
        label = int(a[0])
        feature = a[1].split(' ')
        feature = [int(x) for x in feature]
        test_in.append(feature)

test_in = numpy.array(test_in)
test_in=test_in.reshape(test_in.shape[0],48,48,1)

# start
# input_img = test_in[0]
filter_index = 0  # can be any integer from 0 to 511, as there are 512 filters in that layer

# build a loss function that maximizes the activation
# of the nth filter of the layer considered

fig = plt.figure(figsize=(10,3)) # 大小可自行決定

model = load_model('my_model2.h5')
input_img = model.input
con_model = K.function([input_img, K.learning_phase()], [model.layers[0].output])

layer_output = model.layers[0].output


NUM = 79

for filter_index in range(32):

    loss = K.mean(layer_output[:, :, :, filter_index])
    grads = K.gradients(loss, input_img)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    iterate = K.function([input_img], [loss, grads])
    output = numpy.random.random((1, 48, 48, 1)) 

    '''for i in range(200):
        loss_value, grads_value = iterate([output])
        output += grads_value

    output = output.reshape(48,48)'''

    tmp = test_in[NUM].reshape((1,48, 48, 1)) 
    output = con_model([tmp, 0])[0]

    ax = fig.add_subplot(32/16,16,filter_index + 1)
    ax.imshow(output[0, :, :, filter_index],cmap='Greens')
    plt.xticks(numpy.array([]))
    plt.yticks(numpy.array([]))

# fig.suptitle('Filters of layer %s (# Ascent Epoch 200 )' % (model.layers[0].name))
fig.suptitle('Output of layer%s (Given image %d)' % (model.layers[0].name, NUM))
fig.savefig('layer.png')


os._exit(0)