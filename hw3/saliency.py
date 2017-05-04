import numpy as np
from matplotlib import pyplot as plt

from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import preprocess_input

from vis.utils import utils
from vis.utils.vggnet import VGG16
from vis.visualization import visualize_saliency
from keras.models import load_model
import os
import numpy

model = load_model(os.path.join(os.path.dirname(__file__),'my_model2.h5'))
print('Model loaded.')

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

out = model.predict(test_in)

ids = [1, 5, 10, 18, 20, 21 ,28, 43, 50, 78]
heatmaps = []
mask = []

for NUM in ids:
    x = test_in[NUM]
    pred_class = numpy.argmax(out[NUM])
    heatmap = visualize_saliency(model, 0, [pred_class], test_in[NUM])
    heatmaps.append(heatmap)

    thres = 420
    see = test_in[NUM].reshape(48, 48)
    mean = numpy.mean(see)
    for i in range(48):
        for j in range(48):
            if sum(heatmap[i][j]) <= thres:
                see[i][j] = 0

    plt.figure()
    plt.imshow(see,cmap='gray')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('tmp.png')

    see = utils.load_img('tmp.png')
    mask.append(see)

    

plt.axis('off')
plt.imshow(utils.stitch_images(heatmaps))
plt.title('Saliency map')
plt.colorbar()
plt.tight_layout()
plt.savefig('line.png')


plt.figure()
plt.axis('off')
plt.imshow(utils.stitch_images(mask),cmap='gray')
plt.tight_layout()
plt.savefig('mask.png')
os._exit(0)