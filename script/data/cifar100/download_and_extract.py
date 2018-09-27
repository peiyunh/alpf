# credit: Julien Simon
# https://gist.github.com/juliensimon/273bef4c5b4490c687b2f92ee721b546

import os
import mxnet as mx
import numpy as np
import cPickle
import cv2
from tqdm import trange

def extractImagesAndLabels(path):
    f = open(path, 'rb')
    dict = cPickle.load(f)
    images = dict['data']
    coarse_labels = dict['coarse_labels']
    fine_labels = dict['fine_labels']
    name_array = dict['filenames']
    images = np.reshape(images, (-1, 3, 32, 32))
    image_array = mx.nd.array(images)
    coarse_label_array = coarse_labels
    fine_label_array = fine_labels
    return image_array, coarse_label_array, fine_label_array, name_array

def extractCategories(path):
    f = open(path, 'rb')
    dict = cPickle.load(f)
    return dict['label_names']

def saveCifarImage(array, path):
    # array is 3x32x32. cv2 needs 32x32x3
    array = array.asnumpy().transpose(1,2,0)
    # array is RGB. cv2 needs BGR
    array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    # save to PNG file
    return cv2.imwrite(path, array, [cv2.IMWRITE_PNG_COMPRESSION, 0])


if not os.path.exists('data/cifar100'):
    os.makedirs('data/cifar100')
os.chdir('data/cifar100')

if not os.path.exists('cifar-100-python.tar.gz'):
    os.system('wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz')
    os.system('tar zxvf cifar-100-python.tar.gz')

if not os.path.exists('images'):
    os.makedirs('images')

# save png and lst file
with open('train.lst', 'w') as f:
    image_array, coarse_label_array, fine_label_array, name_array = extractImagesAndLabels('cifar-100-python/train')
    for i in trange(image_array.shape[0]):
        saveCifarImage(image_array[i], './images/%s' % name_array[i])
        # ONLY USE FINE LABELS
        f.write('%d\t%s\t%d\n' % (i, './images/%s' % name_array[i], fine_label_array[i]))

with open('test.lst', 'w') as f:
    image_array, coarse_label_array, fine_label_array, name_array = extractImagesAndLabels('cifar-100-python/test')
    for i in trange(image_array.shape[0]):
        saveCifarImage(image_array[i], './images/%s' % name_array[i])
        # ONLY USE FINE LABELS
        f.write('%d\t%s\t%d\n' % (i, './images/%s' % name_array[i], fine_label_array[i]))

os.chdir('../../')
