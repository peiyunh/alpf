# credit: Julien Simon
# https://gist.github.com/juliensimon/273bef4c5b4490c687b2f92ee721b546

import os
import mxnet as mx
import numpy as np
import cPickle
import cv2
from tqdm import trange

def extractImagesAndLabels(path, file):
    f = open(path+file, 'rb')
    dict = cPickle.load(f)
    images = dict['data']
    images = np.reshape(images, (10000, 3, 32, 32))
    labels = dict['labels']
    namearray = dict['filenames']
    imagearray = mx.nd.array(images)
    labelarray = mx.nd.array(labels)
    return imagearray, labelarray, namearray 

def extractCategories(path, file):
    f = open(path+file, 'rb')
    dict = cPickle.load(f)
    return dict['label_names']

def saveCifarImage(array, path, file):
    # array is 3x32x32. cv2 needs 32x32x3
    array = array.asnumpy().transpose(1,2,0)
    # array is RGB. cv2 needs BGR
    array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    # save to PNG file
    return cv2.imwrite(path+file, array, [cv2.IMWRITE_PNG_COMPRESSION, 0])


if not os.path.exists('data/cifar10'):
    os.makedirs('data/cifar10')
os.chdir('data/cifar10')

if not os.path.exists('cifar-10-python.tar.gz'):
    os.system('wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz')
    os.system('tar zxvf cifar-10-python.tar.gz') 

if not os.path.exists('images'): 
    os.makedirs('images')

# save png and lst file
with open('train.lst', 'w') as f:
    for batch_id in trange(5):
        imgarray, lblarray, namearray = extractImagesAndLabels('cifar-10-batches-py/', 'data_batch_%d' % (batch_id+1))
        for i in trange(imgarray.shape[0]):
            idx = batch_id * imgarray.shape[0] + i
            lbl = int(lblarray[i].asnumpy()[0])

            # save png 
            saveCifarImage(imgarray[i], './images/', namearray[i])

            # write lst
            f.write('%d\t%s\t%d\n' % (idx, './images/%s' % namearray[i], lbl))

with open('test.lst', 'w') as f:
    imgarray, lblarray, namearray = extractImagesAndLabels('cifar-10-batches-py/', 'test_batch') 
    for i in trange(imgarray.shape[0]):
        idx = i
        lbl = int(lblarray[i].asnumpy()[0])

        # save png 
        saveCifarImage(imgarray[i], './images/', namearray[i])

        # write lst
        f.write('%d\t%s\t%d\n' % (idx, './images/%s' % namearray[i], lbl))

os.chdir('../../')
