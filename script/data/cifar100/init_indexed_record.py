# create indexed record with binary labeling
import os
import mxnet as mx
import numpy as np

# read info from lst_file and images from img_dir
# compute multi-dimensional label and write into record with images
# def init_indexed_record(num_class, lst_file, idx_file, rec_file, bin_label_flag=False):

# NOTE: no more support for bin_label_flag (it is inefficient, we will keep it in memory)
def init_indexed_record(num_class, lst_file, idx_file, rec_file):
    record = mx.recordio.MXIndexedRecordIO(idx_file, rec_file, 'w')

    with open(lst_file) as f:
        for line in f:
            idx, name, cls = line.strip('\n').split('\t')
            idx, cls = int(idx), int(cls)

            # generate binary label
            header = mx.recordio.IRHeader(flag=0, label=cls, id=idx, id2=0)

            # read image in binary mode
            img_path = name
            with open(img_path, 'rb') as fin:
                img = fin.read()

            # pack header and image and write into record
            s = mx.recordio.pack(header, img)
            record.write_idx(int(idx), s)

    record.close()


if __name__ == '__main__':
    os.chdir('./data/cifar100/')
    num_class = 10

    # training
    train_lst_file = 'train.lst'
    train_idx_file, train_rec_file = 'train.idx', 'train.rec'
    # init_indexed_record(num_class, train_lst_file, train_idx_file, train_rec_file, bin_label_flag=True)
    # NOTE: we want to separate interactive annotation with image data so that we don't have to update both of them
    init_indexed_record(num_class, train_lst_file, train_idx_file, train_rec_file)

    test_lst_file = 'test.lst'
    test_idx_file, test_rec_file = 'test.idx', 'test.rec'
    init_indexed_record(num_class, test_lst_file, test_idx_file, test_rec_file)

    os.chdir('../../')
