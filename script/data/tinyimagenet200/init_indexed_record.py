# create indexed record with binary labeling
import os
import mxnet as mx
import numpy as np

# read info from lst_file and images from img_dir
# compute multi-dimensional label and write into record with images
def init_indexed_record(num_class, lst_file, idx_file, rec_file, bin_label_flag=False):
    record = mx.recordio.MXIndexedRecordIO(idx_file, rec_file, 'w')

    with open(lst_file) as f:
        for line in f:
            idx, name, cls = line.strip('\n').split('\t')
            idx, cls = int(idx), float(cls)

            # generate binary label
            if bin_label_flag:
                # NOTE: We don't write binary annotation to record any more
                # We keep an numpy array in the memory
                raise NotImplementedError('No need to do this anymore')
                # aux_label = np.array([cls, 0, 0], dtype=np.uint16)  # class index, complete flag, number of edits
                # bin_label = np.ones(num_class, dtype=np.uint16)  # definitely wrong if 0 else 1 
                # label = np.concatenate((aux_label, bin_label))
            else:
                label = cls
            header = mx.recordio.IRHeader(flag=0, label=label, id=idx, id2=0)

            # read image in binary mode
            img_path = name 
            with open(img_path, 'rb') as fin:
                img = fin.read()

            # pack header and image and write into record
            s = mx.recordio.pack(header, img)
            record.write_idx(int(idx), s)

    record.close()


if __name__ == '__main__':
    # os.system('./script/data/tinyimagenet200/create_list.sh')
    
    os.chdir('./data/tinyimagenet200/')
    num_class = 200

    # training
    train_lst_file = 'train.lst'
    train_idx_file, train_rec_file = 'train.idx', 'train.rec'
    # init_indexed_record(num_class, train_lst_file, train_idx_file, train_rec_file, bin_label_flag=True)
    # NOTE: we want to separate interactive annotation with image data so that we don't have to update both of them
    init_indexed_record(num_class, train_lst_file, train_idx_file, train_rec_file, bin_label_flag=False)

    val_lst_file = 'val.lst'
    val_idx_file, val_rec_file = 'val.idx', 'val.rec'
    init_indexed_record(num_class, val_lst_file, val_idx_file, val_rec_file, bin_label_flag=False)

    test_lst_file = 'test.lst'
    test_idx_file, test_rec_file = 'test.idx', 'test.rec'
    init_indexed_record(num_class, test_lst_file, test_idx_file, test_rec_file, bin_label_flag=False)
    
    os.chdir('../../')
