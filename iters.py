import os
import mxnet as mx
from mxnet import ndarray as nd

class MultiSequenceImageIter(mx.image.ImageIter):
    def __init__(self, path_imgrec, path_imgidx, data_shape, batch_size, shuffle, aug_list, logger):
        self.logger = logger
        super(MultiSequenceImageIter, self).__init__(path_imgrec=path_imgrec, path_imgidx=path_imgidx,
                                                     data_shape=data_shape, batch_size=batch_size,
                                                     shuffle=shuffle, aug_list=aug_list)
        self.seqs, self.curs, self.names, self.name = {}, {}, set(), None

    def add(self, name, seq, cur):
        self.names.add(name)
        self.seqs[name] = seq
        self.curs[name] = cur

    def switch(self, name):
        self.logger.debug('Switching to [%s] sequence' % name)
        assert(name in self.names)
        if self.name is not None:
            self.logger.debug('Saving the current[%s] sequence[%d] and cursor[%d]' % (self.name, len(self.seq), self.cur))
            self.seqs[self.name] = self.seq
            self.curs[self.name] = self.cur
        self.logger.debug('Setting the sequence[%d] and cursor[%d] of [%s] as current' % (len(self.seqs[name]), self.curs[name], name))
        self.seq = self.seqs[name]
        self.cur = self.curs[name]
        self.name = name

    def append_sequence(self, seq, name=None):
        if name is None:
            self.logger.debug('Appending a subsequence[%d] to the current[%s] sequence' % (len(seq), self.name))
            for i in seq:
                if i not in self.seq:
                    self.seq.append(i)
            # self.seq.extend(seq)
        elif name in self.names:
            self.logger.debug('Appending a subsequence[%d] to the sequence[%s]' % (len(seq), name))
            for i in seq:
                if i not in self.seqs[name]:
                    self.seqs[name].append(i)
            # self.seqs[name].extend(seq)

    def set_cursor(self, cur):
        self.logger.debug('Setting the current[%s] cursor[%d] to %d' % (self.name, self.cur, cur))
        self.cur = cur

    def next(self):
        self.logger.debug('Grabbing a batch[%d] from [%s]\'s sequence starting at cursor[%d]' % (self.batch_size, self.name, self.cur))
        batch = super(MultiSequenceImageIter, self).next()
        batch.data = [batch.data[0][:self.batch_size-batch.pad]]
        batch.label = [batch.label[0][:self.batch_size-batch.pad]]
        if self.seq is not None:
            batch.index = [self.seq[self.cur-(self.batch_size-batch.pad):self.cur]]
        return batch

    def reset(self):
        self.logger.debug('Resetting the current sequence and cursor')
        super(MultiSequenceImageIter, self).reset()

def get_train_aug(data_shape, mean_rgb, std_rgb, args):
    train_aug = mx.image.CreateAugmenter(data_shape=data_shape,
                                         mean=mean_rgb, std=std_rgb,
                                         rand_crop=args.rand_crop,
                                         rand_resize=args.rand_resize,
                                         rand_gray=0.05 if args.rand_gray else 0.0,
                                         rand_mirror=args.rand_mirror,
                                         brightness=0.125 if args.color_jittering else None,
                                         contrast=0.125 if args.color_jittering else None,
                                         saturation=0.125 if args.color_jittering else None,
                                         pca_noise=0.05 if args.pca_noise else 0)
    return train_aug

def cifar10_iterator(args, logger):
    assert(args.name=='cifar10')
    os.environ['MXNET_CPU_WORKER_NTHREADS'] = '%d' % args.num_threads

    train_size = 50000
    classes = 10

    data_shape = (3,32,32) if len(args.data_shape)==0 else tuple(args.data_shape)
    batch_size = args.batch_size

    # compute mean and std based on 10% of the training data
    stats_iter = mx.image.ImageIter(train_size/10, data_shape, path_imgrec=args.train_rec, path_imgidx=args.train_idx, shuffle=True)
    sample = stats_iter.next().data[0].transpose(axes=(1,0,2,3)).reshape((3,-1))
    mean_rgb = sample.mean(axis=-1,keepdims=True)
    std_rgb = nd.sqrt(nd.mean(nd.square(sample - mean_rgb), axis=-1, keepdims=True))
    mean_rgb, std_rgb = mean_rgb.reshape((-1,)).asnumpy(), std_rgb.reshape((-1,)).asnumpy()

    train_aug = get_train_aug(data_shape, mean_rgb, std_rgb, args)
    train_iter = MultiSequenceImageIter(args.train_rec, args.train_idx, data_shape=data_shape, batch_size=batch_size, shuffle=True, aug_list=train_aug, logger=logger)

    val_aug = mx.image.CreateAugmenter(data_shape=data_shape, mean=mean_rgb, std=std_rgb)
    val_iter = mx.image.ImageIter(batch_size, data_shape, path_imgrec=args.test_rec, path_imgidx=args.test_idx, shuffle=False, aug_list=val_aug)

    return train_iter, val_iter, classes, train_size

def cifar100_iterator(args, logger):
    assert(args.name=='cifar100')
    os.environ['MXNET_CPU_WORKER_NTHREADS'] = '%d' % args.num_threads

    train_size = 50000
    classes = 100

    data_shape = (3,32,32) if len(args.data_shape)==0 else tuple(args.data_shape)
    batch_size = args.batch_size

    # compute mean and std based on 10% of the training data
    stats_iter = mx.image.ImageIter(train_size/10, data_shape, path_imgrec=args.train_rec, path_imgidx=args.train_idx, shuffle=True)
    sample = stats_iter.next().data[0].transpose(axes=(1,0,2,3)).reshape((3,-1))
    mean_rgb = sample.mean(axis=-1,keepdims=True)
    std_rgb = nd.sqrt(nd.mean(nd.square(sample - mean_rgb), axis=-1, keepdims=True))
    mean_rgb, std_rgb = mean_rgb.reshape((-1,)).asnumpy(), std_rgb.reshape((-1,)).asnumpy()

    train_aug = get_train_aug(data_shape, mean_rgb, std_rgb, args)
    train_iter = MultiSequenceImageIter(args.train_rec, args.train_idx, data_shape=data_shape, batch_size=batch_size, shuffle=True, aug_list=train_aug, logger=logger)

    val_aug = mx.image.CreateAugmenter(data_shape=data_shape, mean=mean_rgb, std=std_rgb)
    # val_aug = mx.image.CreateAugmenter(data_shape=data_shape, mean=True, std=True)
    val_iter = mx.image.ImageIter(batch_size, data_shape, path_imgrec=args.test_rec, path_imgidx=args.test_idx, shuffle=False, aug_list=val_aug)

    return train_iter, val_iter, classes, train_size

def tinyimagenet200_iterator(args, logger):
    assert(args.name=='tinyimagenet200')
    os.environ['MXNET_CPU_WORKER_NTHREADS'] = '%d' % args.num_threads

    train_size = 100000
    classes = 200

    data_shape = (3,64,64) if len(args.data_shape)==0 else tuple(args.data_shape)
    batch_size = args.batch_size

    # compute mean and std based on 10% of the training data
    stats_iter = mx.image.ImageIter(train_size/10, data_shape, path_imgrec=args.train_rec, path_imgidx=args.train_idx, shuffle=True)
    sample = stats_iter.next().data[0].transpose(axes=(1,0,2,3)).reshape((3,-1))
    mean_rgb = sample.mean(axis=-1,keepdims=True)
    std_rgb = nd.sqrt(nd.mean(nd.square(sample - mean_rgb), axis=-1, keepdims=True))
    mean_rgb, std_rgb = mean_rgb.reshape((-1,)).asnumpy(), std_rgb.reshape((-1,)).asnumpy()

    train_aug = get_train_aug(data_shape, mean_rgb, std_rgb, args)
    train_iter = MultiSequenceImageIter(args.train_rec, args.train_idx, data_shape=data_shape, batch_size=batch_size, shuffle=True, aug_list=train_aug, logger=logger)

    val_aug = mx.image.CreateAugmenter(data_shape=data_shape, mean=mean_rgb, std=std_rgb)
    # val_aug = mx.image.CreateAugmenter(data_shape=data_shape, mean=True, std=True)
    val_iter = mx.image.ImageIter(batch_size, data_shape, path_imgrec=args.test_rec, path_imgidx=args.test_idx, shuffle=False, aug_list=val_aug)
    # train_aug = get_train_aug(data_shape, args)
    # train_iter = MultiSequenceImageIter(args.train_rec, args.train_idx, data_shape, batch_size, shuffle=True, aug_list=train_aug, logger=logger)

    # val_aug = mx.image.CreateAugmenter(data_shape=data_shape, mean=True, std=True)
    # val_iter = MultiSequenceImageIter(args.test_rec, args.test_idx, data_shape, batch_size, shuffle=False, aug_list=val_aug, logger=logger)

    return train_iter, val_iter, classes, train_size
