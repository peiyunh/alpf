# Active Learning with Partial Feedback 
Peiyun Hu, Zack Lipton, Anima Anandkumar, Deva Ramanan 

## Requirements 
- mxnet-cu90mkl==0.12.1 (or mxnet==0.12.1, mxnet-cu90==0.12.1)
- opencv-python
- numpy
- nltk
- tqdm

## Data preparation
Since MXNet is more efficient when data is nicely serialized, we first prepare dataset in a record file. We include scripts that convert data form the original format into record files. We automated downloading for cifar10 and cifar100. For tinyimagenet200, please download it from [ImageNet's official website](http://www.image-net.org/download-images) after logging in. Please refer to scripts under script/data for more details.

Once data preparation is done, we expect to find a `./data` with the following structure. 
```
data
├── cifar10
│   ├── cifar-10-batches-py
│   ├── test.idx
│   ├── test.lst
│   ├── test.rec
│   ├── train.idx
│   ├── train.lst
│   └── train.rec
├── cifar100
│   ├── cifar-100-python
│   ├── test.idx
│   ├── test.lst
│   ├── test.rec
│   ├── train.idx
│   ├── train.lst
│   └── train.rec
└── tinyimagenet200
    ├── test
    ├── test.idx
    ├── test.lst
    ├── test.rec
    ├── train
    ├── train.idx
    ├── train.lst
    ├── train.rec
    ├── val
    ├── val.idx
    ├── val.lst
    ├── val.rec
    ├── wnids.txt
    └── words.txt
```

## Questions construction 
We need to convert a set of class labels into a set of binary questions based on how class labels can be grouped together. Please refer to `questions.py` for how binary questions are constructed for cifar10, cifar100, and tinyimagenet200 based on wordnet. 

## Training
The code is written in a way that it can be configured for many variants of our model. However, the number of parameters and the effort of configuring can look daunting. To help with this, we include a script that shows how configuration is done for all variants of our model. Please refer to ```script/experiment/train_all_variants.py```. 

```
usage: train_all_variants.py [-h] [--dryrun] [--dataset DATASET]
                             [--gpus GPUS [GPUS ...]] [--run-id RUN_ID]

optional arguments:
  -h, --help            show this help message and exit
  --dryrun              switch for a dry run
  --dataset DATASET     [cifar10, cifar100, tinyimagenet200]
  --gpus GPUS [GPUS ...]
                        the list of gpu ids to cycle over
  --run-id RUN_ID       which cached random seed to use
```

## FAQ
### Why MXNet 0.12.1? 
I was using MXNet 0.12.1 when developing this code base. In later versions (e.g. the latest 1.3.0), there is a bug from mxnet/image/image.py that I am not sure how to fix. Until I found a way around, MXNet 0.12.1 is recommended. 

### How about MXNet 1.5.1? 
Please refer to [this pull request](https://github.com/peiyunh/alpf/pull/1/commits/5034db67aa98c8a322d8987d2ba203cef603996c) created by Abhay Mittal([@abhaymittal](https://github.com/abhaymittal)). Here is a short description: 
> 1. The division train_size / 10 on [this line](https://github.com/peiyunh/alpf/blob/master/iters.py#L85) needs to be casted to int otherwise MxNet throws an error.
> 2. In the MXNet code, I had to make a change in the [next_sample](https://github.com/apache/incubator-mxnet/blob/master/python/mxnet/image/image.py#L1460) method to use len(self.seq) instead of self.num_image
> 3. Finally, in the last batch, MxNet was adding padding by using images in the beginning of the dataset.  Therefore the value of self.cur was resetting and go backwards ([code](https://github.com/peiyunh/alpf/blob/master/iters.py#L54))( For e.g. if initially the cur was at 12200 and the whole seq length was 12300 with batch size of 200, the cursor would reset to 100 by taking 100 images from the beginning of the dataset. I solved this by storing the initial value of the cursor and then just taking the images from that point.

### Others
Write an email to me at peiyunh@cs.cmu.edu.
