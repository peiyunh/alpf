import random
import numpy as np
import mxnet as mx
from model import Model
from config import parse_args

if __name__ == '__main__':
    args = parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.np_random_seed)
    mx.random.seed(args.mx_random_seed)

    model = Model(args)

    # model.query()
    model.work()
