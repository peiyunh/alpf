import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    basic = parser.add_argument_group('basic', 'basic settings')
    basic.add_argument('--log-level', type=str, choices=['info', 'debug'], default='info')
    basic.add_argument('--random-seed', type=str, default=1)
    basic.add_argument('--np-random-seed', type=str, default=1)
    basic.add_argument('--mx-random-seed', type=str, default=1)
    basic.add_argument('--gpus', type=int, nargs='+', default=[])
    basic.add_argument('--model-dir', type=str, required=True)
    basic.add_argument('--optimizer', type=str, default='sgd')
    basic.add_argument('--learning-rate', type=float, default=0.001)
    basic.add_argument('--momentum', type=float, default=0.9)
    basic.add_argument('--beta1', type=float, default=0.9)
    basic.add_argument('--beta2', type=float, default=0.999)
    basic.add_argument('--weight-decay', type=float, default=0.0001)

    network = parser.add_argument_group('network', 'network parameters')
    network.add_argument('--architecture', type=str, default='resnet18_v2')
    network.add_argument('--pretrained', action='store_true', default=False)
    network.add_argument('--thumbnail', action='store_true', default=False)

    initialization = parser.add_argument_group('initialization', 'initialization params')
    initialization.add_argument('--initializer', type=str, default='xavier')
    initialization.add_argument('--rnd-type', type=str, default='gaussian')
    initialization.add_argument('--factor-type', type=str, default='avg')
    initialization.add_argument('--magnitude', type=float, default=1.0)
    initialization.add_argument('--slope', type=float, default=0.25)
    initialization.add_argument('--sigma', type=float, default=0.01)

    dataset = parser.add_argument_group('dataset', 'dataset parameters')
    dataset.add_argument('--name', type=str, default='cifar10', required=True)
    dataset.add_argument('--train-rec', type=str, default='data/cifar10/train.rec', required=True)
    dataset.add_argument('--train-idx', type=str, default='data/cifar10/train.idx', required=True)
    dataset.add_argument('--test-rec', type=str, default='data/cifar10/test.rec', required=True)
    dataset.add_argument('--test-idx', type=str, default='data/cifar10/test.idx', required=True)
    dataset.add_argument('--data-shape', type=int, nargs='+', default=[], required=True)
    dataset.add_argument('--batch-size', type=int, default=200)
    dataset.add_argument('--shuffle', action='store_true', default=True)
    dataset.add_argument('--num-threads', type=int, default=8)

    augmentation = parser.add_argument_group('augmentation', 'augmentation parameters')
    augmentation.add_argument('--rand-crop', action='store_true', default=True)
    augmentation.add_argument('--rand-resize', action='store_true', default=False)
    augmentation.add_argument('--rand-gray', action='store_true', default=False)
    augmentation.add_argument('--rand-mirror', action='store_true', default=True)
    augmentation.add_argument('--color-jittering', action='store_true', default=False)
    augmentation.add_argument('--pca-noise', action='store_true', default=False)

    query = parser.add_argument_group('query', 'query parameters')
    query.add_argument('--feedback-type', type=str, default='full')
    # query.add_argument('--model-prior', action='store_true', default=False)
    query.add_argument('--active-instance', action='store_true', default=False)
    query.add_argument('--active-question', action='store_true', default=False)
    query.add_argument('--least-confident', action='store_true', default=False)
    query.add_argument('--question-set', type=str, default='')
    query.add_argument('--total-budget', type=int, default=500000)
    query.add_argument('--round-budget', type=int, default=10000)
    query.add_argument('--session-budget', type=int, default=0)
    query.add_argument('--score-rule', type=str, default='')
    query.add_argument('--checkpoint-cost', type=int, default=30000)

    # prelabeled
    label = parser.add_argument_group('label', 'labeling parameters')
    label.add_argument('--prelabel-ratio', type=float, default=0)

    loss = parser.add_argument_group('loss', 'loss related')
    loss.add_argument('--normalize-loss', action='store_true', default=False)

    optimization = parser.add_argument_group('optimization', 'optimization parameters')
    optimization.add_argument('--optimize-from-scratch', action='store_true', default=False)
    # optimization.add_argument('--max-optimize-epoch', type=int, default=75)
    optimization.add_argument('--max-optimize-epoch', type=int, default=75)
    optimization.add_argument('--min-optimize-epoch', type=int, default=1)
    optimization.add_argument('--reinit-round', type=int, default=75)

    # NOTE: none of these make sense any more because we will not be looking at the validation set
    # optimization.add_argument('--optimize-validate-frequency', type=int, default=1)
    # optimization.add_argument('--overfit-patience', type=int, default=5)
    # optimization.add_argument('--no-model-selection', action='store_true', default=False)

    # UP does not involve threshold
    # optimization.add_argument('--overfit-threshold', type=float, default=0.0)

    args = parser.parse_args()
    return args
