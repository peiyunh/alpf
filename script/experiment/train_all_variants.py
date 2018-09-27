from __future__ import print_function
import itertools
import subprocess
import numpy as np
import argparse
import sys

processes = []
pid = 0

parser = argparse.ArgumentParser()
parser.add_argument('--dryrun', action='store_true', help='switch for a dry run')
parser.add_argument('--dataset', type=str, default='cifar10', help='[cifar10, cifar100, tinyimagenet200]')
parser.add_argument('--gpus', type=int, nargs='+', default=[0], help='the list of gpu ids to cycle over')
parser.add_argument('--run-id', type=int, default=0, help='which cached random seed to use')
cml_args = parser.parse_args()
if len(cml_args.gpus) == 0:
    raise Exception('Did you forget to specify a list of gpus?')

dataset = cml_args.dataset
gpus = itertools.cycle(cml_args.gpus)

# NOTE: these seeds are randomly generated
# we keep a copy them for reproducibility
#
# eventually, we average over runs with different seeds
seeds = [12489, 50162, 5283, 56281, 63314]
npseeds = [23285, 356, 21365, 12957, 5683]
mxseeds = [3485, 2357, 501, 7684, 43861]

run_id = cml_args.run_id
if run_id == 0:
    seed = 12489
    npseed = 3485
    mxseed = 23285
    print('Using default random seeds')
else:
    seed = seeds[run_id]
    npseed = npseeds[run_id]
    mxseed = mxseeds[run_id]
    print('Using random seeds python(%d) numpy(%d) mxnet(%d)' % (seed, npseed, mxseed))

if dataset == 'tinyimagenet200':
    N = 100000
    K = 200
    tb = 1000000000  # unlimited (until fully labeled)
    rb = 30000
    img_size = 64
    test_rec = 'data/tinyimagenet200/val.rec'
    test_idx = 'data/tinyimagenet200/val.idx'
elif dataset == 'cifar100':
    N = 50000
    K = 100
    tb = 1000000000  # unlimited (until fully labeled)
    rb = 15000
    img_size = 32
    test_rec = 'data/cifar100/test.rec'
    test_idx = 'data/cifar100/test.idx'
elif dataset == 'cifar10':
    N = 50000
    K = 10
    tb = 1000000000  # unlimited (until fully labeled)
    rb = 5000
    img_size = 32
    test_rec = 'data/cifar10/test.rec'
    test_idx = 'data/cifar10/test.idx'
else:
    raise Error('Unknown dataset %s' % dataset)

for feedback_type in ['full', 'partial']:
    param_format = '--feedback-type {} --score-rule {} --total-budget {} ' + \
        '--round-budget {} --session-budget {} --prelabel-ratio {}'

    if feedback_type == 'full':
        active_types = [(False, False), (False, True), (True, False)]
    elif feedback_type == 'partial':
        active_types = [(True, True)]
    else:
        raise Error('Unknown feedback type: %s' % feedback_type)

    for active_instance, active_question in active_types:
        if active_instance == False and active_question == False:
            names = ['full_passive']
            params = [param_format.format(feedback_type, 'MERE', tb, rb, 1, 0.05)]
        elif active_instance == True and active_question == False:
            names = ['full_active_ME', 'full_active_LC']
            params = [param_format.format(feedback_type, 'MERE', tb, rb, 1, 0.05),
                      param_format.format(feedback_type, 'MERE', tb, rb, 1, 0.05)]
        elif active_instance == False and active_question == True:
            names = ['full_active_EIG', 'full_active_ERC', 'full_active_EDRC']
            params = [param_format.format(feedback_type, 'MERE', tb, rb, 1, 0.05),
                      param_format.format(feedback_type, 'MinENRC', tb, rb, 1, 0.05),
                      param_format.format(feedback_type, 'MERNRC', tb, rb, 1, 0.05)]
        elif active_instance == True and active_question == True:
            names = ['partial_active_EIG', 'partial_active_ERC', 'partial_active_EDRC']
            params = [param_format.format(feedback_type, 'MERE', tb, rb, 1, 0.05),
                      param_format.format(feedback_type, 'MinENRC', tb, rb, 1, 0.05),
                      param_format.format(feedback_type, 'MERNRC', tb, rb, 1, 0.05)]

        for name, param in zip(names, params):

            if active_instance:
                param += ' --active-instance'

            if active_question:
                param += ' --active-question'

            if name == 'full_active_LC':
                param += ' --least-confident'

            seed_dir = '_seeds_%d_%d_%d' % (seed, npseed, mxseed) if run_id > 0 else ''
            seed_param = '--random-seed %d --np-random-seed %d --mx-random-seed %d' % (seed, npseed, mxseed) if run_id > 0 else ''

            exe = 'python'
            script = 'main.py'
            args = ('--name %s --architecture resnet18_v2' % dataset + ' '
                    '--gpus %d --num-threads 4' % gpus.next() + ' '
                    '--data-shape 3 %d %d --initializer xavier --batch-size 200' % (img_size, img_size) + ' '
                    '--optimizer adam --learning-rate 0.001' + ' '
                    '--rand-crop --rand-resize --rand-mirror' + ' '
                    '--model-dir model/%s_partial_learning_with_prelabels_noval_%.2f%s/{}/' % (dataset, 0.05, seed_dir) + ' '
                    '%s' % param + ' '
                    '--question-set wordnet' + ' '
                    '%s' % seed_param + ' '
                    '--optimize-from-scratch --max-optimize-epoch 75' + ' '
                    '--train-rec data/%s/train.rec' % dataset + ' '
                    '--train-idx data/%s/train.idx' % dataset + ' '
                    '--test-rec %s' % test_rec + ' '
                    '--test-idx %s' % test_idx)


            model_dir = name
            if active_instance:
                model_dir += '_active_instance'
            else:
                model_dir += '_inactive_instance'

            if active_question:
                model_dir += '_active_question'
            else:
                model_dir += '_inactive_question'

            args = args.format(model_dir)
            print(exe + ' ' + script + ' ' + args)

            args = args.split()
            if not cml_args.dryrun:
                processes.append(subprocess.Popen([exe, script] + args))
            pid += 1


for p in processes:
    p.wait()
