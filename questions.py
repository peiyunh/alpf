import numpy as np
import cPickle as pickle

# CIFAR 10: all off-the-shelf questions (not presented in the paper)
def cifar10_questions():
    # Q = np.diag(np.ones(10, dtype=np.bool))
    Q = np.diag(np.ones(10, dtype=np.int))
    return Q

# CIFAR 10: all possible questions (not presented in the paper)
def cifar10_all_questions():
    K = 10
    # Q = np.zeros((2**K, K), dtype=np.bool)
    Q = np.zeros((2**K, K), dtype=np.int)
    for i in range(0, 2**K):
        num = i
        for j in range(K)[::-1]:
            if num >= 2**j:
                Q[i][j] = 1
                num -= 2**j
    # it is dumb to ask a question that involves no none or all choices
    Q = Q[1:-1]
    return Q

# CIFAR 10: all questions derived from wordnet (presented in the paper)
def cifar10_questions_wordnet():
    K = 10
    # Q = np.diag(np.ones(K, dtype=np.bool))
    Q = np.diag(np.ones(K, dtype=np.int))

    with open('data/cifar10/cifar-10-batches-py/batches.meta', 'rb') as f:
        meta = pickle.load(f)

    labels = {}
    clubs = {}
    from nltk.corpus import wordnet
    for label_index, label_name in enumerate(meta['label_names']):
        labels[label_name] = label_index
        synsets = wordnet.synsets(label_name, pos='n')
        for synset in synsets:
            paths = synset.hypernym_paths()
            for path in paths:
                for hypernym in path:
                    hypernym_name = hypernym.name()
                    if hypernym_name not in clubs:
                        clubs[hypernym_name] = set()
                    clubs[hypernym_name].add(label_name)

    # Q2 = np.zeros((len(clubs), K), dtype=np.bool)
    Q2 = np.zeros((len(clubs), K), dtype=np.int)
    hypernym_names = []
    for i, (hypernym_name, label_names) in enumerate(clubs.items()):
        hypernym_names.append(hypernym_name)
        for label_name in label_names:
            label_index = labels[label_name]
            Q2[i][label_index] = True

    # prune Q2
    I = np.where(np.logical_and(Q2.sum(axis=-1)>1, Q2.sum(axis=-1)<K))
    Q2 = Q2[I[0]]
    hypernym_names = [hypernym_names[i] for i in I[0]]

    Q = np.concatenate((Q, Q2))
    clubs = meta['label_names'] + hypernym_names

    # NOTE: we do sometimes get weird clubs like (bird, cat, dog, frog)
    # and their common ancestor is person
    # this is because they can all be used to describe a person (oh well)

    return Q

# CIFAR 100: all off-the-shelf questions (not presented in the paper)
#   these questions are derived from CIFAR100's two layer label tree
def cifar100_questions():
    f = open('data/cifar100/cifar-100-python/train', 'rb')
    data = pickle.load(f)
    coarse_labels = data['coarse_labels']
    fine_labels = data['fine_labels']
    assert(len(coarse_labels) == len(fine_labels))

    K1 = len(np.unique(coarse_labels))
    K2 = len(np.unique(fine_labels))

    # Q = np.zeros((K1+K2, K2), dtype=np.bool)
    Q = np.zeros((K1+K2, K2), dtype=np.int)
    for i in range(len(coarse_labels)):
        cl, fl = coarse_labels[i], fine_labels[i]
        Q[fl][fl] = True
        Q[K2+cl][fl] = True
    return Q

# CIFAR 100: all questions derived from wordnet (presented in the paper)
def cifar100_questions_wordnet():
    with open('data/cifar100/cifar-100-python/train', 'rb') as f:
        data = pickle.load(f)

    coarse_labels = data['coarse_labels']
    fine_labels = data['fine_labels']

    assert(len(coarse_labels) == len(fine_labels))

    K1 = len(np.unique(coarse_labels))
    K2 = len(np.unique(fine_labels))

    # Q = np.zeros((K1+K2, K2), dtype=np.bool)
    Q = np.zeros((K1+K2, K2), dtype=np.int)
    for i in range(len(coarse_labels)):
        cl, fl = coarse_labels[i], fine_labels[i]
        Q[fl][fl] = True
        Q[K2+cl][fl] = True

    # add questions constructed based on wordnet
    with open('data/cifar100/cifar-100-python/meta', 'rb') as f:
        meta = pickle.load(f)

    labels = {}
    clubs = {}
    from nltk.corpus import wordnet
    for fine_label_index, fine_label_name in enumerate(meta['fine_label_names']):
        labels[fine_label_name] = fine_label_index
        synsets = wordnet.synsets(fine_label_name, pos='n')
        for synset in synsets:
            paths = synset.hypernym_paths()
            for path in paths:
                for hypernym in path:
                    hypernym_name = hypernym.name()
                    if hypernym_name not in clubs:
                        clubs[hypernym_name] = set()
                    clubs[hypernym_name].add(fine_label_name)

    # Q2 = np.zeros((len(clubs), K2), dtype=np.bool)
    Q2 = np.zeros((len(clubs), K2), dtype=np.int)
    hypernym_names = []
    for i, (hypernym_name, fine_label_names) in enumerate(clubs.items()):
        hypernym_names.append(hypernym_name)
        for fine_label_name in fine_label_names:
            label_index = labels[fine_label_name]
            Q2[i][label_index] = True

    # prune Q2
    I = np.where(np.logical_and(Q2.sum(axis=-1)>1, Q2.sum(axis=-1)<K2))
    Q2 = Q2[I[0]]
    hypernym_names = [hypernym_names[i] for i in I[0]]

    Q = np.concatenate((Q, Q2))
    clubs = meta['fine_label_names'] + meta['coarse_label_names'] + hypernym_names

    assert(Q.shape[0] == len(clubs))

    return Q

# Tiny ImageNet: all questions derived from wordnet (presented in the paper)
def tinyimagenet200_questions_wordnet(return_clubs=False):
    K = 200
    # Q = np.diag(np.ones(K, dtype=np.bool))
    Q = np.diag(np.ones(K, dtype=np.int))

    from nltk.corpus import wordnet
    with open('data/tinyimagenet200/wnids.txt') as f:
        wnids = [l.split()[0] for l in f]
    synsets = [wordnet.synset_from_pos_and_offset(wnid[0], int(wnid[1:])) for wnid in wnids]
    meta = {'label_names': [synset.name() for synset in synsets]}

    labels = {synset.name(): i for i, synset in enumerate(synsets)}
    clubs = {}
    for synset in synsets:
        paths = synset.hypernym_paths()
        for path in paths:
            for hypernym in path:
                hypernym_name = hypernym.name()
                if hypernym_name not in clubs:
                    clubs[hypernym_name] = set()
                clubs[hypernym_name].add(synset.name())

    # Q2 = np.zeros((len(clubs), K), dtype=np.bool)
    Q2 = np.zeros((len(clubs), K), dtype=np.int)
    hypernym_names = []
    for i, (hypernym_name, fine_label_names) in enumerate(clubs.items()):
        hypernym_names.append(hypernym_name)
        for fine_label_name in fine_label_names:
            label_index = labels[fine_label_name]
            Q2[i][label_index] = True

    # prune Q2
    I = np.where(np.logical_and(Q2.sum(axis=-1)>1, Q2.sum(axis=-1)<K))
    Q2 = Q2[I[0]]
    hypernym_names = [hypernym_names[i] for i in I[0]]

    Q = np.concatenate((Q, Q2))
    clubs = meta['label_names'] + hypernym_names

    assert(Q.shape[0] == len(clubs))

    if not return_clubs:
        return Q
    else:
        return Q, clubs
