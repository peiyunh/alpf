import os
import random
import logging
import datetime
import mxnet as mx
import numpy as np

from mxnet import ndarray as nd
from mxnet.gluon.utils import split_and_load

from init import init_network, init_logger, reinit_network
from questions import *

from heapq import *


def loss_func(o, na):
    pr = nd.softmax(o, axis=-1)
    ls = -nd.log(1e-12 + nd.sum(pr * na, axis=-1))
    return ls


def unpack_batch(batch, ctx_list):
    data = split_and_load(
        batch.data[0], ctx_list=ctx_list, batch_axis=0, even_split=False)
    label = split_and_load(
        batch.label[0], ctx_list=ctx_list, batch_axis=0, even_split=False)
    if batch.index is None:
        return data, label
    else:
        return data, label, batch.index[0]


def net_copy(net_src, net_dst, ctxs, logger):
    logger.debug('Copying network %s to network %s' %
                 (net_src.name, net_dst.name))
    params_src, params_dst = net_src.collect_params(), net_dst.collect_params()
    for (name_src, name_dst) in zip(params_src, params_dst):
        for ctx in ctxs:
            params_dst[name_dst].set_data(
                params_src[name_src].data(ctx).copy())


class Model():
    def __init__(self, args):
        self.args = args
        self.model_dir = args.model_dir
        self.param_dir = os.path.join(self.model_dir, 'params')
        self.tag = datetime.datetime.now().strftime('training_%Y_%m_%d_%H_%M_%S')
        self.log_dir = os.path.join(self.model_dir, 'logs')
        self.log_file = os.path.join(self.log_dir, self.tag+'.log')
        self.anno_dir = os.path.join(self.model_dir, 'annotations')
        self.book_file = os.path.join(self.anno_dir, self.tag+'.csv')
        for d in (self.model_dir, self.log_dir, self.param_dir, self.anno_dir):
            if not os.path.exists(d): os.makedirs(d)
        self.logger = init_logger(self.log_file, args.log_level)

        self.context = [mx.gpu(i) for i in args.gpus] if len(
            args.gpus) > 0 else [mx.cpu()]

        self.dataset = args.name
        if self.dataset == 'cifar10':
            from iters import cifar10_iterator
            self.train_iter, self.val_iter, self.K, self.N = cifar10_iterator(
                args, self.logger)
            if args.question_set == 'wordnet':
                self.Q = cifar10_questions_wordnet()
            else:
                self.Q = cifar10_questions()
        elif self.dataset == 'cifar100':
            from iters import cifar100_iterator
            self.train_iter, self.val_iter, self.K, self.N = cifar100_iterator(
                args, self.logger)
            from questions import cifar100_questions
            if args.question_set == 'wordnet':
                self.Q = cifar100_questions_wordnet()
            else:
                self.Q = cifar100_questions()
        elif self.dataset == 'tinyimagenet200':
            from iters import tinyimagenet200_iterator
            self.train_iter, self.val_iter, self.K, self.N = tinyimagenet200_iterator(
                args, self.logger)
            if args.question_set == 'wordnet':
                self.Q = tinyimagenet200_questions_wordnet()
            else:
                raise Error('TinyImageNet has to use wordnet as hierarchy')
        else:
            raise NotImplementedError('%s is not supported' % self.dataset)

        self.train_iter.add('query', self.train_iter.seq, self.train_iter.cur)
        self.train_iter.add('optimization', [], 0)

        self.net = init_network(self.K, self.context, self.args)
        self.cnet = init_network(self.K, self.context, self.args)
        # run one batch to make sure it initializes
        self.net(nd.ones(self.train_iter.data_shape,
                 self.context[0]).expand_dims(0))
        self.cnet(nd.ones(self.train_iter.data_shape,
                  self.context[0]).expand_dims(0))

        # NOTE: if gpu0 runs out of memory in multigpu case, initialize trainier with device='local'
        # otherwise all resources will be put on gpu 0
        # https://github.com/apache/incubator-mxnet/blob/master/python/mxnet/gluon/trainer.py#L54
        if args.optimizer == 'sgd':
            optargs = {'learning_rate': args.learning_rate,
                'momentum': args.momentum, 'wd': args.weight_decay}
        elif args.optimizer == 'adam':
            optargs = {'learning_rate': args.learning_rate,
                'beta1': args.beta1, 'beta2': args.beta2}

        # NOTE: added on 05/15 to prevent gradient explosion
        # HOPEFULLY this addresses the unstable problem.
        optargs['clip_gradient'] = 2.0

        self.trainer = mx.gluon.Trainer(
            self.net.collect_params(), args.optimizer, optargs)
        self.train_annotation = np.ones((self.N, self.K), dtype=np.bool)
        self.train_annotation_sum = np.full((self.N), self.K, dtype=np.float)
        self.train_prior = np.full((self.N, self.K), 1.0/self.K)
        self.train_label = np.zeros(self.N, dtype=np.int)
        self.train_iter.switch('query')
        self.train_iter.reset()
        for batch in self.train_iter:
            index, label = batch.index[0], batch.label[0]
            self.train_label[index] = label.asnumpy()

        # iterator's basic settings
        self.batch_size = args.batch_size

        # NOTE: remove many awkward variants and leave the most core functionalities
        self.feedback_type = args.feedback_type
        self.active_instance = args.active_instance
        self.least_confident = args.least_confident
        self.active_question = args.active_question
        self.total_budget = args.total_budget
        self.round_budget = args.round_budget
        self.session_budget = args.session_budget if args.session_budget > 0 else self.round_budget
        self.score_rule = args.score_rule
        self.checkpoint_cost = args.checkpoint_cost
        self.total_cost = 0

        # NOTE: reinitialization helps fight against online learning's bias
        self.optimize_from_scratch = args.optimize_from_scratch
        self.max_optimize_epoch = args.max_optimize_epoch
        self.min_optimize_epoch = args.min_optimize_epoch
        self.reinit_round = args.reinit_round

        # NOTE: we want to start with a small set of data that is fully pre-labeled.
        self.prelabel_ratio = args.prelabel_ratio

        if self.prelabel_ratio > 0:
            indices = np.random.permutation(self.N)
            prelabel_num = int(self.N * self.prelabel_ratio)
            prelabel_inds = indices[:prelabel_num]
            for i in prelabel_inds:
                self.train_annotation[i] = False
                self.train_annotation[i][self.train_label[i]] = True
                self.train_annotation_sum[i] = 1

            self.logger.info('Pre-labeling examples %d %d ... %d %d' % (
                prelabel_inds[0], prelabel_inds[1], prelabel_inds[-2], prelabel_inds[-1]))

            self.train_iter.append_sequence(prelabel_inds, 'optimization')
            self.logger.info(
                'Adding %d prelabeled examples into optimization pool' % prelabel_num)

    def reinitialize(self):
        reinit_network(self.net, self.context, self.args)

    # def snapshot(self):
    #     self.logger.debug('Saving snapshot')
    #     net_copy(self.net, self.cnet, self.context, self.logger)

    # def rollback(self):
    #     self.logger.debug('Rolling back')
    #     net_copy(self.cnet, self.net, self.context, self.logger)

    def read_annotation(self, index):
        return split_and_load(self.train_annotation[index], ctx_list=self.context, batch_axis=0, even_split=False)

    def get_metrics(self):
        metrics = [mx.metric.Accuracy(name='acc-top1')]
        if self.dataset != 'cifar10':
            metrics.append(mx.metric.TopKAccuracy(top_k=5, name='acc-top5'))
        return metrics

    def log(self, trainval, oepoch, loss, metrics):
        msg = ['Query count=[%d]' % self.total_cost]
        if oepoch is not None:
            msg.append('Optimization epoch=[%d]' % oepoch)
        msg.append('full=[%f]' % (self.train_annotation_sum == 1).mean())
        msg.append('avg=[%f]' % self.train_annotation_sum.mean())
        if loss is not None:
            msg.append('%s-loss=[%f]' % (trainval, loss))
        for metric in metrics:
            msg.append('%s-%s=[%f]' %
                       (trainval, metric.get()[0], metric.get()[1]))
        self.logger.info(' '.join(msg))

    def log_no_metric(self, trainval, oepoch, loss, acc1, acc5=None):
        msg = ['Query count=[%d]' % self.total_cost]
        if oepoch is not None:
            msg.append('Optimization epoch=[%d]' % oepoch)
        msg.append('full=[%f]' % (self.train_annotation_sum == 1).mean())
        msg.append('avg=[%f]' % self.train_annotation_sum.mean())
        if loss is not None:
            msg.append('%s-loss=[%f]' % (trainval, loss))

        msg.append('%s-acc-top1=[%f]' % (trainval, acc1))
        if acc5:
            msg.append('%s-acc-top5_5=[%f]' % (trainval, acc5))
        self.logger.info(' '.join(msg))

    # when qj = -1, it means we fully labeled it (asking all questions)

    def record_annotation(self, index, question, old_annotation, new_annotation, prior):
        lines = ['%d %d %s %s\n' % (
            i, qj, ' '.join(['%d' % x for x in np.logical_xor(oa, na)]), ' '.join(['%f' % x for x in pr]))
                 for (i, qj, oa, na, pr) in zip(index, question, old_annotation, new_annotation, prior)]
        with open(self.book_file, 'a') as book:
            book.write(''.join(lines))

    def work(self):
        # train on prelabels
        round_num = 0

        if self.prelabel_ratio > 0:
            self.train_iter.switch('optimization')
            self.train_iter.reset()

            self.optimize(self.max_optimize_epoch)

            _, val_metrics = self.validate()
            self.log('val', None, None, val_metrics)

        while self.total_cost < self.total_budget and np.any(self.train_annotation_sum > 1):
            # ROUND
            # round_cost = 0
            # STEP 1: recompute prior if needed
            if self.active_instance or self.active_question:
                # switch to query mode (we look at the whole training set)
                self.train_iter.switch('query')
                self.train_iter.reset()
                batch_softmax = []
                batch_index = []
                self.logger.info('Updating prior for every data point in query pool (%d)' % len(
                    self.train_iter.seqs['query']))
                for i, batch in enumerate(self.train_iter):
                    data, _, index = unpack_batch(batch, self.context)
                    output = [self.net(d) for d in data]
                    # compute softmax prediction
                    softmax = [nd.softmax(o, axis=-1) for o in output]
                    batch_softmax += softmax
                    batch_index += index
                batch_softmax = nd.concatenate(batch_softmax).asnumpy()
                train_softmax = np.zeros_like(batch_softmax)
                train_softmax[batch_index] = batch_softmax
                masked_train_softmax = train_softmax * self.train_annotation
                self.train_prior = masked_train_softmax / \
                    masked_train_softmax.sum(axis=-1, keepdims=True)

            # STEP 2: query with full feedback or partial feedback
            self.logger.info('Scoring every data point in the query pool')
            round_set = set()  # examples actually queried
            if self.feedback_type == 'full':
                scope = np.where(self.train_annotation_sum > 1)[0]
                if self.active_instance:
                    if self.least_confident:
                        maxprob = np.max(self.train_prior[scope], axis=-1)
                        argsort = np.argsort(maxprob)
                    else:
                        # we compute the entropy for each training instance and choose the most uncertain ones
                        fullent = - \
                            np.sum(
                                self.train_prior[scope]*np.log2(self.train_prior[scope]), axis=-1)
                        # replace nan with zero and inf with finite numbers.
                        fullent = np.nan_to_num(fullent)
                        # sort entropy from high to low
                        argsort = np.argsort(-fullent)
                else:
                    # randomly pick from not fully labeled ones
                    argsort = np.arange(len(scope))
                    np.random.shuffle(argsort)

                scope = scope[argsort]

                if self.active_question:
                    train_prior = self.train_prior.copy()
                else:
                    train_prior = self.train_annotation / \
                        self.train_annotation_sum[:, np.newaxis]

                data_count = 0
                round_cost = 0
                # ii: data index
                for ii in scope:
                    annotation = np.ones(self.K, dtype=np.int)

                    I = []
                    J = []
                    old_annotation = []
                    new_annotation = []
                    prior = []

                    cost = 0
                    # HAD A BUG: When using score rule of EIG, this network might be too confident
                    # that it does not even think any question is worth asking. What happen is,
                    # every score will be zero. Then when we do argmax, it always return 0, which
                    # means we will ask the first question. If the first question has been answered,
                    # there will be no annotation change, thus we will be stuck here forever in a
                    # infinite loop.
                    while annotation.sum(axis=-1) > 1:
                        pr = train_prior[ii] * annotation
                        # if pr.sum() == 0:
                        if pr.sum() < 1e-8:
                            # pr[annotation==1] = 1.0 / (annotation==1).sum()
                            pr = annotation * 1.0 / annotation.sum()
                        else:
                            pr = pr / pr.sum()
                        prior.append(pr)

                        qp = np.dot(pr, self.Q.T)
                        mere = np.nan_to_num(-(qp *
                                             np.log2(qp)+(1-qp)*np.log2(1-qp)))
                        nrc_correct = np.dot(annotation, self.Q.T)
                        nrc_incorrect = np.dot(annotation, (1-self.Q).T)
                        enrc = qp * nrc_correct + (1-qp) * nrc_incorrect
                        mernrc = self.train_annotation_sum[ii] - enrc
                        if self.score_rule == 'MERE':
                            score = mere
                        elif self.score_rule == 'MERNRC':
                            score = mernrc
                        elif self.score_rule == 'MinENRC':
                            score = -enrc
                        else:
                            raise Exception(
                                'Unknown score rule: %s' % self.score_rule)

                        I.append(ii)
                        # argmax = np.argmax(mere)
                        if np.all(score == 0):
                            argmax = pr.argmax(axis=-1)
                        else:
                            argmax = score.argmax(axis=-1)

                        J.append(argmax)
                        answer = self.Q[argmax, self.train_label[ii]]
                        # old_annotation.append(annotation)
                        old_annotation.append(annotation.copy())
                        annotation *= (answer*self.Q[argmax] +
                                       (1-answer)*(1-self.Q[argmax]))

                        # new_annotation.append(annotation)
                        new_annotation.append(annotation.copy())
                        self.train_annotation[ii] = annotation
                        self.train_annotation_sum[ii] = annotation.sum()
                        cost += 1

                    data_count += 1
                    round_cost += cost
                    self.record_annotation(
                        I, J, old_annotation, new_annotation, prior)

                    if round_cost >= self.round_budget:
                        break

                self.logger.info('Fully labeled %d data points with a round budget of %d' % (
                    data_count, round_cost))
                # self.logger.info('round budget %d, #data %d, cost %d' % (self.round_budget, data_count, data_cost))
                round_set = round_set.union(scope[:data_count])

            elif self.feedback_type == 'partial':
                if self.active_question:
                    train_prior = self.train_prior.copy()
                else:
                    train_prior = self.train_annotation / \
                        self.train_annotation_sum[:, np.newaxis]

                # precompute the score at once
                qp = np.dot(train_prior, self.Q.T)
                mere = np.nan_to_num(-(qp*np.log2(qp)+(1-qp)*np.log2(1-qp)))
                # NOTE: there is a very efficient and elegant way of counting the number of remaining classes
                # after some bookkeeping, I found if I multiply annotation matrix with the transposed version of Q
                # the results is a N x Q matrix with the number of remaining classes
                nrc_correct = np.dot(self.train_annotation, self.Q.T)
                nrc_incorrect = np.dot(self.train_annotation, (1-self.Q).T)
                enrc = qp * nrc_correct + (1-qp) * nrc_incorrect
                # maximum expected reduction in number of remaining classes
                mernrc = self.train_annotation_sum[:, np.newaxis] - enrc
                # compute scores
                if self.score_rule == 'RMERE':
                    score = mere/enrc
                elif self.score_rule == 'MERE':
                    score = mere
                elif self.score_rule == 'MERNRC':
                    score = mernrc
                elif self.score_rule == 'MinENRC':  # minimum expected number of remaining classes
                    score = -enrc
                else:
                    raise Exception('Unknown score rule: %s' % self.score_rule)

                # pick the best question for each data
                argmax = score.argmax(axis=-1)
                best_score = score[np.arange(score.shape[0]), argmax]

                # NOTE: build the heap (min heap)
                heap = []
                for ii in np.arange(self.N):
                    if self.train_annotation_sum[ii] == 1:
                        continue
                    heappush(heap, (-best_score[ii], ii, argmax[ii]))

                # enter the querying round
                session_count = 0
                round_cost = 0
                while round_cost < self.round_budget and len(heap) > 0:
                    # scope = np.where(self.train_annotation_sum > 1)[0]
                    # if not self.active_instance:
                    #     scope = np.random.choice(scope, self.session_budget, replace=False)
                    if self.active_question and (not self.active_instance):
                        raise NotImplementedError(
                            'Have not thought about this very carefully')

                    # pop the m best
                    mbest = [heappop(heap) for _ in range(self.session_budget)]
                    # I = [mb[1] for mb in mbest]
                    # J = argmax[I]
                    I = [mb[1] for mb in mbest]
                    J = [mb[2] for mb in mbest]
                    anno_if_correct = self.train_annotation[I]*(
                        self.Q[J].astype(np.bool))
                    anno_if_incorrect = self.train_annotation[I]*(
                        1-self.Q[J].astype(np.bool))
                    answer = self.Q[J][np.arange(len(J)), self.train_label[I]]
                    answer = answer[:, np.newaxis]
                    new_annotation = answer*anno_if_correct + \
                        (1-answer)*anno_if_incorrect
                    # record queries
                    self.record_annotation(
                        I, J, self.train_annotation[I], new_annotation, train_prior[I])
                    self.train_annotation[I] = new_annotation
                    self.train_annotation_sum[I] = new_annotation.sum(axis=-1)
                    # update prior, score, argmax, best_score, and heap
                    new_prior = train_prior[I] * new_annotation
                    new_prior = new_prior / \
                        new_prior.sum(axis=-1, keepdims=True)
                    train_prior[I] = new_prior
                    # recompute score
                    qp_ = np.dot(train_prior[I], self.Q.T)
                    mere_ = np.nan_to_num(-(qp_*np.log2(qp_) +
                                          (1-qp_)*np.log2(1-qp_)))
                    nrc_correct_ = np.dot(self.train_annotation[I], self.Q.T)
                    nrc_incorrect_ = np.dot(
                        self.train_annotation[I], (1-self.Q).T)
                    enrc_ = qp_ * nrc_correct_ + (1-qp_) * nrc_incorrect_
                    # maximum expected reduction in number of remaining classes
                    mernrc_ = self.train_annotation_sum[I, np.newaxis] - enrc_
                    # compute scores
                    if self.score_rule == 'RMERE':
                        score_ = mere_/enrc_
                    elif self.score_rule == 'MERE':
                        score_ = mere_
                    elif self.score_rule == 'MERNRC':
                        score_ = mernrc_
                    elif self.score_rule == 'MinENRC':  # minimum expected number of remaining classes
                        score_ = -enrc_
                    # pick the best question for each data
                    argmax_ = score_.argmax(axis=-1)
                    best_score_ = score_[np.arange(score_.shape[0]), argmax_]
                    #
                    for bs_, ii, jj in zip(best_score_, I, argmax_):
                        if self.train_annotation_sum[ii] == 1:
                            continue
                        heappush(heap, (-bs_, ii, jj))
                    # update round stats
                    round_cost += self.session_budget
                    round_set = round_set.union(I)
                    session_count += 1
                    if session_count % 1000 == 0:
                        self.logger.info('Session %d, budget %d/%d, %d examples queried in total' % (
                            session_count, round_cost, self.round_budget, len(round_set)))

            # optimization
            old_num = len(self.train_iter.seqs['optimization'])
            self.train_iter.append_sequence(round_set, 'optimization')
            new_num = len(self.train_iter.seqs['optimization'])
            self.logger.info(
                'Adding %d new data points into optimization pool' % (new_num-old_num))

            # NOTE: yeah this was a bug
            # self.total_cost += self.round_budget
            self.total_cost += round_cost

            # # if the previous round was the last round
            # max_epoch = self.max_epoch_each_round

            if self.optimize_from_scratch and (round_num+1) % self.reinit_round == 0:
                # save those parameters everytime we reinitilaize
                self.logger.info('Saving network parameters...')
                param_file = os.path.join(
                    self.param_dir, '%s_%d.params' % (self.tag, self.total_cost))
                self.net.save_params(param_file)

                # reinitialize
                self.logger.info('Reinitializing the network...')
                self.reinitialize()

                #
                self.logger.info('Resetting training iterators...')
                self.train_iter.switch('optimization')
                self.train_iter.reset()

                #
                self.logger.info('Start optimization seriously...')
                self.optimize(self.max_optimize_epoch)

                # now we validate
                self.logger.info('Start validatiion...')
                _, val_metrics = self.validate()
                self.log('val', None, None, val_metrics)

            elif self.min_optimize_epoch > 0:
                self.logger.info('Resetting training iterators...')
                self.train_iter.switch('optimization')
                self.train_iter.reset()

                self.logger.info('Start optimization casually...')
                self.optimize(self.min_optimize_epoch)

            round_num += 1

        self.logger.info('Resetting training iterators...')
        self.train_iter.switch('optimization')
        self.train_iter.reset()

        # this ensures that everytime we evaluate performance, all data has been trained for 75
        # epochs. what we do in between only affects how we asks questions.
        self.logger.info('Start optimization seriously...')
        self.optimize(self.max_optimize_epoch)

    def optimize(self, max_epoch):

        self.logger.info('Optimizing for %d epochs' % max_epoch)

        # NOTE: we don't evaluate on validation
        train_metrics = self.get_metrics()
        for epoch in range(max_epoch):
            for metric in train_metrics:
                metric.reset()
            train_loss, train_num = 0, 0
            self.train_iter.reset()

            for i, batch in enumerate(self.train_iter):
                data, label, index = unpack_batch(batch, self.context)
                annotation = self.read_annotation(index)
                with mx.autograd.record():
                    output = [self.net(x) for x in data]
                    loss = [loss_func(o, a)
                                      for (o, a) in zip(output, annotation)]
                for L in loss:
                    L.backward()
                self.trainer.step(batch.data[0].shape[0])
                train_loss += np.sum([nd.sum(L).asnumpy() for L in loss])
                train_num += batch.data[0].shape[0]

                for metric in train_metrics:
                    metric.update(label, output)
            train_loss /= train_num
            self.log('train', epoch+1, train_loss, train_metrics)

    def validate(self):
        self.val_iter.reset()
        val_metrics = self.get_metrics()
        val_loss, val_num = 0, 0
        for i, batch in enumerate(self.val_iter):
            data, label = unpack_batch(batch, self.context)
            output = [self.net(x) for x in data]
            loss = [loss_func(o, nd.one_hot(l, self.K)) for (o, l) in zip(output, label)]
            val_loss += np.sum([nd.sum(L).asnumpy() for L in loss])
            val_num += batch.data[0].shape[0]
            for val_metric in val_metrics:
                val_metric.update(label, output)
        return val_loss/val_num, val_metrics
