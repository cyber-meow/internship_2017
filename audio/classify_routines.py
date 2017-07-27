from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from audio.basics import TrainAudio, EvaluateAudio
from classify.train import TrainClassifyCNN
from classify.evaluate import EvaluateClassifyCNN
from data.avicar import load_batch_avicar, get_split_avicar

slim = tf.contrib.slim


def CNN_mfcc(inputs,
             final_endpoint='Conv2d_e_3x3',
             scope=None):

    with tf.variable_scope(scope, 'CNN', [inputs]):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                            stride=1, padding='VALID'):

            # 26 x 24 x 1
            print(inputs.get_shape())
            endpoint = 'Conv2d_a_2x2'
            net = slim.conv2d(inputs, 5, [2, 2], scope='Conv2d_a_2x2')
            if final_endpoint == endpoint:
                return net

            # 25 x 23 x 5
            print(net.get_shape())
            endpoint = 'Conv2d_b_3x3'
            net = slim.conv2d(net, 13, [3, 3], stride=2,
                              scope='Conv2d_b_3x3')
            if final_endpoint == endpoint:
                return net

            # 12 x 11 x 13
            print(net.get_shape())
            endpoint = 'MaxPool_a_2x2'
            net = slim.max_pool2d(net, [2, 2], stride=2, padding='SAME',
                                  scope='MaxPool_a_2x2')
            if final_endpoint == endpoint:
                return net

            # 6 x 6 x 13
            print(net.get_shape())
            endpoint = 'Conv2d_c_2x2'
            net = slim.conv2d(net, 23, [2, 2], scope='Conv2d_c_2x2')
            if final_endpoint == endpoint:
                return net

            # 5 x 5 x 23
            print(net.get_shape())
            endpoint = 'Conv2d_d_3x3'
            net = slim.conv2d(net, 31, [3, 3], scope='Conv2d_d_3x3')
            if final_endpoint == endpoint:
                return net

            # 3 x 3 x 31
            print(net.get_shape())
            endpoint = 'Conv2d_e_3x3'
            net = slim.conv2d(net, 43, [3, 3], scope='Conv2d_e_3x3')
            print(net.get_shape())
            if final_endpoint == endpoint:
                return net

            # 1 x 1 x 43
            raise ValueError('Unknown final endpoint %s' % final_endpoint)


def CNN_mfcc4(inputs,
              final_endpoint='Conv2d_c_4x3',
              scope=None):

    with tf.variable_scope(scope, 'CNN', [inputs]):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                            stride=2, padding='SAME'):

            # 26 x 24 x 1
            print(inputs.get_shape())
            endpoint = 'Conv2d_a_3x3'
            net = slim.conv2d(inputs, 7, [3, 3], scope='Conv2d_a_3x3')
            if final_endpoint == endpoint:
                return net

            # 13 x 12 x 7
            print(net.get_shape())
            endpoint = 'MaxPool_a_2x2'
            net = slim.max_pool2d(net, [2, 2], scope='MaxPool_a_2x2')
            if final_endpoint == endpoint:
                return net

            # 7 x 6 x 7
            print(net.get_shape())
            endpoint = 'Conv2d_b_3x3'
            net = slim.conv2d(net, 17, [3, 3], scope='Conv2d_b_3x3')
            if final_endpoint == endpoint:
                return net

            # 4 x 3 x 17
            print(net.get_shape())
            endpoint = 'Conv2d_c_4x3'
            net = slim.conv2d(net, 61, [4, 3], stride=1,
                              padding='VALID', scope='Conv2d_c_4x3')
            print(net.get_shape())
            if final_endpoint == endpoint:
                return net

            # 1 x 1 x 61
            raise ValueError('Unknown final endpoint %s' % final_endpoint)


def delta(coefs, N=2):
    """
    delta: A tensor of shape [batch, feature_len, num_frames, 1]
    """
    res = tf.zeros_like(coefs)
    for n in range(1, N+1):
        minus_part = tf.concat([
                tf.tile(tf.expand_dims(coefs[:, :, 0, :], 2), [1, 1, n, 1]),
                coefs[:, :, :-n, :]
            ], 2)
        plus_part = tf.concat([
                coefs[:, :, n:, :],
                tf.tile(tf.expand_dims(coefs[:, :, -1, :], 2), [1, 1, n, 1])
            ], 2)
        res += n * (plus_part-minus_part)
    res /= 2*sum([n**2 for n in range(1, N+1)])
    return res


class TrainClassifyAudio(TrainAudio, TrainClassifyCNN):

    def decide_used_data(self):
        self.mfccs = tf.cond(
            self.training, lambda: self.mfccs_train, lambda: self.mfccs_test)
        self.labels = tf.cond(
            self.training, lambda: self.labels_train, lambda: self.labels_test)

    def compute(self, use_delta=False, **kwargs):
        if use_delta:
            mfcc_deltas = delta(self.mfccs)
            delta_deltas = delta(mfcc_deltas)
            data = tf.concat([self.mfccs, mfcc_deltas, delta_deltas], axis=3)
        else:
            data = self.mfccs
        self.logits = self.compute_logits(
            data, self.dataset_train.num_classes, **kwargs)

    def get_summary_op(self):
        self.get_batch_norm_summary()
        tf.summary.scalar('learning_rate', self.learning_rate)
        tf.summary.histogram('logits', self.logits)
        tf.summary.scalar('losses/train/cross_entropy',
                          self.cross_entropy_loss)
        tf.summary.scalar('losses/train/total', self.total_loss)
        tf.summary.scalar('accuracy/train', self.accuracy_no_streaming)
        tf.summary.scalar('accuracy/train/streaming', self.accuracy)
        self.summary_op = tf.summary.merge_all()
        return self.summary_op

    def get_test_summary_op(self):
        # Summaries for the test part
        ac_test_summary = tf.summary.scalar(
            'accuracy/test', self.accuracy_no_streaming)
        ls_test_summary = tf.summary.scalar(
            'losses/test/total', self.total_loss)
        self.test_summary_op = tf.summary.merge(
            [ac_test_summary, ls_test_summary])
        return self.test_summary_op


class EvaluateClassifyAudio(EvaluateAudio, EvaluateClassifyCNN):

    def compute(self, use_delta=False, **kwargs):
        if use_delta:
            mfcc_deltas = delta(self.mfccs)
            delta_deltas = delta(mfcc_deltas)
            data = tf.concat([self.mfccs, mfcc_deltas, delta_deltas], axis=3)
        else:
            data = self.mfccs
        self.logits = self.compute_logits(
            data, self.dataset.num_classes, **kwargs)

    def last_step_log_info(self, sess, batch_size):
        return self.step_log_info(sess)


class TrainClassifyAvicar(TrainClassifyCNN):

    def get_data(self, tfrecord_dir, batch_size):
        self.dataset_train = get_split_avicar('train', tfrecord_dir)
        self.mfccs_train, self.labels_train = \
            load_batch_avicar(self.dataset_train, batch_size=batch_size)
        self.dataset_test = get_split_avicar('validation', tfrecord_dir)
        self.mfccs_test, self.labels_test = \
            load_batch_avicar(self.dataset_test, batch_size=batch_size)
        return self.dataset_train

    def decide_used_data(self):
        self.mfccs = tf.cond(
            self.training, lambda: self.mfccs_train, lambda: self.mfccs_test)
        self.labels = tf.cond(
            self.training, lambda: self.labels_train, lambda: self.labels_test)

    def compute(self, use_delta=False, **kwargs):
        if use_delta:
            mfcc_deltas = delta(self.mfccs)
            delta_deltas = delta(mfcc_deltas)
            data = tf.concat([self.mfccs, mfcc_deltas, delta_deltas], axis=3)
        else:
            data = self.mfccs
        self.logits = self.compute_logits(
            data, self.dataset_train.num_classes, **kwargs)

    def get_summary_op(self):
        self.get_batch_norm_summary()
        tf.summary.scalar('learning_rate', self.learning_rate)
        tf.summary.histogram('logits', self.logits)
        tf.summary.scalar('losses/train/cross_entropy',
                          self.cross_entropy_loss)
        tf.summary.scalar('losses/train/total', self.total_loss)
        tf.summary.scalar('accuracy/train', self.accuracy_no_streaming)
        tf.summary.scalar('accuracy/train/streaming', self.accuracy)
        self.summary_op = tf.summary.merge_all()
        return self.summary_op

    def get_test_summary_op(self):
        # Summaries for the test part
        ac_test_summary = tf.summary.scalar(
            'accuracy/test', self.accuracy_no_streaming)
        ls_test_summary = tf.summary.scalar(
            'losses/test/total', self.total_loss)
        self.test_summary_op = tf.summary.merge(
            [ac_test_summary, ls_test_summary])
        return self.test_summary_op
