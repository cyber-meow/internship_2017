from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from audio.basics import TrainAudio, EvaluateAudio
from classify.train import TrainClassifyCNN
from classify.evaluate import EvaluateClassifyCNN

slim = tf.contrib.slim


def CNN_mfcc(inputs,
             final_endpoint='Conv2d_e_3x3',
             scope=None):

    with tf.variable_scope(scope, 'CNN', [inputs]):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                            stride=1, padding='VALID'):

            # 26 x 24 x 1
            endpoint = 'Conv2d_a_2x2'
            net = slim.conv2d(inputs, 5, [2, 2], scope='Conv2d_a_2x2')
            if final_endpoint == endpoint:
                return net

            # 25 x 23 x 5
            endpoint = 'Conv2d_b_3x3'
            net = slim.conv2d(net, 13, [3, 3], stride=2,
                              scope='Conv2d_b_3x3')
            if final_endpoint == endpoint:
                return net

            # 12 x 11 x 13
            endpoint = 'MaxPool_a_2x2'
            net = slim.max_pool2d(net, [2, 2], stride=2, padding='SAME',
                                  scope='MaxPool_a_2x2')
            if final_endpoint == endpoint:
                return net

            # 6 x 6 x 13
            endpoint = 'Conv2d_c_2x2'
            net = slim.conv2d(net, 23, [2, 2], scope='Conv2d_c_2x2')
            if final_endpoint == endpoint:
                return net

            # 5 x 5 x 23
            endpoint = 'Conv2d_d_3x3'
            net = slim.conv2d(net, 31, [3, 3], scope='Conv2d_d_3x3')
            if final_endpoint == endpoint:
                return net

            # 3 x 3 x 31
            endpoint = 'Conv2d_e_3x3'
            net = slim.conv2d(net, 43, [3, 3], scope='Conv2d_e_3x3')
            if final_endpoint == endpoint:
                return net

            # 1 x 1 x 43
            raise ValueError('Unknown final endpoint %s' % final_endpoint)


class TrainClassifyAudio(TrainAudio, TrainClassifyCNN):

    def decide_used_data(self):
        self.mfccs = tf.cond(
            self.training, lambda: self.mfccs_train, lambda: self.mfccs_test)
        self.labels = tf.cond(
            self.training, lambda: self.labels_train, lambda: self.labels_test)

    def compute(self, **kwargs):
        self.logits = self.compute_logits(
            self.mfccs, self.dataset_train.num_classes, **kwargs)

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

    def compute(self, **kwargs):
        self.logits = self.compute_logits(
            self.mfccs, self.dataset.num_classes, **kwargs)

    def last_step_log_info(self, sess, batch_size):
        return self.step_log_info(sess)
