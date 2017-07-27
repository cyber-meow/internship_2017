from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf

from routines.train import TrainImages

slim = tf.contrib.slim


class TrainClassify(TrainImages):

    @abc.abstractmethod
    def decide_used_data(self):
        pass

    @abc.abstractmethod
    def compute(self, **kwargs):
        """Feed proper inputs to the method compute_logits"""
        pass

    @abc.abstractmethod
    def compute_logits(self, inputs, num_classes):
        pass

    def get_total_loss(self):
        num_classes = self.dataset_train.num_classes
        one_hot_labels = tf.one_hot(self.labels, num_classes)
        self.cross_entropy_loss = \
            tf.losses.softmax_cross_entropy(one_hot_labels, self.logits)
        self.total_loss = tf.losses.get_total_loss()
        return self.total_loss

    def get_metric_op(self):
        self.predictions = tf.argmax(tf.nn.softmax(self.logits), 1)
        self.streaming_accuracy, self.streaming_accuracy_update = \
            tf.metrics.accuracy(self.predictions, self.labels)
        self.metric_op = tf.group(self.accuracy_update)
        self.accuracy = tf.reduce_mean(tf.cast(
            tf.equal(self.predictions, self.labels), tf.float32))
        return self.metric_op

    def get_summary_op(self):
        self.get_batch_norm_summary()
        tf.summary.scalar('learning_rate', self.learning_rate)
        tf.summary.histogram('logits', self.logits)
        tf.summary.scalar('losses/train/cross_entropy',
                          self.cross_entropy_loss)
        tf.summary.scalar('losses/train/total', self.total_loss)
        tf.summary.scalar('accuracy/train', self.accuracy)
        tf.summary.scalar('accuracy/train/streaming', self.streaming_accuracy)
        self.summary_op = tf.summary.merge_all()
        return self.summary_op

    def get_test_summary_op(self):
        # Summaries for the test part
        accuracy_test_summary = tf.summary.scalar(
            'accuracy/test', self.accuracy)
        loss_test_summary = tf.summary.scalar(
            'losses/test/total', self.total_loss)
        self.test_summary_op = tf.summary.merge(
            [accuracy_test_summary, loss_test_summary])
        return self.test_summary_op

    def summary_log_info(self, sess):
        loss, _, _, summaries, streaming_accuracy_rate, accuracy_rate = \
            self.train_step(
                sess, self.train_op, self.sv.global_step, self.metric_op,
                self.summary_op, self.streaming_accuracy, self.accuracy)
        tf.logging.info(
            'Current Streaming Accuracy:%s', streaming_accuracy_rate)
        tf.logging.info('Current Accuracy:%s', accuracy_rate)
        self.sv.summary_computed(sess, summaries)

    def test_log_info(self, sess, test_use_batch):
        loss, accuracy_rate, summaries_test = sess.run(
            [self.total_loss, self.accuracy, self.test_summary_op],
            feed_dict={self.training: False,
                       self.batch_stat: test_use_batch})
        tf.logging.info('Current Test Loss: %s', loss)
        tf.logging.info('Current Test Accuracy: %s', accuracy_rate)
        self.sv.summary_computed(sess, summaries_test)


class TrainClassifyCNN(TrainClassify):

    def __init__(self, CNN_structure, **kwargs):
        super(TrainClassifyCNN, self).__init__(**kwargs)
        self.CNN_structure = CNN_structure

    def compute_logits(self, inputs, num_classes,
                       dropout_keep_prob=0.8, endpoint=None,
                       per_layer_dropout=None):
        if self.CNN_structure is not None:
            if endpoint is not None:
                if per_layer_dropout is not None:
                    net = self.CNN_structure(
                        inputs, final_endpoint=endpoint,
                        per_layer_dropout=per_layer_dropout,
                        dropout_keep_prob=dropout_keep_prob)
                else:
                    net = self.CNN_structure(inputs, final_endpoint=endpoint)
            else:
                if per_layer_dropout is not None:
                    net = self.CNN_structure(
                        inputs,
                        per_layer_dropout=per_layer_dropout,
                        dropout_keep_prob=dropout_keep_prob)
                else:
                    net = self.CNN_structure(inputs)
        else:
            net = inputs
        net = slim.dropout(net, dropout_keep_prob, scope='PreLogitsDropout')
        net = slim.flatten(net, scope='PreLogitsFlatten')
        print('Prelogits shape: ', net.get_shape())
        logits = slim.fully_connected(
            net, num_classes, activation_fn=None, scope='Logits')
        return logits
