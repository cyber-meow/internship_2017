from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import abc
import tensorflow as tf

from routines.evaluate import Evaluate

slim = tf.contrib.slim


class EvaluateClassify(Evaluate):

    @abc.abstractmethod
    def compute(self, **kwargs):
        """Feed proper inputs to the method compute_logits"""
        pass

    @abc.abstractmethod
    def compute_logits(self, inputs):
        pass

    def compute_log_data(self):
        self.predictions = tf.argmax(tf.nn.softmax(self.logits), 1)
        self.accuracy = tf.reduce_mean(tf.cast(
            tf.equal(self.predictions, self.labels), tf.float32))
        self.accuracy_summary = tf.summary.scalar('accuracy', self.accuracy)

    def step_log_info(self, sess):
        start_time = time.time()
        global_step_count, accuracy_rate, ac_summary = sess.run(
            [self.global_step_op, self.accuracy, self.accuracy_summary])
        time_elapsed = time.time() - start_time
        tf.logging.info(
            'global step %s: accurarcy: %.4f (%.2f sec/step)',
            global_step_count, accuracy_rate, time_elapsed)
        if hasattr(self, 'fw'):
            self.fw.add_summary(ac_summary, global_step=global_step_count)


class EvaluateClassifyCNN(EvaluateClassify):

    def __init__(self, CNN_structure, **kwargs):
        super(EvaluateClassifyCNN, self).__init__(**kwargs)
        self.CNN_structure = CNN_structure

    def compute_logits(self, inputs, num_classes, endpoint=None):
        if self.CNN_structure is not None:
            if endpoint is not None:
                net = self.CNN_structure(inputs, final_endpoint=endpoint)
            else:
                net = self.CNN_structure(inputs)
        else:
            net = inputs
        net = slim.flatten(net, scope='PreLogitsFlatten')
        logits = slim.fully_connected(
            net, num_classes, activation_fn=None, scope='Logits')
        return logits
