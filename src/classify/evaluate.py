"""Routines used to evaluate a classifier

The file contains two classes `EvaluateClassify` and `EvaluateClassifyCNN`.

`EvaluateClassify` can be used for any network architecture and prediction
accuracy is evaluated. `EvaluateClassifyCNN` is used for a CNN classifcation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import abc
import tensorflow as tf

from routines.evaluate import Evaluate

slim = tf.contrib.slim


class EvaluateClassify(Evaluate):
    """Implement routines for evaluating a classifier.

    To inherit from this class one should implement `get_data`,
    `compute` and `comput_logits`.

    See `EvaluateClassifyImagesCNN` for an example.
    """

    @abc.abstractmethod
    def compute(self, **kwargs):
        """Feed proper inputs to the method `compute_logits`.

        N.B. The returned value of `compute_logits` must be stored
        in the attribute `self.logits`.

        Args:
            **kwargs: Arbitrary arguments. Normally most of them are
                directly fed to `compute_logits`.
        """
        pass

    @abc.abstractmethod
    def compute_logits(self, inputs, num_classes, **kwargs):
        """Compute logits of the model.

        This should be the most important part of a classifier.
        Logits are unscaled log probabilities and they're mapped
        to real probabilities by applying a softmax function.
        For example to implement a perceptron we use simply a
        fully connected layer.

        Args:
            inputs: The input(s) of the network/algorithm.
            num_classes: The number of classes to be classified.
                This is also the number of neurons in the output layer.
            **kwargs: Other arguments.

        Returns:
            Computed logits of the model.
        """
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
    """Class that is used to evaluate any CNN architectures

    To inherit from this class one should implement `get_data`
    and `compute`.

    See `EvaluateClassifyImagesCNN` for an example.
    """

    def __init__(self, CNN_architecture, **kwargs):
        """Declare the architecture that is used by the class instance.

        Args:
            CNN_architecture: The architecture that is used to compute
                the layer just before logits. If given as `None` no
                particular computations are done and we train therefore
                a single-layer perception.
            **kwargs: Other arguments used by the superclass.
        """
        super(EvaluateClassifyCNN, self).__init__(**kwargs)
        self.CNN_architecture = CNN_architecture

    def compute_logits(self, inputs, num_classes, endpoint=None):
        """Compute logits using some CNN architecture.

        Args:
            inputs: The input(s) of the network.
            num_classes: The number of classes to be classified.
                This is also the number of neurons in the output layer.
            endpoint: The endpoint of the network. If `None` use the
                default endpoint of each network.
        """
        if self.CNN_architecture is not None:
            if endpoint is not None:
                net = self.CNN_architecture(inputs, final_endpoint=endpoint)
            else:
                net = self.CNN_architecture(inputs)
        else:
            net = inputs
        net = slim.flatten(net, scope='PreLogitsFlatten')
        logits = slim.fully_connected(
            net, num_classes, activation_fn=None, scope='Logits')
        return logits
