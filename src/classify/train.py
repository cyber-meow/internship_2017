"""Routines used to train a classifier

The file contains two classes `TrainClassify` and `TrainClassifyCNN`.

`TrainClassify` can be used for any network structure as long as
we always use cross entropy loss. Prediction accuracy is evaluated
through the whole training process.

To inherit from this class one should implement `get_data`,
`decide_use_data`, `compute` and `comput_logits`.

CNN structures are quite often employed for classification. The class
`TrainClassifyCNN` suppose in input there is only one modality
and we feed this modality to a (CNN) network to do classification.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf

from routines.train import Train

slim = tf.contrib.slim


class TrainClassify(Train):
    """Implement routines for training a classifier.

    The cross entropy loss is used. We compute prediction accuracy every
    `save_summaries_steps` steps for a batch of training data and
    validation data (if `do_test` is `True`).

    A subclass of this class must implement `get_data`, `decide_use_data`,
    `compute` and `compute_logits`. See `TrainClassifyImagesCNN` for
    an example.
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

    def get_total_loss(self):
        """Use cross entropy loss"""
        num_classes = self.dataset_train.num_classes
        one_hot_labels = tf.one_hot(self.labels, num_classes)
        self.cross_entropy_loss = \
            tf.losses.softmax_cross_entropy(one_hot_labels, self.logits)
        self.total_loss = tf.losses.get_total_loss()
        return self.total_loss

    def get_metric_op(self):
        """Trace prediction accuracy and streaming prediction accuracy."""
        self.predictions = tf.argmax(tf.nn.softmax(self.logits), 1)
        self.streaming_accuracy, self.streaming_accuracy_update = \
            tf.metrics.accuracy(self.predictions, self.labels)
        self.metric_op = tf.group(self.streaming_accuracy_update)
        self.accuracy = tf.reduce_mean(tf.cast(
            tf.equal(self.predictions, self.labels), tf.float32))
        return self.metric_op

    def get_summary_op(self):
        """Show learning rate, logit activations, losses and prediction
        accuracy on Tensorboard."""
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
        """Show total loss and prediction accuracy for test on Tensorboard."""
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
    """Class that is used to train any CNN architectures.

    In its subclass we only need to define `get_data`, `decide_use_data`
    and `compute` to deal with input data. The method `compute`
    is only meant to call `compute_logits` with proper arguments (depending
    on the implementation of `get_data` and `decide_use_data`) and store
    its returned values in `self.logits`.

    See `TrainClassifyImagesCNN` for an example.
    """

    def __init__(self, CNN_structure, **kwargs):
        """Declare the architecture that is used by the class instance.

        Args:
            CNN_structure: The architecture that is used to compute
                the layer just before logits. If given as `None` no
                particular computations are done and we train therefore
                a single-layer perception.
            **kwargs: Other arguments used by the superclass.
        """
        super(TrainClassifyCNN, self).__init__(**kwargs)
        self.CNN_structure = CNN_structure

    def compute_logits(self, inputs, num_classes,
                       dropout_keep_prob=0.8, endpoint=None,
                       per_layer_dropout=None):
        """Compute logits using some CNN architecture.

        Args:
            inputs: The input(s) of the network.
            num_classes: The number of classes to be classified.
                This is also the number of neurons in the output layer.
            dropout_keep_prob: Dropout is used just before the
                last fully connected layer which computes logits.
                This is the probability that each individual neuron
                value is kept. Must be in the interval (0, 1].
            endpoint: The endpoint of the network. If `None` use the
                default endpoint of each network.
            per_layer_dropout: If we apply dropout after every layer
                of the network. This was just for test purpose and it
                should better be left as `None`.
        """
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
        print('Prelogits shape: ', net.get_shape())
        net = slim.flatten(net, scope='PreLogitsFlatten')
        logits = slim.fully_connected(
            net, num_classes, activation_fn=None, scope='Logits')
        return logits
