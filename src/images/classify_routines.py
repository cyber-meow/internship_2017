"""Train and evaluate image classifiers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import tensorflow as tf

from nets_base import inception_v4
from images.basics import TrainImages, EvaluateImages
from classify.train import TrainClassify, TrainClassifyCNN
from classify.evaluate import EvaluateClassify, EvaluateClassifyCNN

slim = tf.contrib.slim


class TrainClassifyImages(TrainImages, TrainClassify):
    """Abstract class to train an image classifier."""

    def compute(self, **kwargs):
        self.logits = self.compute_logits(
            self.images, self.dataset_train.num_classes, **kwargs)

    def get_summary_op(self):
        """Also show some training input images on Tensorboard."""
        super(TrainClassifyImages, self).get_summary_op()
        tf.summary.image('train', self.images)
        self.summary_op = tf.summary.merge_all()
        return self.summary_op

    def get_test_summary_op(self):
        """Also show some test input images on Tensorboard."""
        summary_op = super(TrainClassifyImages, self).get_test_summary_op()
        images_test_summary = tf.summary.image('test', self.images)
        self.test_summary_op = tf.summary.merge(
            [summary_op, images_test_summary])
        return self.test_summary_op


class EvaluateClassifyImages(EvaluateImages, EvaluateClassify):
    """Abstract class to evaluate an image classifier."""

    def compute(self, **kwargs):
        self.logits = self.compute_logits(
            self.images, self.dataset.num_classes, **kwargs)

    def last_step_log_info(self, sess, batch_size):
        """Give particular information for the last batch.

        For the last batch of input, we print ground truth labels
        and predictions of images. These images (and relative
        information) are only put on Tensorboard for visualization.
        However, only the infomation of at most 20 images are
        provided (if `batch_size` > 20 we take the first 20 images).
        """
        start_time = time.time()
        global_step_count, accuracy_rate, ac_summary, labels, \
            predictions, images = sess.run([
                self.global_step_op, self.accuracy,
                self.accuracy_summary,
                self.labels, self.predictions, self.images])
        time_elapsed = time.time() - start_time

        tf.logging.info(
            'global step %s: accurarcy: %.4f (%.2f sec/step)',
            global_step_count, accuracy_rate, time_elapsed)

        dataset = self.dataset
        true_names = [dataset.labels_to_names[i] for i in labels]
        predicted_names = [dataset.labels_to_names[i] for i in predictions]

        if batch_size > 20:
            batch_size = 20
        image_size = self.image_size

        tf.logging.info('Information for the last batch')
        tf.logging.info('Ground Truth: [%s]', true_names[:batch_size])
        tf.logging.info('Prediciotn: [%s]', predicted_names[:batch_size])

        if hasattr(self, 'fw'):
            with tf.name_scope('last_images'):
                for i in range(batch_size):
                    image_pl = tf.placeholder(
                        dtype=tf.float32,
                        shape=(1, image_size, image_size, self.channels))
                    image_summary = tf.summary.image(
                        'image_true_{}_predicted_{}'.format(
                            true_names[i], predicted_names[i]), image_pl)
                    self.fw.add_summary(
                        sess.run(image_summary,
                                 feed_dict={image_pl: [images[i]]}),
                        global_step=self.global_step_count)
            self.fw.add_summary(ac_summary, global_step=global_step_count)


class TrainClassifyInception(TrainClassifyImages):
    """Train the InceptionV4 model.

    Since the InceptionV4 network is quite huge and it takes very
    much time to train it from scratch (from days to weeks depending
    on used hardware), normally it's suggested to use a pre-trained
    model for fine-tuning. Here we choose to fine tune the last layer
    of the core inception and the logit layer by default. This
    can be changed by giving the `trainable_scopes` argument.

    I take directly the implementation of InceptionV4 on
    the github directory:
    https://github.com/tensorflow/models/tree/master/slim

    You can also find the checkpoint of the model (download it and
    put its directory as the `checkpoint_dir` argument) and more
    details about the inception architecture on this page.
    """

    @property
    def default_trainable_scopes(self):
        return ['InceptionV4/Mixed_7d', 'InceptionV4/Logits']

    # An argument scope is already contained in the inception
    # architecture so we must feed `is_training` with the correct value.
    def compute_logits(self, inputs, num_classes, **kwargs):
        logits, _ = inception_v4.inception_v4(
            inputs, num_classes=num_classes,
            is_training=self.batch_stat, **kwargs)
        return logits

    def get_init_fn(self, checkpoint_dirs):
        """Restore the pre-trained model from the checkpoint."""

        checkpoint_exclude_scopes = [
            'InceptionV4/Logits', 'InceptionV4/AuxLogits']
        variables_to_restore = self.get_variables_to_restore(
            scopes=None, exclude=checkpoint_exclude_scopes)

        assert len(checkpoint_dirs) == 1
        checkpoint_path = tf.train.latest_checkpoint(checkpoint_dirs[0])
        if checkpoint_path is None:
            checkpoint_path = os.path.join(
                checkpoint_dirs[0], 'inception_v4.ckpt')

        return slim.assign_from_checkpoint_fn(
            checkpoint_path, variables_to_restore)


def fine_tune_inception(tfrecord_dir,
                        checkpoint_dir,
                        log_dir,
                        number_of_steps=None,
                        image_size=299,
                        **kwargs):
    """A convenient function to easily do fine-tuning for inception model."""
    fine_tune = TrainClassifyInception(image_size)
    for key in kwargs.copy():
        if hasattr(fine_tune, key):
            setattr(fine_tune, key, kwargs[key])
            del kwargs[key]
    fine_tune.train(
        tfrecord_dir, checkpoint_dir, log_dir,
        number_of_steps=number_of_steps, **kwargs)


class EvaluateClassifyInception(EvaluateClassifyImages):
    """Evaluate the trained InceptionV4 model."""

    def compute_logits(self, inputs, num_classes):
        logits, _ = inception_v4.inception_v4(
            inputs, num_classes=num_classes, is_training=False)
        return logits


class TrainClassifyImagesCNN(TrainClassifyCNN, TrainClassifyImages):
    """Train a CNN to classify image.

    If `self.CNN_architecture` is `None` in fact we train only a perceptron.
    Several CNN architectures can be found in the `CNN_architecture.py` file.
    """
    pass


class EvaluateClassifyImagesCNN(EvaluateClassifyCNN, EvaluateClassifyImages):
    """Evaluate a trained CNN that is used to classify image.

    If `self.CNN_architecture` is `None` in fact it's just a perceptron.
    Several CNN architectures can be found in the `CNN_architecture.py` file.
    """
    pass


class TrainClassifyImagesCAE(TrainClassifyImages):
    """Train a perceptron from some high-level features learned by a CAE.

    Convolutional auto-encoders (CAE) are supposes to be able to
    learn meaningful representation of (image) input in an unsupervised
    manner. We first train a CAE on the data (see `CAE_routines.py`)
    and then we take activation values of some hidden layer of the network
    as a high-level representation of the image. This class only trains
    a perceptron on this representation.

    However, with the arguments `trainable_scopes` and
    `use_default_trainable_scopes` one can decide to train more than
    just the final perceptron part in a supervised way.

    Several CAE architectures can be found in the `CAE_architecture.py` file.
    """

    def __init__(self, CAE_architecture, endpoint='Middle', **kwargs):
        """Give the used CAE architecture and the representation layer.

        Args:
            CAE_architecture: The CAE artictecture to compute the high-level
                representation of image.
            endpoint: Indicate the layer of the network that is used
                as the high-level representation of image. It becomes
                then the input of the perceptron for classifaction.
            **kwargs: Other arguments used by the superclass.
        """
        super(TrainClassifyImagesCAE, self).__init__(**kwargs)
        self.CAE_architecture = CAE_architecture
        self.endpoint = endpoint

    @property
    def default_trainable_scopes(self):
        """Only train a perceptron from the image representation by default."""
        return ['Logits']

    def compute_logits(self, inputs, num_classes,
                       dropout_keep_prob=0.8, do_avg=False):
        """Compute logits using perceptron from CAE representation.

        Args:
            inputs: The input(s) of the network.
            num_classes: The number of classes to be classified.
                This is also the number of neurons in the output layer.
            dropout_keep_prob: Dropout is used just before the
                last fully connected layer which computes logits.
                This is the probability that each individual neuron
                value is kept. Must be in the interval (0, 1].
            do_avg: Whether to do an average pooling before the flatten
                layer. This is used historically for test purpose and it
                should better be `False`.
        """
        net = self.CAE_architecture(
            inputs, dropout_keep_prob=1, final_endpoint=self.endpoint)
        net = slim.dropout(net, dropout_keep_prob, scope='PreLogitsDropout')
        if do_avg:
            net = slim.avg_pool2d(
                net, net.get_shape()[1:3], padding='VALID',
                scope='PreLogitsAvgPool')
        print('Representation shape', net.get_shape())
        net = slim.flatten(net, scope='PreLogitsFlatten')
        logits = slim.fully_connected(
            net, num_classes, activation_fn=None, scope='Logits')
        return logits

    def get_init_fn(self, checkpoint_dirs):
        """Restore the trained CAE model to compute high-level
        image features."""
        assert len(checkpoint_dirs) == 1
        checkpoint_path = tf.train.latest_checkpoint(checkpoint_dirs[0])
        assert checkpoint_path is not None
        variables_to_restore = self.get_variables_to_restore(['CAE'])
        return slim.assign_from_checkpoint_fn(
            checkpoint_path, variables_to_restore)


class EvaluateClassifyImagesCAE(EvaluateClassifyImages):
    """Evaluate the trained perceptron built on CAE representation.

    This is the evaluatio part of `TrainClassifyImagesCAE`.
    """

    def __init__(self, CAE_architecture, endpoint='Middle', **kwargs):
        """Give the used CAE architecture and the representation layer.

        Args:
            CAE_architecture: The CAE artictecture to compute the high-level
                representation of image.
            endpoint: Indicate the layer of the network that is used
                as the high-level representation of image. It becomes
                then the input of the perceptron for classifaction.
        """
        super(EvaluateClassifyImagesCAE, self).__init__(**kwargs)
        self.CAE_architecture = CAE_architecture
        self.endpoint = endpoint

    def compute_logits(self, inputs, num_classes):
        """Compute logits using perceptron from CAE representation.

        Args:
            inputs: The input(s) of the network.
            num_classes: The number of classes to be classified.
                This is also the number of neurons in the output layer.
        """
        net = self.CAE_architecture(
            inputs, final_endpoint=self.endpoint)
        net = slim.flatten(net, scope='PreLogitsFlatten')
        self.logits = slim.fully_connected(
            net, num_classes, activation_fn=None, scope='Logits')
        return self.logits
