"""Basic classes to deal with image input.

We define `TrainImages`, `EvaluateImages` and `VisualizeImages`
that inherits respectively from `Train`, `Evaluate` and
`Visualize`. Note that `VisualizeImages` is different from the
others. It's less general but it can be used directly.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from routines.train import Train
from routines.evaluate import Evaluate
from routines.visualize import Visualize
from data.images import load_batch_images, get_split_images

slim = tf.contrib.slim


class TrainImages(Train):
    """Subclass of `Train` that reads image data."""

    def __init__(self, image_size=299, channels=3, **kwargs):
        """One should define some parameters for input images.

        Args:
            image_size: The image is of size image_size x image_size.
                For historical reason the input image has always
                the same height and width.
            channels: The number of channels of image. 1 for a grayscale
                image and 3 for a RGB image.
            **kwargs: Other arguments used by the superclass.
        """
        super(TrainImages, self).__init__(**kwargs)
        self.image_size = image_size
        self.channels = channels

    def get_data(self, tfrecord_dir, batch_size):
        self.dataset_train = get_split_images(
            'train', tfrecord_dir, channels=self.channels)
        self.images_train, self.labels_train = load_batch_images(
            self.dataset_train, height=self.image_size,
            width=self.image_size, batch_size=batch_size)
        self.dataset_test = get_split_images(
            'validation', tfrecord_dir, channels=self.channels)
        self.images_test, self.labels_test = load_batch_images(
            self.dataset_test, height=self.image_size,
            width=self.image_size, batch_size=batch_size)
        return self.dataset_train

    def decide_used_data(self):
        self.images = tf.cond(
            self.training, lambda: self.images_train,
            lambda: self.images_test)
        self.labels = tf.cond(
            self.training, lambda: self.labels_train,
            lambda: self.labels_test)


class EvaluateImages(Evaluate):
    """Subclass of `Evaluate` that reads image data."""

    def __init__(self, image_size=299, channels=3):
        """One should define some parameters for input images.

        Args:
            image_size: The image is of size image_size x image_size.
                For historical reason the input image has always
                the same height and width.
            channels: The number of channels of image. 1 for a grayscale
                image and 3 for a RGB image.
        """
        self.image_size = image_size
        self.channels = channels

    def get_data(self, split_name, tfrecord_dir, batch_size, shuffle):
        self.dataset = get_split_images(
            split_name, tfrecord_dir, channels=self.channels)
        if batch_size is None:
            batch_size = self.dataset.num_samples
        self.images, self.labels = load_batch_images(
            self.dataset, height=self.image_size,
            width=self.image_size, batch_size=batch_size, shuffle=shuffle)
        return self.dataset


class VisualizeImages(Visualize):
    """Subclass of `Visualize` to visualize image concerning information.

    It's mainly for visualizing the representations learned by CAEs
    (so the default endpoint of `compute` is 'Middle'.

    However, it can be equally used to visualize raw data distribution
    or rerpresentations learned by CNNs."""

    def __init__(self, architecture, image_size=299, channels=3):
        """Defeine parameters for input images and the used architecture.

        Args:
            architecture: The architecture to compute representations.
            image_size: The image is of size image_size x image_size.
                For historical reason the input image has always
                the same height and width.
            channels: The number of channels of image. 1 for a grayscale
                image and 3 for a RGB image.
        """
        self.architecture = architecture
        self.image_size = image_size
        self.channels = channels

    def get_data(self, split_name, tfrecord_dir, batch_size, shuffle):
        self.dataset = get_split_images(
            split_name, tfrecord_dir, channels=self.channels)
        if batch_size is None:
            batch_size = self.dataset.num_samples
        self.images, self.labels = load_batch_images(
            self.dataset, height=self.image_size,
            width=self.image_size, batch_size=batch_size, shuffle=shuffle)
        return self.dataset

    def compute(self, endpoint='Middle', do_avg=False):
        """Compute the representation, save it in a variable and define the
        saver to save this variable.

        Args:
            endpoint: The endpoint passed to `self.architecture`. Use `None`
                if no endpoint argument should be given.
            do_avg: Whether to do average pooling to compute the
                representation, just ignore this argument.
        """
        if self.architecture is None:
            self.representations = self.images
        elif endpoint is None:
            self.representations = self.architecture(self.images)
        else:
            self.representations = self.architecture(
                self.images, final_endpoint=endpoint)
        if do_avg:
            self.representations = slim.avg_pool2d(
                self.representations, self.representations.get_shape()[1:3],
                stride=1, scope='AvgPool')
        self.representations = slim.flatten(
            self.representations, scope='Flatten')

        self.repr_var = tf.Variable(
            tf.zeros_like(self.representations), name='Representation')
        self.assign = tf.assign(self.repr_var, self.representations)
        self.saver_repr = tf.train.Saver([self.repr_var])

    def config_embedding(self, sess, log_dir):
        """Configurations for embedding visualization.

        After running the necessary `Tensor` operations (assigning the
        computed values to a variable for saving), we save this variable
        to checkpoint, write metadata, and configures the embedding by
        linking it to the variable and the metafile. Several embeddings
        can be defined in the same time.
        """
        _, labels = sess.run([self.assign, self.labels])
        self.saver_repr.save(sess, os.path.join(log_dir, 'repr.ckpt'))

        metadata = os.path.abspath(os.path.join(log_dir, 'metadata.tsv'))
        with open(metadata, 'w') as metadata_file:
            metadata_file.write('index\tlabel\n')
            for index, label in enumerate(labels):
                metadata_file.write('%d\t[%d]\n' % (index, label))

        config = projector.ProjectorConfig()

        embedding = config.embeddings.add()
        embedding.tensor_name = self.repr_var.name
        embedding.metadata_path = metadata
        projector.visualize_embeddings(
            tf.summary.FileWriter(log_dir), config)
