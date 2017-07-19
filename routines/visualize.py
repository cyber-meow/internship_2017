from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from data.images import load_batch_images, get_split_images
from data.color_depth import load_batch_color_depth, get_split_color_depth
from nets_base.arg_scope import nets_arg_scope

slim = tf.contrib.slim


class Visualize(object):

    def visualize(self,
                  tfrecord_dir,
                  checkpoint_dirs,
                  log_dir=None,
                  batch_size=300,
                  split_name='train',
                  use_batch_norm=True,
                  use_batch_stat=False,
                  **kwargs):

        if log_dir is not None and not tf.gfile.Exists(log_dir):
            tf.gfile.MakeDirs(log_dir)

        if (checkpoint_dirs is not None
                and not isinstance(checkpoint_dirs, (tuple, list))):
            checkpoint_dirs = [checkpoint_dirs]

        with tf.Graph().as_default():

            with tf.name_scope('Data_provider'):
                self.get_data(split_name, tfrecord_dir, batch_size)

            with slim.arg_scope(self.used_arg_scope(
                    use_batch_stat, use_batch_norm)):
                self.compute(**kwargs)

            with tf.Session() as sess:
                with slim.queues.QueueRunners(sess):
                    self.init_model(sess, checkpoint_dirs)
                    self.config_embedding(sess, log_dir)

    def used_arg_scope(self, use_batch_stat, use_batch_norm):
        return nets_arg_scope(
            is_training=use_batch_stat, use_batch_norm=use_batch_norm)

    def init_model(self, sess, checkpoint_dirs):
        if checkpoint_dirs is not None:
            assert len(checkpoint_dirs) == 1
            checkpoint_path = tf.train.latest_checkpoint(checkpoint_dirs[0])
            saver = tf.train.Saver(tf.model_variables())
            saver.restore(sess, checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())


class VisualizeImages(Visualize):

    def __init__(self, structure, image_size=299, channels=3):
        self.structure = structure
        self.image_size = image_size
        self.channels = channels

    def get_data(self, split_name, tfrecord_dir, batch_size):
        self.dataset = get_split_images(
            split_name, tfrecord_dir, channels=self.channels)
        self.images, self.labels = load_batch_images(
            self.dataset, height=self.image_size,
            width=self.image_size, batch_size=batch_size)
        return self.dataset

    def compute(self, endpoint='Middle', do_avg=False):

        if self.structure is None:
            self.representations = self.images
        elif endpoint is None:
            self.representations, _ = self.structure(self.images)
        else:
            self.representations, _ = self.structure(
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

        _, lbs = sess.run([self.assign, self.labels])
        self.saver_repr.save(sess, os.path.join(log_dir, 'repr.ckpt'))

        metadata = os.path.abspath(os.path.join(log_dir, 'metadata.tsv'))
        with open(metadata, 'w') as metadata_file:
            metadata_file.write('index\tlabel\n')
            for index, label in enumerate(lbs):
                metadata_file.write('%d\t%d\n' % (index, label))

        config = projector.ProjectorConfig()

        embedding = config.embeddings.add()
        embedding.tensor_name = self.repr_var.name
        embedding.metadata_path = metadata
        projector.visualize_embeddings(
            tf.summary.FileWriter(log_dir), config)


class VisualizeColorDepth(Visualize):

    def __init__(self, structure, image_size=299,
                 color_channels=3, depth_channels=3):
        self.structure = structure
        self.image_size = image_size
        self.color_channels = color_channels
        self.depth_channels = depth_channels

    def get_data(self, split_name, tfrecord_dir, batch_size):
        self.dataset = get_split_color_depth(
            split_name,
            tfrecord_dir,
            color_channels=self.color_channels,
            depth_channels=self.depth_channels)
        self.images_color, self.images_depth, self.labels = \
            load_batch_color_depth(
                self.dataset, height=self.image_size,
                width=self.image_size, batch_size=batch_size)
        return self.dataset


def visualize(visualize_class,
              used_structure,
              tfrecord_dir,
              checkpoint_dirs,
              log_dir,
              **kwargs):
    visualize_instance = visualize_class(used_structure)
    for key in kwargs.copy():
        if hasattr(visualize_instance, key):
            setattr(visualize_instance, key, kwargs[key])
            del kwargs[key]
    visualize_instance.visualize(
        tfrecord_dir, checkpoint_dirs, log_dir, **kwargs)
