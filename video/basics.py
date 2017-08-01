from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from data.lips import load_batch_lips, get_split_lips
from routines.train import Train
from routines.evaluate import Evaluate
from routines.visualize import Visualize

slim = tf.contrib.slim


class TrainVideo(Train):

    def get_data(self, tfrecord_dir, batch_size):
        self.dataset_train = get_split_lips('train', tfrecord_dir)
        self.videos_train, self.labels_train = load_batch_lips(
            self.dataset_train, batch_size=batch_size)
        self.dataset_test = get_split_lips('validation', tfrecord_dir)
        self.videos_test, self.labels_test = load_batch_lips(
            self.dataset_test, batch_size=batch_size, is_training=False)
        return self.dataset_train

    def decide_used_data(self):
        self.videos = tf.cond(
            self.training, lambda: self.videos_train, lambda: self.videos_test)
        self.labels = tf.cond(
            self.training, lambda: self.labels_train, lambda: self.labels_test)


class EvaluateVideo(Evaluate):

    def get_data(self, split_name, tfrecord_dir, batch_size, shuffle):
        self.dataset = get_split_lips(split_name, tfrecord_dir)
        if batch_size is None:
            batch_size = self.dataset.num_samples
        self.videos, self.labels = load_batch_lips(
            self.dataset, batch_size=batch_size,
            shuffle=shuffle, is_training=False)
        return self.dataset


class VisualizeVideo(Visualize):

    def __init__(self, structure, num_frames=24):
        self.structure = structure

    def get_data(self, split_name, tfrecord_dir, batch_size, shuffle):
        self.dataset = get_split_lips(split_name, tfrecord_dir)
        if batch_size is None:
            batch_size = self.dataset.num_samples
        self.videos, self.labels = load_batch_lips(
            self.dataset, batch_size=batch_size, shuffle=shuffle)
        return self.dataset

    def compute(self, endpoint=None, use_delta=False):

        if self.structure is None:
            self.representations = self.videos
        elif endpoint is None:
            self.representations = self.structure(self.videos)
        else:
            self.representations = self.structure(
                self.videos, final_endpoint=endpoint)
        self.representations = slim.flatten(
            self.representations, scope='Flatten')

        self.repr_var = tf.Variable(
            tf.zeros_like(self.representations), name='Representation')
        self.assign = tf.assign(self.repr_var, self.representations)
        self.saver_repr = tf.train.Saver([self.repr_var])

    def config_embedding(self, sess, log_dir):

        _, labels = sess.run([self.assign, self.labels])
        self.saver_repr.save(sess, os.path.join(log_dir, 'repr.ckpt'))

        metadata = os.path.abspath(os.path.join(log_dir, 'metadata.tsv'))
        with open(metadata, 'w') as metadata_file:
            metadata_file.write('index\tlabel\n')
            for index, label in enumerate(labels):
                metadata_file.write('%d\t%d\n' % (index, label))

        config = projector.ProjectorConfig()

        embedding = config.embeddings.add()
        embedding.tensor_name = self.repr_var.name
        embedding.metadata_path = metadata
        projector.visualize_embeddings(
            tf.summary.FileWriter(log_dir), config)
