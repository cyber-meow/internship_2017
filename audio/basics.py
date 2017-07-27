from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from data.mfcc import load_batch_mfcc, get_split_mfcc
from routines.train import Train
from routines.evaluate import Evaluate
from routines.visualize import Visualize

slim = tf.contrib.slim


class TrainAudio(Train):

    def get_data(self, tfrecord_dir, batch_size):
        self.dataset_train = get_split_mfcc('train', tfrecord_dir)
        self.mfccs_train, self.labels_train = load_batch_mfcc(
            self.dataset_train, batch_size=batch_size)
        self.dataset_test = get_split_mfcc('validation', tfrecord_dir)
        self.mfccs_test, self.labels_test = load_batch_mfcc(
            self.dataset_test, batch_size=batch_size)
        return self.dataset_train


class EvaluateAudio(Evaluate):

    def get_data(self, split_name, tfrecord_dir, batch_size, shuffle):
        self.dataset = get_split_mfcc(split_name, tfrecord_dir)
        self.mfccs, self.labels = load_batch_mfcc(
            self.dataset, batch_size=batch_size, shuffle=shuffle)
        return self.dataset


# It didn't work
class VisualizeMfccs(Visualize):

    def get_data(self, split_name, tfrecord_dir, batch_size):
        self.dataset = get_split_mfcc(split_name, tfrecord_dir)
        self.mfccs, self.labels = load_batch_mfcc(
            self.dataset, batch_size=batch_size)
        return self.dataset

    def compute(self, endpoint='Middle', do_avg=False):

        self.representations = self.mfccs
        self.representations = slim.flatten(
            self.representations, scope='Flatten')

        self.repr_var = tf.Variable(
            tf.zeros_like(self.representations), name='Representation')
        self.assign = tf.assign(self.repr_var, self.representations)
        self.saver_repr = tf.train.Saver([self.repr_var])

    def config_embedding(self, sess, log_dir):

        _, lbs = sess.run([self.mfccs, self.labels])
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
