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

    def __init__(self, num_frames=24,
                 file_pattern='mfcc_%s_*.tfrecord', **kwargs):
        self.num_frames = num_frames
        self.file_pattern = file_pattern
        super(TrainAudio, self).__init__(**kwargs)

    def get_data(self, tfrecord_dir, batch_size):
        self.dataset_train = get_split_mfcc(
            'train', tfrecord_dir,
            file_pattern=self.file_pattern, num_frames=self.num_frames)
        self.mfccs_train, self.labels_train = load_batch_mfcc(
            self.dataset_train, batch_size=batch_size)
        self.dataset_test = get_split_mfcc(
            'validation', tfrecord_dir,
            file_pattern=self.file_pattern, num_frames=self.num_frames)
        self.mfccs_test, self.labels_test = load_batch_mfcc(
            self.dataset_test, batch_size=batch_size)
        return self.dataset_train

    def decide_used_data(self):
        self.mfccs = tf.cond(
            self.training, lambda: self.mfccs_train, lambda: self.mfccs_test)
        self.labels = tf.cond(
            self.training, lambda: self.labels_train, lambda: self.labels_test)


class EvaluateAudio(Evaluate):

    def __init__(self, num_frames=24,
                 file_pattern='mfcc_%s_*.tfrecord', **kwargs):
        self.num_frames = num_frames
        self.file_pattern = file_pattern
        super(EvaluateAudio, self).__init__(**kwargs)

    def get_data(self, split_name, tfrecord_dir, batch_size, shuffle):
        self.dataset = get_split_mfcc(
            split_name, tfrecord_dir,
            file_pattern=self.file_pattern, num_frames=self.num_frames)
        if batch_size is None:
            batch_size = self.dataset.num_samples
        self.mfccs, self.labels = load_batch_mfcc(
            self.dataset, batch_size=batch_size, shuffle=shuffle)
        return self.dataset


def delta(coefs, N=2):
    """
    delta: A tensor of shape [batch, feature_len, num_frames, 1]
    """
    res = tf.zeros_like(coefs)
    for n in range(1, N+1):
        minus_part = tf.concat([
                tf.tile(tf.expand_dims(coefs[:, :, 0, :], 2), [1, 1, n, 1]),
                coefs[:, :, :-n, :]
            ], 2)
        plus_part = tf.concat([
                coefs[:, :, n:, :],
                tf.tile(tf.expand_dims(coefs[:, :, -1, :], 2), [1, 1, n, 1])
            ], 2)
        res += n * (plus_part-minus_part)
    res /= 2*sum([n**2 for n in range(1, N+1)])
    return res


class VisualizeAudio(Visualize):

    def __init__(self, structure, num_frames=24):
        self.structure = structure
        self.num_frames = num_frames

    def get_data(self, split_name, tfrecord_dir, batch_size, shuffle):
        self.dataset = get_split_mfcc(
            split_name, tfrecord_dir, num_frames=self.num_frames)
        if batch_size is None:
            batch_size = self.dataset.num_samples
        self.mfccs, self.labels = load_batch_mfcc(
            self.dataset, batch_size=batch_size, shuffle=shuffle)
        return self.dataset

    def compute(self, endpoint=None, use_delta=False):

        if use_delta:
            mfcc_deltas = delta(self.mfccs)
            delta_deltas = delta(mfcc_deltas)
            self.mfccs = tf.concat(
                [self.mfccs, mfcc_deltas, delta_deltas], axis=3)

        if self.structure is None:
            self.representations = self.mfccs
        elif endpoint is None:
            self.representations = self.structure(self.mfccs)
        else:
            self.representations = self.structure(
                self.mfccs, final_endpoint=endpoint)
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
