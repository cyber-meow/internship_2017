from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from data.lips import load_batch_lips, get_split_lips
from routines.train import Train
from routines.evaluate import Evaluate


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
