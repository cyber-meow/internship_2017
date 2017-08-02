from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from routines.train import Train
from routines.evaluate import Evaluate
from routines.visualize import Visualize
from data.color_depth import load_batch_color_depth, get_split_color_depth


class TrainColorDepth(Train):

    def __init__(self, image_size=299,
                 color_channels=3, depth_channels=3, **kwargs):
        super(TrainColorDepth, self).__init__(**kwargs)
        self.image_size = image_size
        self.color_channels = color_channels
        self.depth_channels = depth_channels

    def get_data(self, tfrecord_dir, batch_size):

        self.dataset_train = get_split_color_depth(
            'train',
            tfrecord_dir,
            color_channels=self.color_channels,
            depth_channels=self.depth_channels)

        self.images_color_train, self.images_depth_train, self.labels_train = \
            load_batch_color_depth(
                self.dataset_train, height=self.image_size,
                width=self.image_size, batch_size=batch_size)

        self.dataset_test = get_split_color_depth(
            'validation',
            tfrecord_dir,
            color_channels=self.color_channels,
            depth_channels=self.depth_channels)

        self.images_color_test, self.images_depth_test, self.labels_test = \
            load_batch_color_depth(
                self.dataset_test, height=self.image_size,
                width=self.image_size, batch_size=batch_size)

        return self.dataset_train

    def decide_used_data(self):
        self.images_color = tf.cond(
            self.training, lambda: self.images_color_train,
            lambda: self.images_color_test)
        self.images_depth = tf.cond(
            self.training, lambda: self.images_depth_train,
            lambda: self.images_depth_test)
        self.labels = tf.cond(
            self.training, lambda: self.labels_train,
            lambda: self.labels_test)


class EvaluateColorDepth(Evaluate):

    def __init__(self, image_size=299, color_channels=3, depth_channels=3):
        self.image_size = image_size
        self.color_channels = color_channels
        self.depth_channels = depth_channels

    def get_data(self, split_name, tfrecord_dir, batch_size, shuffle):
        self.dataset = get_split_color_depth(
            split_name,
            tfrecord_dir,
            color_channels=self.color_channels,
            depth_channels=self.depth_channels)
        if batch_size is None:
            batch_size = self.dataset.num_samples
        self.images_color, self.images_depth, self.labels = \
            load_batch_color_depth(
                self.dataset, height=self.image_size,
                width=self.image_size, batch_size=batch_size, shuffle=shuffle)
        return self.dataset


class VisualizeColorDepth(Visualize):

    def __init__(self, structure, image_size=299,
                 color_channels=3, depth_channels=3):
        self.structure = structure
        self.image_size = image_size
        self.color_channels = color_channels
        self.depth_channels = depth_channels

    def get_data(self, split_name, tfrecord_dir, batch_size, shuffle):
        self.dataset = get_split_color_depth(
            split_name,
            tfrecord_dir,
            color_channels=self.color_channels,
            depth_channels=self.depth_channels)
        self.images_color, self.images_depth, self.labels = \
            load_batch_color_depth(
                self.dataset, height=self.image_size,
                width=self.image_size, batch_size=batch_size, shuffle=shuffle)
        return self.dataset
