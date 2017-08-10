"""Read from TFRecords of the Montalbano gesture dataset.

This file has never been tested since I haven't succeeded in
converting the TFRecords for this dataset.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import fnmatch
import tensorflow as tf

from data import dataset_utils

slim = tf.contrib.slim


_FILE_PATTERN = 'montalbano_%s_*.tfrecord'

_ITEMS_TO_DESCRIPTIONS = {
    'color': 'Intensity videos.',
    'depth': 'Depth videos.',
    'label': 'A single integer representing the label',
}


def get_split_montalbano(split_name,
                         tfrecord_dir,
                         file_pattern=None,
                         reader=None):

    if not file_pattern:
        file_pattern = _FILE_PATTERN

    num_samples = 0
    tfrecords_to_count = [
        os.path.join(tfrecord_dir, file)
        for file in os.listdir(tfrecord_dir)
        if fnmatch.fnmatch(file, file_pattern % split_name)]
    for tfrecord_file in tfrecords_to_count:
        for record in tf.python_io.tf_record_iterator(tfrecord_file):
            num_samples += 1

    file_pattern = os.path.join(tfrecord_dir, file_pattern % split_name)

    if reader is None:
        reader = tf.TFRecordReader

    keys_to_features = {
        'video/color/data': tf.VarLenFeature(tf.float32),
        'video/color/shape': tf.FixedLenFeature([4], tf.int64),
        'video/depth/data': tf.VarLenFeature(tf.float32),
        'video/depth/shape': tf.FixedLenFeature([4], tf.int64),
        'video/label': tf.FixedLenFeature(
          (), tf.int64, default_value=tf.zeros((), dtype=tf.int64)),
    }

    items_to_handlers = {
        'color': slim.tfexample_decoder.Tensor(
            'video/color/data', shape_key='video/color/shape'),
        'depth': slim.tfexample_decoder.Tensor(
            'video/depth/data', shape_key='video/depth/shape'),
        'label': slim.tfexample_decoder.Tensor('video/label'),
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

    labels_to_names = None
    num_classes = None

    if dataset_utils.has_labels(tfrecord_dir):
        labels_to_names = dataset_utils.read_label_file(tfrecord_dir)
        num_classes = len(labels_to_names)

    return slim.dataset.Dataset(
        data_sources=file_pattern,
        reader=reader,
        decoder=decoder,
        num_samples=num_samples,
        items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
        num_classes=num_classes,
        labels_to_names=labels_to_names)


def load_batch_montalbano(dataset,
                          batch_size=32,
                          common_queue_capacity=800,
                          common_queue_min=400,
                          shuffle=True):

    data_provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset, common_queue_capacity=common_queue_capacity,
        common_queue_min=common_queue_min, shuffle=shuffle)
    color_video, depth_video, label = \
        data_provider.get(['color', 'depth', 'label'])

    transformed_color_images = []

    for i in range(color_video.get_shape()[2]):
        color_image = color_video[:, :, i, :]
        color_image = tf.image.per_image_standardization(color_image)
        transformed_color_images.append(tf.expand_dims(color_image, 2))
    color_video = tf.concat(transformed_color_images, 2)

    transformed_depth_images = []

    for i in range(depth_video.get_shape()[2]):
        depth_image = depth_video[:, :, i, :]
        depth_image = tf.image.per_image_standardization(depth_image)
        transformed_depth_images.append(tf.expand_dims(depth_image, 2))
    depth_video = tf.concat(transformed_depth_images, 2)

    # Batch it up.
    color_videos, depth_videos, labels = tf.train.batch(
        [color_video, depth_video, label],
        batch_size=batch_size,
        num_threads=1,
        capacity=2*batch_size)

    return color_videos, depth_videos, labels
