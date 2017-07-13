"""Provides data for the dataset from TFRecords for general images"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import fnmatch
import tensorflow as tf

from data import dataset_utils

slim = tf.contrib.slim


_FILE_PATTERN = 'data_%s_*.tfrecord'

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying size.',
    'label': 'A single integer between 0 and 10',
}


def get_split(split_name, dataset_dir,
              file_pattern=None, reader=None, channels=3):
    """Gets a dataset tuple with instructions for reading flowers.
    Args:
      split_name: A train/validation split name.
      dataset_dir: The base directory of the dataset sources.
      file_pattern: The file pattern to use when matching the dataset sources.
        It is assumed that the pattern contains a '%s' string so that the
        split name can be inserted.
      reader: The TensorFlow reader type.

    Returns:
      A `Dataset` namedtuple.

    Raises:
      ValueError: if `split_name` is not a valid train/validation split.
    """
    if split_name not in ['train', 'validation']:
        raise ValueError(
            'The split_name %s is not recognized.' % (split_name)
            + 'Please input either train or validation as the split_name')

    if not file_pattern:
        file_pattern = _FILE_PATTERN

    num_samples = 0
    tfrecords_to_count = [
        os.path.join(dataset_dir, file)
        for file in os.listdir(dataset_dir)
        if fnmatch.fnmatch(file, file_pattern % split_name)]
    for tfrecord_file in tfrecords_to_count:
        for record in tf.python_io.tf_record_iterator(tfrecord_file):
            num_samples += 1

    file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

    if reader is None:
        reader = tf.TFRecordReader

    # Create the keys_to_features dictionary for the decoder
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string),
        'image/format': tf.FixedLenFeature((), tf.string),
        'image/class/label': tf.FixedLenFeature(
          (), tf.int64, default_value=tf.zeros((), dtype=tf.int64)),
    }

    items_to_handlers = {
        'image': slim.tfexample_decoder.Image(channels=channels),
        'label': slim.tfexample_decoder.Tensor('image/class/label'),
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

    labels_to_names = None
    if dataset_utils.has_labels(dataset_dir):
        labels_to_names = dataset_utils.read_label_file(dataset_dir)

    return slim.dataset.Dataset(
        data_sources=file_pattern,
        reader=reader,
        decoder=decoder,
        num_samples=num_samples,
        items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
        num_classes=len(labels_to_names),
        labels_to_names=labels_to_names)
