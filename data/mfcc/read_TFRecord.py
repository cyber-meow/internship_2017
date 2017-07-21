from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import fnmatch
import tensorflow as tf

from data import dataset_utils

slim = tf.contrib.slim


_FILE_PATTERN = 'mfcc_%s_*.tfrecord'

_ITEMS_TO_DESCRIPTIONS = {
    'mfcc': 'The mfcc data of the audio',
    'label': 'A single integer representing the label',
}


def get_split_mfcc(split_name,
                   tfrecord_dir,
                   file_pattern=None,
                   reader=None,
                   num_frames=24):
    """Gets a dataset tuple with instructions for reading flowers.
    Args:
      split_name: A train/validation split name.
      tfrecord_dir: The base directory of the dataset sources.
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
        os.path.join(tfrecord_dir, file)
        for file in os.listdir(tfrecord_dir)
        if fnmatch.fnmatch(file, file_pattern % split_name)]
    for tfrecord_file in tfrecords_to_count:
        for record in tf.python_io.tf_record_iterator(tfrecord_file):
            num_samples += 1

    file_pattern = os.path.join(tfrecord_dir, file_pattern % split_name)

    if reader is None:
        reader = tf.TFRecordReader

    # Create the keys_to_features dictionary for the decoder
    keys_to_features = {
        'audio/mfcc': tf.FixedLenFeature((26, num_frames), tf.float32),
        'audio/label': tf.FixedLenFeature(
          (), tf.int64, default_value=tf.zeros((), dtype=tf.int64)),
    }

    items_to_handlers = {
        'mfcc': slim.tfexample_decoder.Tensor(
            'audio/mfcc', shape=(26, num_frames, 1)),
        'label': slim.tfexample_decoder.Tensor('audio/label'),
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


def load_batch_mfcc(dataset,
                    batch_size=32,
                    common_queue_capacity=800,
                    common_queue_min=400,
                    shuffle=True):
    """Loads a single batch of data.

    Args:
      dataset: The dataset to load
      batch_size: The number of images in the batch
      common_queue_capacity, common_queue_min: Decide the shuffle degree
      shuffle: Whether to shuffle or not

    Returns:
      mfccs: A Tensor of size [batch_size, feature_len, time_frames, 1]
      labels: A Tensor of size [batch_size], whose values range between
        0 and dataset.num_classes.
    """
    data_provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset, common_queue_capacity=common_queue_capacity,
        common_queue_min=common_queue_min, shuffle=shuffle)
    mfcc, label = data_provider.get(['mfcc', 'label'])

    # Batch it up.
    mfccs, labels = tf.train.batch(
        [mfcc, label],
        batch_size=batch_size,
        num_threads=1,
        capacity=2*batch_size)

    return mfccs, labels
