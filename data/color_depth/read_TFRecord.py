"""Provides data for the dataset from TFRecords for color-depth image pairs"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import fnmatch
import tensorflow as tf

from data import dataset_utils
from nets_base import inception_preprocessing

slim = tf.contrib.slim


_FILE_PATTERN = 'color_depth_%s_*.tfrecord'

_ITEMS_TO_DESCRIPTIONS = {
    'image/color': 'A color image of varying size.',
    'image/depth': 'A depth map of varying size',
    'label': 'A single integer representing the label',
}


def get_split_color_depth(split_name,
                          tfrecord_dir,
                          file_pattern=None,
                          reader=None,
                          color_channels=3,
                          depth_channels=3):
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
        'image/color/encoded': tf.FixedLenFeature((), tf.string),
        'image/color/format': tf.FixedLenFeature((), tf.string),
        'image/depth/encoded': tf.FixedLenFeature((), tf.string),
        'image/depth/format': tf.FixedLenFeature((), tf.string),
        'image/class/label': tf.FixedLenFeature(
          (), tf.int64, default_value=tf.zeros((), dtype=tf.int64)),
    }

    items_to_handlers = {
        'image/color': slim.tfexample_decoder.Image(
            image_key='image/color/encoded',
            format_key='image/color/format',
            channels=color_channels),
        'image/depth': slim.tfexample_decoder.Image(
            image_key='image/depth/encoded',
            format_key='image/depth/format',
            channels=depth_channels),
        'label': slim.tfexample_decoder.Tensor('image/class/label'),
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


def load_batch_color_depth(dataset,
                           batch_size=32,
                           height=299,
                           width=299,
                           common_queue_capacity=800,
                           common_queue_min=400,
                           shuffle=True):
    """Loads a single batch of data.

    Args:
      dataset: The dataset to load
      batch_size: The number of images in the batch
      height: The size of each image after preprocessing
      width: The size of each image after preprocessing
      common_queue_capacity, common_queue_min: Decide the shuffle degree
      shuffle: Whether to shuffle or not

    Returns:
      images_color: A Tensor of size [batch_size, height, width, 3],
        RGB image samples that have been preprocessed.
      images_depth: A Tensor of size [batch_size, height, width, 3],
        depth image samples that have been preprocessed.
      labels: A Tensor of size [batch_size], whose values range between
        0 and dataset.num_classes.
    """
    data_provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset, common_queue_capacity=common_queue_capacity,
        common_queue_min=common_queue_min)
    image_color, image_depth, label = data_provider.get(
        ['image/color', 'image/depth', 'label'])

    # Preprocess image for usage by Inception.
    image_color = inception_preprocessing.preprocess_image(
        image_color, height, width, is_training=False)
    image_color = tf.image.per_image_standardization(image_color)
    image_depth = inception_preprocessing.preprocess_image(
        image_depth, height, width, is_training=False)
    image_depth = tf.image.per_image_standardization(image_depth)

    # Batch it up.
    images_color, images_depth, labels = tf.train.batch(
        [image_color, image_depth, label],
        batch_size=batch_size,
        num_threads=1,
        capacity=2*batch_size)

    return images_color, images_depth, labels
