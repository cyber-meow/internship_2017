from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import fnmatch
import tensorflow as tf

from data import dataset_utils

slim = tf.contrib.slim


_FILE_PATTERN = 'mfcc_lips_%s_*.tfrecord'

_ITEMS_TO_DESCRIPTIONS = {
    'mfcc': 'The mfcc data of the audio',
    'lips': 'The images for the video of lipreading',
    'label': 'A single integer representing the label',
}


def get_split_mfcc_lips(split_name,
                        tfrecord_dir,
                        file_pattern=None,
                        reader=None,
                        num_frames_audio=24,
                        num_frames_video=12):
    """Gets a dataset tuple with instructions for reading flowers.
    Args:
      split_name: A train_all/train_AT/train_UZ/validation split name.
      tfrecord_dir: The base directory of the dataset sources.
      file_pattern: The file pattern to use when matching the dataset sources.
        It is assumed that the pattern contains a '%s' string so that the
        split name can be inserted.
      reader: The TensorFlow reader type.

    Returns:
      A `Dataset` namedtuple.
    """

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
        'audio/mfcc': tf.FixedLenFeature((26, num_frames_audio), tf.float32),
        'video/data': tf.FixedLenFeature(
            (60, 80, num_frames_video), tf.float32),
        'label': tf.FixedLenFeature(
          (), tf.int64, default_value=tf.zeros((), dtype=tf.int64)),
    }

    items_to_handlers = {
        'mfcc': slim.tfexample_decoder.Tensor(
            'audio/mfcc', shape=(26, num_frames_audio, 1)),
        'video': slim.tfexample_decoder.Tensor(
            'video/data', shape=(60, 80, num_frames_video, 1)),
        'label': slim.tfexample_decoder.Tensor('label'),
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


def load_batch_mfcc_lips(dataset,
                         batch_size=32,
                         common_queue_capacity=800,
                         common_queue_min=400,
                         shuffle=True,
                         is_training=True):
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
    mfcc, video, label = data_provider.get(['mfcc', 'video', 'label'])

    if is_training:

        transformed_images = []

        delta = tf.random_uniform((), -1, 1)
        contrast_factor = tf.random_uniform((), 0.2, 1.8)
        bbox_begin, bbox_end, _ = tf.image.sample_distorted_bounding_box(
            [60, 80, 1], [[[0, 0, 1, 1]]], area_range=[0.8, 1])

        for i in range(video.get_shape()[2]):
            image = video[:, :, i, :]
            image = tf.image.adjust_brightness(image, delta)
            image = tf.image.adjust_contrast(image, contrast_factor)
            image = tf.slice(image, bbox_begin, bbox_end)
            image.set_shape([None, None, 1])
            image = tf.image.resize_images(image, [60, 80])
            transformed_images.append(tf.expand_dims(image, 2))
        video = tf.concat(transformed_images, 2)

    # Batch it up.
    mfccs, videos, labels = tf.train.batch(
        [mfcc, video, label],
        batch_size=batch_size,
        num_threads=1,
        capacity=2*batch_size)

    return mfccs, videos, labels
