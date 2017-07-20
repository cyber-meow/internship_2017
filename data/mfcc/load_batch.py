from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


def load_batch_mfcc(dataset,
                    batch_size=32,
                    common_queue_capacity=800,
                    common_queue_min=400):
    """Loads a single batch of data.

    Args:
      dataset: The dataset to load.
      batch_size: The number of images in the batch.
      common_queue_capacity, common_queue_min: decide the shuffle degree

    Returns:
      mfccs: A Tensor of size [batch_size, feature_len, time_frames, 1]
      labels: A Tensor of size [batch_size], whose values range between
        0 and dataset.num_classes.
    """
    data_provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset, common_queue_capacity=common_queue_capacity,
        common_queue_min=common_queue_min)
    mfcc, label = data_provider.get(['mfcc', 'label'])

    # Batch it up.
    mfccs, labels = tf.train.batch(
        [mfcc, label],
        batch_size=batch_size,
        num_threads=1,
        capacity=2*batch_size)

    return mfccs, labels
