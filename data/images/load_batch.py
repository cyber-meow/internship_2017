from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from nets_base import inception_preprocessing

slim = tf.contrib.slim


def load_batch(dataset, batch_size=32, height=299, width=299):
    """Loads a single batch of data.

    Args:
      dataset: The dataset to load.
      batch_size: The number of images in the batch.
      height: The size of each image after preprocessing.
      width: The size of each image after preprocessing.

    Returns:
      images: A Tensor of size [batch_size, height, width, 3],
        image samples that have been preprocessed.
      labels: A Tensor of size [batch_size], whose values range between
        0 and dataset.num_classes.
    """
    data_provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset, common_queue_capacity=800,
        common_queue_min=400)
    image_raw, label = data_provider.get(['image', 'label'])

    # Preprocess image for usage by Inception.
    image = inception_preprocessing.preprocess_image(
        image_raw, height, width, is_training=False)
    image = tf.image.adjust_contrast(image, 10)

    # Batch it up.
    images, labels = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=1,
        capacity=2*batch_size)

    return images, labels
