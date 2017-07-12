from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from nets_base import inception_preprocessing

slim = tf.contrib.slim


def load_batch_color_depth(dataset,
                           batch_size=32,
                           height=299,
                           width=299,
                           common_queue_capacity=800,
                           common_queue_min=400):
    """Loads a single batch of data.

    Args:
      dataset: The dataset to load.
      batch_size: The number of images in the batch.
      height: The size of each image after preprocessing.
      width: The size of each image after preprocessing.
      common_queue_capacity, common_queue_min: decide the shuffle degree

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
    image_color = tf.image.adjust_contrast(image_color, 10)
    image_depth = inception_preprocessing.preprocess_image(
        image_depth, height, width, is_training=False)
    image_depth = tf.image.adjust_contrast(image_depth, 10)

    # Batch it up.
    images_color, images_depth, labels = tf.train.batch(
        [image_color, image_depth, label],
        batch_size=batch_size,
        num_threads=1,
        capacity=2*batch_size)

    return images_color, images_depth, labels
