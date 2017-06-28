"""Convert some given dataset to TFrecords"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math

import tensorflow as tf

from datasets import dataset_utils


# The number of shards per dataset split
_NUM_SHARDS = 5


class ImageReader(object):

    def __init__(self):
        self._decode_data = tf.placeholder(dtype=tf.string)
        self._decode = tf.image.decode_png(self._decode_data, channels=3)

    def read_image_dims(self, sess, image_data):
        image = self.decode(sess, image_data)
        return image.shape[0], image.shape[1]

    def decode(self, sess, image_data):
        image = sess.run(self._decode,
                         feed_dict={self._decode_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def filenames_and_classes(dataset_dir, subjects=True):
    """Returns a list of filenames and inferred class names.

    Args:
      dataset_dir: If subjects is False, this directory contains a set
        of subdirectoires representing class names and each subdirectory
        containts thus the correspondant images (JPG or PNG).
        If subjects is True, the directories for classes are first
        contained in some arbitrary directories (e.g. one directory for
        each subject in the dataset)
      subjects: This argument determines the structure of `dataset_dir`

    Returns:
      A list of image file paths, relative to `dataset_dir` and the
      list of class names
    """
    directories = []
    class_names = set()
    photo_filenames = []

    for filename in os.listdir(dataset_dir):
        path = os.path.join(dataset_dir, filename)
        if os.path.isdir(path):
            if subjects:
                photos, clss = filenames_and_classes(path, subjects=False)
                photo_filenames.extend(photos)
                class_names.update(clss)
            else:
                directories.append(path)
                class_names.add(filename)

    if not subjects:
        for directory in directories:
            for filename in os.listdir(directory):
                path = os.path.join(directory, filename)
                name, extension = os.path.splitext(path)
                if extension in ['.jpg', '.png']:
                    photo_filenames.append(path)

    return photo_filenames, sorted(list(class_names))


def get_filenames_and_classes(dataset_dir, subjects=True):
    """Returns lists of filenames for test/validation set and
    inferred class names.

    Args:
      dataset_dir: A directory contating two subdirectories 'train'
        and 'validation', for the structure of these subdirectories
        please see the function filenames_and_classes
      subjects: This argument determines the structure of `dataset_dir`

    returns
      A list of image file paths, relative to `dataset_dir` and the
      list of class names
    """
    train_dir = os.path.join(dataset_dir, 'train')
    validation_dir = os.path.join(dataset_dir, 'validation')
    train_files, train_clss = filenames_and_classes(
        train_dir, subjects=subjects)
    validation_files, validation_clss = filenames_and_classes(
        validation_dir, subjects=subjects)
    assert train_clss == validation_clss
    return train_files, validation_files, train_clss


def _get_dataset_filename(dataset_dir, split_name, shard_id):
    output_filename = 'data_%s_%d-of-%d.tfrecord' % (
        split_name, shard_id, _NUM_SHARDS)
    return os.path.join(dataset_dir, output_filename)


def convert_dataset(split_name, filenames, class_names_to_ids, dataset_dir):
    """Converts the given filenames to a TFRecord dataset.

    Args:
      split_name: The name of the dataset, either 'train' or 'validation'.
      filenames: A list of absolute paths to png or jpg images.
      class_names_to_ids: A dictionary from class names (strings) to ids
        (integers).
      dataset_dir: The directory where the converted datasets are stored.
    """
    assert split_name in ['train', 'validation']

    num_per_shard = int(math.ceil(len(filenames) / float(_NUM_SHARDS)))

    with tf.Graph().as_default():
        image_reader = ImageReader()

        with tf.Session('') as sess:

            for shard_id in range(_NUM_SHARDS):
                output_filename = _get_dataset_filename(
                    dataset_dir, split_name, shard_id)

                with tf.python_io.TFRecordWriter(output_filename)\
                        as tfrecord_writer:
                    start_ndx = shard_id * num_per_shard
                    end_ndx = min((shard_id+1) * num_per_shard, len(filenames))
                    for i in range(start_ndx, end_ndx):
                        sys.stdout.write(
                            '\r>> Converting image %d/%d shard %s %d' % (
                                i+1, len(filenames), split_name, shard_id))
                        sys.stdout.flush()

                        # Read the filename:
                        image_data = tf.gfile.FastGFile(
                            filenames[i], 'r').read()
                        height, width = image_reader.read_image_dims(
                            sess, image_data)

                        class_name = os.path.basename(
                            os.path.dirname(filenames[i]))
                        class_id = class_names_to_ids[class_name]
                        _, ext = os.path.splitext(filenames[i])

                        example = dataset_utils.image_to_tfexample(
                            image_data, ext, height, width, class_id)
                        tfrecord_writer.write(example.SerializeToString())

    sys.stdout.write('\n')
    sys.stdout.flush()


def run(dataset_dir, tfrecord_dir, subjects=True):
    """Runs the conversion operation.

    Args:
        dataset_dir: where the data (i.e. images) is stored
        tfrecord_dir: where to store the generated data (i.e. TFRecords)
    """
    if not tf.gfile.Exists(tfrecord_dir):
        tf.gfile.MakeDirs(tfrecord_dir)

    # get filenames and classnames
    training_filenames, validation_filenames, class_names = \
        get_filenames_and_classes(dataset_dir, subjects=subjects)
    class_names_to_ids = dict(zip(class_names, range(len(class_names))))

    # convert datasets
    convert_dataset('train', training_filenames,
                    class_names_to_ids, tfrecord_dir)
    convert_dataset('validation', validation_filenames,
                    class_names_to_ids, tfrecord_dir)

    # write the label file
    labels_to_class_names = dict(zip(range(len(class_names)), class_names))
    dataset_utils.write_label_file(labels_to_class_names, tfrecord_dir)

    print('\nFinished converting dataset!')
