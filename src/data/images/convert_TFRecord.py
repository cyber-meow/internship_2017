"""Convert some image dataset to TFRecords.

This script was historically used for generating TFRecords of the Creative
Senz3d and ASL Finger Spelling dataset. We have one directory for
each subject and in these directories we have one directory for each class.
Finally all of these subject directories are contained in either the
`train` or `validation` directory of the main directory.

Put keywords=['depth'] for generating TFRecords of depth images and
keywords=['color'] to generate TFRecords of color images.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import random

import tensorflow as tf

from data import dataset_utils


class ImageReader(object):
    """"Used by `convert_dataset` for reading some image information."""

    def __init__(self):
        self._decode_data = tf.placeholder(dtype=tf.string)
        self._decode = tf.image.decode_image(self._decode_data, channels=3)

    def read_image_dims(self, sess, image_data):
        image = self.decode(sess, image_data)
        return image.shape[0], image.shape[1]

    def decode(self, sess, image_data):
        image = sess.run(self._decode,
                         feed_dict={self._decode_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def get_filenames_and_classes(dataset_dir, keywords=None, subjects=True):
    """Returns a list of filenames and inferred class names.

    Args:
        dataset_dir: If `subjects` is `False`, this directory contains
            a set of subdirectoires representing class names and each
            subdirectory containts thus the corresponding images
            (JPG or PNG).
            If `subjects` is `True`, the directories for classes are
            first contained in some arbitrary directories (e.g. one
            directory for each subject in the dataset).
        keywords: filenames must containg all of these keywords.
        subjects: This argument determines the structure of `dataset_dir`.

    Returns:
        A list of image file paths and the list of class names
    """
    directories = []
    class_names = set()
    photo_filenames = []

    for filename in os.listdir(dataset_dir):
        path = os.path.join(dataset_dir, filename)
        if os.path.isdir(path):
            if subjects:
                photos, clss = get_filenames_and_classes(
                    path, keywords, subjects=False)
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
                if extension in ['.jpg', '.png', '.jpeg']:
                    if keywords is None:
                        photo_filenames.append(path)
                    else:
                        to_add = True
                        for keyword in keywords:
                            if keyword not in filename:
                                to_add = False
                                break
                        if to_add:
                            photo_filenames.append(path)

    return photo_filenames, sorted(list(class_names))


def train_validate_filenames_classes(dataset_dir, keywords, subjects=True):
    """Returns lists of filenames for test/validation set and
    inferred class names.

    Args:
        dataset_dir: A directory contating two subdirectories 'train'
            and 'validation', for the structure of these subdirectories
            please see the function `get_filenames_and_classes`.
        keywords: filenames must containg all of these keywords.
        subjects: This argument determines the structure of `dataset_dir`.

    Returns:
        Lists of image file paths for training and test directory and
        the list of class names.
    """
    train_dir = os.path.join(dataset_dir, 'train')
    validation_dir = os.path.join(dataset_dir, 'validation')
    train_files, train_clss = get_filenames_and_classes(
        train_dir, keywords, subjects=subjects)
    validation_files, validation_clss = get_filenames_and_classes(
        validation_dir, keywords, subjects=subjects)
    assert train_clss == validation_clss
    return train_files, validation_files, train_clss


def get_tfrecord_filename(split_name, tfrecord_dir, shard_id, num_shards):
    output_filename = 'data_%s_%d-of-%d.tfrecord' % (
        split_name, shard_id, num_shards)
    return os.path.join(tfrecord_dir, output_filename)


def convert_dataset(split_name, filenames, class_names_to_ids,
                    tfrecord_dir, num_shards=5):
    """Converts the given filenames to a TFRecord dataset.

    Args:
        split_name: The name of the dataset, either 'train' or 'validation'.
        filenames: A list of paths to png or jpg images.
        class_names_to_ids: A dictionary from class names (strings) to ids
            (integers).
        tfrecord_dir: The directory where the converted datasets are stored.
        num_shards: The number of shards per dataset split
    """
    assert split_name in ['train', 'validation']

    num_per_shard = int(math.ceil(len(filenames) / float(num_shards)))

    with tf.Graph().as_default():
        image_reader = ImageReader()

        with tf.Session() as sess:

            for shard_id in range(num_shards):
                output_filename = get_tfrecord_filename(
                    split_name, tfrecord_dir, shard_id, num_shards)

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


def convert_images(dataset_dir,
                   tfrecord_dir,
                   keywords=None,
                   subjects=True,
                   sep='user',
                   num_val_clss=2,
                   num_shards=5):
    """Runs the conversion operation.

    Args:
        dataset_dir: Where the data (i.e. images) is stored.
        tfrecord_dir: Where to store the generated data (i.e. TFRecords).
        keywords: Filenames must contain these keywords.
        subjects: Determine directory structure, please refer to
            `get_filenames_and_classes`.
        sep: The way to separate train and validation data.
            'user'- uses the given separation (`train` and `validation`
                directories).
            'mixed'- put all the data samples toghether and
                conducts a random split.
            'class'- puts different classes in training and validation set
                (used for some particular purpose).
        num_val_clss: Used only when sep=='class', the number of classes
            in validation set.
        num_shards: The number of shards per dataset split.
    """
    if not tf.gfile.Exists(tfrecord_dir):
        tf.gfile.MakeDirs(tfrecord_dir)

    # get filenames and classnames
    training_filenames, validation_filenames, class_names = \
        train_validate_filenames_classes(
            dataset_dir, keywords, subjects=subjects)
    class_names_to_ids = dict(zip(class_names, range(len(class_names))))

    assert sep in ['user', 'mixed', 'class']

    if sep == 'user':
        random.shuffle(training_filenames)
        random.shuffle(validation_filenames)

    elif sep == 'mixed':
        num_train_ex = len(training_filenames)
        all_filenames = training_filenames + validation_filenames
        random.shuffle(all_filenames)
        training_filenames = all_filenames[:num_train_ex]
        validation_filenames = all_filenames[num_train_ex:]

    elif sep == 'class':
        all_filenames = training_filenames + validation_filenames
        random.shuffle(all_filenames)
        training_filenames = []
        validation_filenames = []
        for filename in all_filenames:
            cls = os.path.basename(os.path.dirname(filename))
            if cls in class_names[:-num_val_clss]:
                training_filenames.append(filename)
            else:
                assert cls in class_names[-num_val_clss:]
                validation_filenames.append(filename)

    # convert datasets
    convert_dataset('train', training_filenames,
                    class_names_to_ids, tfrecord_dir, num_shards=num_shards)
    convert_dataset('validation', validation_filenames,
                    class_names_to_ids, tfrecord_dir, num_shards=num_shards)

    # write the label file
    labels_to_class_names = dict(zip(range(len(class_names)), class_names))
    dataset_utils.write_label_file(labels_to_class_names, tfrecord_dir)

    print('\nFinished converting dataset!')
