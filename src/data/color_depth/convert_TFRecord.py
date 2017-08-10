"""Convert some image dataset to TFrecords containing color and depth images.

In the generated TFReocrds each read sample contains a color image, its
corresponding depth map and the label of this image.

This script was historically used for generating TFRecords of the Creative
Senz3d and ASL Finger Spelling dataset. We have one directory for
each subject and in these directories we have one directory for each class.
Finally all of these subject directories are contained in either the
`train` or `validation` directory of the main directory.
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


def to_tfexample(color_data, depth_data,
                 color_format, depth_format, class_id):
    return tf.train.Example(features=tf.train.Features(feature={
        'image/color/encoded': dataset_utils.bytes_feature(color_data),
        'image/color/format': dataset_utils.bytes_feature(color_format),
        'image/depth/encoded': dataset_utils.bytes_feature(depth_data),
        'image/depth/format': dataset_utils.bytes_feature(depth_format),
        'image/class/label': dataset_utils.int64_feature(class_id),
    }))


def get_fpairs_and_classes(dataset_dir, subjects=True):
    """Returns a list of filename pairs and inferred class names.

    Args:
        dataset_dir: If `subjects` is `False`, this directory contains
            a set of subdirectoires representing class names and each
            subdirectory containts thus the corresponding images
            (JPG or PNG).
            If `subjects` is `True`, the directories for classes are
            first contained in some arbitrary directories (e.g. one
            directory for each subject in the dataset).
        subjects: This argument determines the structure of `dataset_dir`

    Returns:
        A list of image file path pairs and the list of class names
    """
    directories = []
    class_names = set()
    photo_paths = []

    for filename in os.listdir(dataset_dir):
        path = os.path.join(dataset_dir, filename)
        if os.path.isdir(path):
            if subjects:
                photos, clss = get_fpairs_and_classes(path, subjects=False)
                photo_paths.extend(photos)
                class_names.update(clss)
            else:
                directories.append(path)
                class_names.add(filename)

    if not subjects:
        for directory in directories:
            color_filenames = set()
            depth_filenames = set()
            for filename in os.listdir(directory):
                _, extension = os.path.splitext(filename)
                if extension in ['.jpg', '.png', '.jpeg']:
                    if 'color' in filename:
                        depth_name = filename.replace('color', 'depth')
                        if depth_name in depth_filenames:
                            depth_filenames.remove(depth_name)
                            color_path = os.path.join(directory, filename)
                            depth_path = os.path.join(directory, depth_name)
                            photo_paths.append([color_path, depth_path])
                        else:
                            color_filenames.add(filename)
                    elif 'depth' in filename:
                        color_name = filename.replace('depth', 'color')
                        if color_name in color_filenames:
                            color_filenames.remove(color_name)
                            color_path = os.path.join(directory, color_name)
                            depth_path = os.path.join(directory, filename)
                            photo_paths.append([color_path, depth_path])
                        else:
                            depth_filenames.add(filename)

    return photo_paths, class_names


def train_validate_filename_pairs_classes(dataset_dir, subjects=True):
    """Returns lists of filename pairs for test/validation set and
    inferred class names.

    Args:
        dataset_dir: A directory contating two subdirectories 'train'
            and 'validation', for the structure of these subdirectories
            please see the function `get_filenames_and_classes`.
        subjects: This argument determines the structure of `dataset_dir`.

    Returns:
        Lists of image file path pairs for training and test directory
        and the list of class names.
    """
    train_dir = os.path.join(dataset_dir, 'train')
    validation_dir = os.path.join(dataset_dir, 'validation')
    train_files, train_clss = get_fpairs_and_classes(
        train_dir, subjects=subjects)
    validation_files, validation_clss = get_fpairs_and_classes(
        validation_dir, subjects=subjects)
    if train_clss != validation_clss:
        print('Warning: different class names for training and validation')
    return (train_files, validation_files,
            sorted(list(train_clss | validation_clss)))


def get_tfrecord_filename(split_name, tfrecord_dir, shard_id, num_shards):
    output_filename = 'color_depth_%s_%d-of-%d.tfrecord' % (
        split_name, shard_id, num_shards)
    return os.path.join(tfrecord_dir, output_filename)


def convert_dataset(split_name, filename_pairs, class_names_to_ids,
                    tfrecord_dir, num_shards=5):
    """Converts the given filenamei pairs to a TFRecord dataset.

    Args:
        split_name: The name of the dataset, either 'train' or 'validation'.
        filename_pairs: A list of path pairs to png or jpg images.
        class_names_to_ids: A dictionary from class names (strings) to ids
            (integers).
        tfrecord_dir: The directory where the converted datasets are stored.
        num_shards: The number of shards per dataset split
    """
    assert split_name in ['train', 'validation']

    num_per_shard = int(math.ceil(len(filename_pairs)/float(num_shards)))

    with tf.Graph().as_default():

        for shard_id in range(num_shards):
            output_filename = get_tfrecord_filename(
                split_name, tfrecord_dir, shard_id, num_shards)

            with tf.python_io.TFRecordWriter(output_filename)\
                    as tfrecord_writer:
                start_ndx = shard_id * num_per_shard
                end_ndx = min((shard_id+1)*num_per_shard, len(filename_pairs))
                for i in range(start_ndx, end_ndx):
                    sys.stdout.write(
                        '\r>> Converting image %d/%d shard %s %d' % (
                            i+1, len(filename_pairs), split_name, shard_id))
                    sys.stdout.flush()

                    # Read the filename:
                    color_data = tf.gfile.FastGFile(
                        filename_pairs[i][0], 'r').read()
                    depth_data = tf.gfile.FastGFile(
                        filename_pairs[i][1], 'r').read()

                    class_name = os.path.basename(
                        os.path.dirname(filename_pairs[i][0]))
                    class_id = class_names_to_ids[class_name]

                    _, color_format = os.path.splitext(filename_pairs[i][0])
                    _, depth_format = os.path.splitext(filename_pairs[i][1])

                    example = to_tfexample(
                        color_data, depth_data, color_format,
                        depth_format, class_id)
                    tfrecord_writer.write(example.SerializeToString())

    sys.stdout.write('\n')
    sys.stdout.flush()


def convert_color_depth(dataset_dir,
                        tfrecord_dir,
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
    training_filename_pairs, validation_filename_pairs, class_names = \
        train_validate_filename_pairs_classes(dataset_dir, subjects=subjects)
    class_names_to_ids = dict(zip(class_names, range(len(class_names))))

    assert sep in ['user', 'mixed', 'class']

    if sep == 'user':
        random.shuffle(training_filename_pairs)
        random.shuffle(validation_filename_pairs)

    elif sep == 'mixed':
        num_train_ex = len(training_filename_pairs)
        all_pairs = training_filename_pairs + validation_filename_pairs
        random.shuffle(all_pairs)
        training_filename_pairs = all_pairs[:num_train_ex]
        validation_filename_pairs = all_pairs[num_train_ex:]

    elif sep == 'class':
        all_pairs = training_filename_pairs + validation_filename_pairs
        random.shuffle(all_pairs)
        training_filename_pairs = []
        validation_filename_pairs = []
        for fpair in all_pairs:
            cls = os.path.basename(os.path.dirname(fpair[0]))
            if cls in class_names[:-num_val_clss]:
                training_filename_pairs.append(fpair)
            else:
                assert cls in class_names[-num_val_clss:]
                validation_filename_pairs.append(fpair)

    # convert datasets
    convert_dataset('train', training_filename_pairs,
                    class_names_to_ids, tfrecord_dir, num_shards=num_shards)
    convert_dataset('validation', validation_filename_pairs,
                    class_names_to_ids, tfrecord_dir, num_shards=num_shards)

    # write the label file
    labels_to_class_names = dict(zip(range(len(class_names)), class_names))
    dataset_utils.write_label_file(labels_to_class_names, tfrecord_dir)

    print('\nFinished converting dataset!')
