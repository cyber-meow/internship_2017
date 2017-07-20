from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import random

import numpy as np
import scipy.signal

import tensorflow as tf
from data import dataset_utils


def parse_mfcc(file_path, feature_len=26):
    """Parse a htk mfcc ascii file (output by HList) to a numpy array

    Args:
      file_path: Where to find the file
      feature_len: The feature length for each time frame

    Returns:
      a numpy array of size (num_frames, feature_len)
    """
    with open(file_path, 'r') as f:
        f.readline()
        content = f.read().split(':')[1:]
    mfccs = np.empty((feature_len, len(content)))

    for i, frame in enumerate(content):
        mfccs[:, i] = [float(x) for x in frame.split()[:feature_len]]
    return mfccs


def to_tfexample(mfcc_data, class_id):
    return tf.train.Example(features=tf.train.Features(feature={
        'audio/mfcc': dataset_utils.float_feature(mfcc_data),
        'audio/label': dataset_utils.int64_feature(class_id)
    }))


def get_tfrecord_filename(split_name, tfrecord_dir, shard_id, num_shards):
    output_filename = 'mfcc_%s_%d-of-%d.tfrecord' % (
        split_name, shard_id, num_shards)
    return os.path.join(tfrecord_dir, output_filename)


def convert_dataset(split_name,
                    file_paths,
                    class_names_to_ids,
                    tfrecord_dir,
                    num_shards=5,
                    feature_len=26,
                    num_frames=24):

    assert split_name in ['train', 'validation']

    num_per_shard = int(math.ceil(len(file_paths)/float(num_shards)))

    with tf.Graph().as_default():

        for shard_id in range(num_shards):
            output_filename = get_tfrecord_filename(
                split_name, tfrecord_dir, shard_id, num_shards)

            with tf.python_io.TFRecordWriter(output_filename)\
                    as tfrecord_writer:
                start_ndx = shard_id * num_per_shard
                end_ndx = min((shard_id+1)*num_per_shard, len(file_paths))
                for i in range(start_ndx, end_ndx):
                    sys.stdout.write(
                        '\r>> Converting file %d/%d shard %s %d' % (
                            i+1, len(file_paths), split_name, shard_id))
                    sys.stdout.flush()

                    mfccs = parse_mfcc(file_paths[i], feature_len=feature_len)
                    mfcc_data = list(scipy.signal.resample(
                        mfccs, num_frames, axis=1).reshape(-1))

                    class_name = os.path.basename(file_paths[i])[0]
                    class_id = class_names_to_ids[class_name]

                    example = to_tfexample(mfcc_data, class_id)
                    tfrecord_writer.write(example.SerializeToString())

    sys.stdout.write('\n')
    sys.stdout.flush()


def convert_mfcc(dataset_dir,
                 tfrecord_dir,
                 sep='user',
                 num_shards=5,
                 num_val_samples=None,
                 num_frames=24):

    if not tf.gfile.Exists(tfrecord_dir):
        tf.gfile.MakeDirs(tfrecord_dir)

    train_dir = os.path.join(dataset_dir, 'train')
    training_filenames = [os.path.join(train_dir, filename)
                          for filename in os.listdir(train_dir)]

    validation_dir = os.path.join(dataset_dir, 'validation')
    validation_filenames = [os.path.join(validation_dir, filename)
                            for filename in os.listdir(validation_dir)]
    alphabets = [chr(i) for i in range(ord('A'), ord('Z')+1)]

    class_names_to_ids = dict(zip(alphabets, range(26)))

    assert sep in ['user', 'mixed']

    if sep == 'user':
        random.shuffle(training_filenames)
        random.shuffle(validation_filenames)

    elif sep == 'mixed':
        if num_val_samples is None:
            num_val_samples = len(validation_filenames)
        all_filenames = training_filenames + validation_filenames
        random.shuffle(all_filenames)
        training_filenames = all_filenames[:-num_val_samples]
        validation_filenames = all_filenames[-num_val_samples:]

    convert_dataset('train', training_filenames,
                    class_names_to_ids, tfrecord_dir,
                    num_shards=num_shards, num_frames=num_frames)
    convert_dataset('validation', validation_filenames,
                    class_names_to_ids, tfrecord_dir,
                    num_shards=num_shards, num_frames=num_frames)

    labels_to_class_names = dict(zip(range(26), alphabets))
    dataset_utils.write_label_file(labels_to_class_names, tfrecord_dir)

    print('\nFinished converting dataset!')
