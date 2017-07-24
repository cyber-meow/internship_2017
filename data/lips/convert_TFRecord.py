from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import random

import numpy as np
import scipy.io
import scipy.signal
import scipy.ndimage
from sklearn import preprocessing

import tensorflow as tf
from data import dataset_utils


def read_mat(file_path, num_frames=12, laplace=False):
    """Read and preprocess a mat file containing images

    Args:
      file_path: Where to find the file
      num_frames: The number of frames used for the video

    Returns:
      a numpy array of size (num_frames, feature_len)
    """
    video_data = scipy.io.loadmat(file_path)['vid']
    video_data = scipy.signal.resample(video_data, num_frames, axis=1)
    video_data = preprocessing.scale(video_data, axis=0)
    video_data = np.rot90(video_data.reshape((80, 60, num_frames)), k=-1)
    if laplace:
        new_video_data = np.empty((60, 80, num_frames))
        for i in range(num_frames):
            new_video_data[:, :, i] = \
                scipy.ndimage.filters.laplace(video_data[:, :, i])
        video_data = new_video_data
    return video_data


def to_tfexample(mfcc_data, class_id):
    return tf.train.Example(features=tf.train.Features(feature={
        'video/data': dataset_utils.float_feature(mfcc_data),
        'video/label': dataset_utils.int64_feature(class_id)
    }))


def get_tfrecord_filename(split_name, tfrecord_dir, shard_id, num_shards):
    output_filename = 'lips_%s_%d-of-%d.tfrecord' % (
        split_name, shard_id, num_shards)
    return os.path.join(tfrecord_dir, output_filename)


def convert_dataset(split_name,
                    file_paths,
                    class_names_to_ids,
                    tfrecord_dir,
                    num_shards=5,
                    num_frames=12):

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

                    video_data = list(read_mat(
                        file_paths[i], num_frames=num_frames).reshape(-1))

                    class_name = os.path.basename(file_paths[i])[0]
                    class_id = class_names_to_ids[class_name]

                    example = to_tfexample(video_data, class_id)
                    tfrecord_writer.write(example.SerializeToString())

    sys.stdout.write('\n')
    sys.stdout.flush()


def convert_lips(dataset_dir,
                 tfrecord_dir,
                 sep='user',
                 num_shards=5,
                 num_val_samples=None,
                 num_frames=12):

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
