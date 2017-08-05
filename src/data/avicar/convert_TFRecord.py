from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import random

import scipy.io.wavfile
import scipy.signal
import librosa

import tensorflow as tf
from data import dataset_utils


def read_wav(file_path, feature_len=26, num_frames=20):
    """Read a .wav file and compute its mfcc features

    Args:
      file_path: Where to find the file
      feature_len: The feature length for each time frame
      num_frames: The number of time frames in output for mfcc

    Returns:
      Sample rate, wav data and mfcc features of size
        (num_frames, feature_len)
    """
    sr, audio_data = scipy.io.wavfile.read(file_path)
    mfcc = librosa.feature.mfcc(audio_data, sr, n_mfcc=feature_len)
    mfcc_res = scipy.signal.resample(mfcc, num_frames, axis=1)
    return sr, audio_data, mfcc_res


def to_tfexample(raw_data, mfcc_data, class_id):
    return tf.train.Example(features=tf.train.Features(feature={
        'audio/mfcc': dataset_utils.float_feature(mfcc_data),
        'audio/wav/data': dataset_utils.float_feature(raw_data),
        'audio/wav/length': dataset_utils.int64_feature(len(raw_data)),
        'audio/label': dataset_utils.int64_feature(class_id)
    }))


def get_tfrecord_filename(split_name, tfrecord_dir, shard_id, num_shards):
    output_filename = 'avicar_%s_%d-of-%d.tfrecord' % (
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

                    sr, audio_data, mfcc = read_wav(
                        file_paths[i], feature_len=feature_len,
                        num_frames=num_frames)
                    audio_data = list(audio_data)
                    mfcc_data = list(mfcc.reshape(-1))

                    class_name = os.path.basename(file_paths[i])[9]
                    class_id = class_names_to_ids[class_name]

                    example = to_tfexample(audio_data, mfcc_data, class_id)
                    tfrecord_writer.write(example.SerializeToString())

    sys.stdout.write('\n')
    sys.stdout.flush()


def convert_avicar(dataset_dir,
                   tfrecord_dir,
                   num_shards=5,
                   num_val_samples=2000,
                   num_frames=20):

    if not tf.gfile.Exists(tfrecord_dir):
        tf.gfile.MakeDirs(tfrecord_dir)

    filenames = [os.path.join(dataset_dir, filename)
                 for filename in os.listdir(dataset_dir)]
    random.shuffle(filenames)

    alphabets = [chr(i) for i in range(ord('A'), ord('Z')+1)]

    class_names_to_ids = dict(zip(alphabets, range(26)))

    training_filenames = filenames[:-num_val_samples]
    validation_filenames = filenames[-num_val_samples:]

    convert_dataset('train', training_filenames,
                    class_names_to_ids, tfrecord_dir,
                    num_shards=num_shards, num_frames=num_frames)
    convert_dataset('validation', validation_filenames,
                    class_names_to_ids, tfrecord_dir,
                    num_shards=num_shards, num_frames=num_frames)

    labels_to_class_names = dict(zip(range(26), alphabets))
    dataset_utils.write_label_file(labels_to_class_names, tfrecord_dir)

    print('\nFinished converting dataset!')
