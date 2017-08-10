"""Convert TFrecords for the Montalbano gesture dataset.

This is used to deal with the Montalbano gesture dataset.
http://gesture.chalearn.org/2013-multi-modal-challenge/
data-2013-challengedataset.

I read the segmentation annotation data from .mat files and segment
the color and depth videos according to this.
I also resize and resample the videos to have smaller size for latter use.

However, the dataset is too large (~40G) for this script to work.
On one hand it's very slow (it takes several hours) and on the other hand
python will run out of memory. Therefore to do the conversion we may
need to do it in several times separately.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import random

import scipy.io
import imageio

import numpy as np
import scipy.misc
import scipy.signal

import tensorflow as tf
from data import dataset_utils


def process_session(basename, num_frames=40,
                    width=100, height=100, grayscale=True):
    """Preprocess the datafiles of one session.

    This includes the segmentation and resampling of intensity
    and depth videos.

    Args:
        basename: The basename of the session.
        num_frames: The number of times frames contained in the
            output videos.
        width: The width of output videos.
        height: The height of output videos.
        grayscale: To return grayscale video or not.

    Returns:
        A list of triples (color video, depth video, label) representing
        information extrated from the files.
        Each color/depth video is a numpy array of dimension 4
        [Height, Width, Time frames, Channels].
    """
    if not os.path.exists(basename + '_data.mat'):
        return None

    metadata = scipy.io.loadmat(basename + '_data.mat')
    label_data = metadata['Video'][0, 0]['Labels']

    if label_data.shape[1] == 0:
        return None

    color_video = imageio.get_reader(basename + '_color.mp4', 'ffmpeg')
    depth_video = imageio.get_reader(basename + '_depth.mp4', 'ffmpeg')
    segmented_data = []

    for i, seg_info in enumerate(label_data[0]):
        label, start_frame, end_frame = seg_info
        extracted_color_video = extract_video(
            color_video, start_frame[0, 0]-1, end_frame[0, 0]-1,
            width, height, grayscale)
        extracted_depth_video = extract_video(
            depth_video, start_frame[0, 0]-1, end_frame[0, 0]-1,
            width, height, grayscale)
        segmented_data.append([
            scipy.signal.resample(extracted_color_video, num_frames, axis=2),
            scipy.signal.resample(extracted_depth_video, num_frames, axis=2),
            label[0],
        ])
    return segmented_data


def extract_video(video_data, start_frame, end_frame,
                  width=100, height=100, grayscale=True):
    """Extract video data from a given frame range.

    Args:
        video_data: An imageio ffmpeg reader instance that contains
            the video data.
        start_frame: The starting frame of the extracted video.
        end_frame: The final frame of the extracted video.
        width: The width of output video.
        height: The height of output video.
        grayscale: To return grayscale video or not.

    Returns:
        A numpy array of dimension 4 [Height, Width, Time frames, Channels]
    """
    channels = video_data.get_data(0).shape[2]
    extracted_video = np.empty(
        (width, height, end_frame-start_frame+1, channels))
    for i in range(start_frame, end_frame+1):
        extracted_video[:, :, i-start_frame, :] = \
            scipy.misc.imresize(video_data.get_data(i), (100, 100))
    if grayscale and channels == 3:
        extracted_video = (
            0.299 * extracted_video[:, :, :, 0]
            + 0.587 * extracted_video[:, :, :, 1]
            + 0.114 * extracted_video[:, :, :, 2])
    return extracted_video


class_names = [
    'vattene', 'vieniqui', 'perfetto', 'furbo', 'cheduepalle',
    'chevuoi', 'daccordo', 'seipazzo', 'combinato', 'freganiente',
    'ok', 'cosatifarei', 'basta', 'prendere', 'noncenepiu',
    'fame', 'tantotempo', 'buonissimo', 'messidaccordo', 'sonostufo',
]

class_names_to_ids = dict(zip(class_names, range(1, 21)))
labels_to_class_names = dict(zip(range(1, 21), class_names))


def to_tfexample(color_video, depth_video, class_id):
    return tf.train.Example(features=tf.train.Features(feature={
        'video/color/data': dataset_utils.float_feature(list(color_video)),
        'video/color/shape': dataset_utils.float_feature(color_video.shape),
        'video/depth/data': dataset_utils.float_feature(list(depth_video)),
        'video/depth/shape': dataset_utils.float_feature(depth_video.shape),
        'video/label': dataset_utils.int64_feature(class_id),
    }))


def get_tfrecord_filename(split_name, tfrecord_dir, shard_id, num_shards):
    output_filename = 'montalbano_%s_%d-of-%d.tfrecord' % (
        split_name, shard_id, num_shards)
    return os.path.join(tfrecord_dir, output_filename)


def convert_dataset(split_name,
                    data_triples,
                    tfrecord_dir,
                    num_shards=10):

    num_per_shard = int(math.ceil(len(data_triples)/float(num_shards)))

    with tf.Graph().as_default():

        for shard_id in range(num_shards):
            output_filename = get_tfrecord_filename(
                split_name, tfrecord_dir, shard_id, num_shards)

            with tf.python_io.TFRecordWriter(output_filename)\
                    as tfrecord_writer:
                start_ndx = shard_id * num_per_shard
                end_ndx = min((shard_id+1)*num_per_shard, len(data_triples))
                for i in range(start_ndx, end_ndx):
                    sys.stdout.write(
                        '\r>> Converting data instance %d/%d shard %s %d' % (
                            i+1, len(data_triples), split_name, shard_id))
                    sys.stdout.flush()

                    color_video, depth_video, class_name = data_triples[i]
                    class_id = class_names_to_ids[class_name]

                    example = to_tfexample(color_video, depth_video, class_id)
                    tfrecord_writer.write(example.SerializeToString())

    sys.stdout.write('\n')
    sys.stdout.flush()


def convert_montalbano(dataset_dir_train,
                       dataset_dir_validation,
                       tfrecord_dir,
                       num_shards=10,
                       width=100,
                       height=100,
                       num_frames=40, grayscale=True):

    if not tf.gfile.Exists(tfrecord_dir):
        tf.gfile.MakeDirs(tfrecord_dir)

    training_data = []

    for i in range(1, 404):
        basename = os.path.join(
            dataset_dir_train,
            'Sample' + '{:5d}'.format(i).replace(' ', '0'))
        sys.stdout.write('\r >> Processing training session %d/403' % i)
        sys.stdout.flush()
        processed_data = process_session(
            basename, num_frames, width, height, grayscale)
        if processed_data is not None:
            training_data.extend(processed_data)
    sys.stdout.write('\n>> Shuffling training session data\n')
    sys.stdout.flush()
    random.shuffle(training_data)
    convert_dataset('train', training_data, tfrecord_dir, num_shards)

    validation_data = []

    for i in range(410, 711):
        basename = os.path.join(
            dataset_dir_validation,
            'Sample' + '{:5d}'.format(i).replace(' ', '0'))
        sys.stdout.write('\r >> Processing validation session %d/301' % i-410)
        sys.stdout.flush()
        processed_data = process_session(
            basename, num_frames, width, height)
        if processed_data is not None:
            validation_data.extend(processed_data)
    sys.stdout.write('\n>> Shuffling validation_data session data\n')
    sys.stdout.flush()
    random.shuffle(validation_data)
    convert_dataset('validation', validation_data, tfrecord_dir, num_shards)

    dataset_utils.write_label_file(labels_to_class_names, tfrecord_dir)
    print('\nFinished converting dataset!')
