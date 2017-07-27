from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from images import TrainClassifyImagesCNN
from images import CNN_9layers


tfrecord_dir = '../dataset/senz3d_dataset/tfrecords/color_mixed'
log_dir_raw = 'test/log/classify_images/raw'
log_dir_CNN9 = 'test/log/classify_images/raw'


def train_raw(num_steps):
    TrainClassifyImagesCNN(None).train(
        tfrecord_dir, None, log_dir_raw, num_steps)


def train_CNN9(num_steps):
    TrainClassifyImagesCNN(CNN_9layers).train(
        tfrecord_dir, None, log_dir_raw, num_steps)
