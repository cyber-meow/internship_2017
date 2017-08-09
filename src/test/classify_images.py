"""
Apply basic image classify functions on the senz3d dataset.

Three different classifiers are linear classifiers, hand-coded CNN
and inceptionV4.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from images import TrainClassifyImagesCNN, EvaluateClassifyImagesCNN
from images import TrainClassifyInception, EvaluateClassifyInception
from images import CNN_9layers


tfrecord_dir = '../dataset/senz3d_dataset/tfrecords/color_mixed'
log_dir_raw = 'test/log/classify_images/raw'
log_dir_CNN9 = 'test/log/classify_images/CNN9'

log_dir_inception = 'test/log/classify_images/inception'
inception_checkpoint_dir = '../checkpoints'


def train_raw(num_steps):
    TrainClassifyImagesCNN(None).train(
        tfrecord_dir, None, log_dir_raw, num_steps)


def evaluate_raw(split_name='validation'):
    EvaluateClassifyImagesCNN(None).evaluate(
        tfrecord_dir, log_dir_raw, None,
        split_name=split_name, batch_size=None)


def train_CNN9(num_steps):
    TrainClassifyImagesCNN(CNN_9layers).train(
        tfrecord_dir, None, log_dir_CNN9, num_steps)


def evaluate_CNN9(split_name='validation'):
    EvaluateClassifyImagesCNN(CNN_9layers).evaluate(
        tfrecord_dir, log_dir_CNN9, None,
        split_name=split_name, batch_size=None)


def fine_tune_inception(num_steps):
    TrainClassifyInception().train(
        tfrecord_dir, inception_checkpoint_dir, log_dir_inception, num_steps)


def evaluate_inception(split_name='validation'):
    EvaluateClassifyInception().evaluate(
        tfrecord_dir, log_dir_inception, None,
        split_name=split_name, batch_size=None)
