from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from routines.evaluate import evaluate
from CAE import EvaluateCAE, EvaluateClassifyCAE, CAE_6layers
from fusion import EvaluateFusionAE, EvaluateFusionAESingle, fusion_AE_6layers


"""
General arguments included in **kwargs:
  number_of_epochs, use_batch_norm, trainable_scopes,
  split_name, batch_size, number_of_steps

Arguments for single modality inputs:
  image_size, channels

Arguments for color and depth inputs:
  image_size, color_channels, depth_channels
"""


# endpoint
def evaluate_CAE6_classify_test(
        tfrecord_dir=None, checkpoint_dirs=None, log_dir=None, **kwargs):
    if tfrecord_dir is None:
        tfrecord_dir = '../dataset/senz3d_dataset/tfrecords/color_separated'
    if checkpoint_dirs is None:
        checkpoint_dirs = ['test/log/CAE6_classify']
    if log_dir is None:
        log_dir = 'test/log/eva/CAE6_classify'
    evaluate(EvaluateClassifyCAE, CAE_6layers, tfrecord_dir,
             checkpoint_dirs, log_dir, **kwargs)


# dropout_input, dropout_keep_prob
def evaluate_CAE6_test(
        tfrecord_dir=None, checkpoint_dirs=None, log_dir=None, **kwargs):
    if tfrecord_dir is None:
        tfrecord_dir = '../dataset/senz3d_dataset/tfrecords/color_separated'
    if checkpoint_dirs is None:
        checkpoint_dirs = ['test/log/CAE6']
    if log_dir is None:
        log_dir = 'test/log/eva/CAE6'
    evaluate(EvaluateCAE, CAE_6layers, tfrecord_dir,
             checkpoint_dirs, log_dir, **kwargs)


# color_keep_prob, depth_keep_prob, dropout_position
def evaluate_fusionAE6_test(
        tfrecord_dir=None, checkpoint_dirs=None, log_dir=None, **kwargs):
    if tfrecord_dir is None:
        tfrecord_dir = \
            '../dataset/senz3d_dataset/tfrecords/color_depth_separated'
    if checkpoint_dirs is None:
        checkpoint_dirs = ['test/log/fusionAE6']
    if log_dir is None:
        log_dir = 'test/log/eva/fusionAE6'
    evaluate(EvaluateFusionAE, fusion_AE_6layers, tfrecord_dir,
             checkpoint_dirs, log_dir, **kwargs)


# modality
def evaluate_fusionAE6_single_test(
        tfrecord_dir=None, checkpoint_dirs=None, log_dir=None, **kwargs):
    if tfrecord_dir is None:
        tfrecord_dir = \
            '../dataset/senz3d_dataset/tfrecords/color_separated'
    if checkpoint_dirs is None:
        checkpoint_dirs = ['test/log/fusionAE6']
    if log_dir is None:
        log_dir = 'test/log/eva/fusionAE6_single'
    evaluate(EvaluateFusionAESingle, fusion_AE_6layers, tfrecord_dir,
             checkpoint_dirs, log_dir, batch_size=24, **kwargs)
