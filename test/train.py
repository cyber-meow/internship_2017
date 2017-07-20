from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from routines.train import train
from classify import TrainClassifyCNN, CNN_9layers

from CAE import TrainCAE, TrainClassifyCAE
from CAE import CAE_6layers, CAE_12layers

from fusion import TrainFusionAE, fusion_AE_6layers, TrainEmbedding
from fusion import TrainClassifyCommonRepr, TrainClassifyFusion

from audio import CNN_mfcc, TrainClassifyAudio

"""
General arguments included in **kwargs:
  renorm, test_use_batch, use_batch_norm, trainable_scopes
  batch_size, number_of_steps, number_of_epochs,
  lr_decary_steps, lr_decay_rate, initial_learning_rate

Arguments for single modality inputs:
  image_size, channels

Arguments for color and depth inputs:
  image_size, color_channels, depth_channels
"""


# dropout_keep_prob
def train_raw_classify_test(tfrecord_dir=None, log_dir=None, **kwargs):
    if tfrecord_dir is None:
        tfrecord_dir = '../dataset/senz3d_dataset/tfrecords/color_separated'
    if log_dir is None:
        log_dir = 'test/log/raw_classify'
    train(TrainClassifyCNN, None, tfrecord_dir, None, log_dir, 50, **kwargs)


# dropout_keep_prob
def train_CNN9_classify_test(tfrecord_dir=None, log_dir=None, **kwargs):
    if tfrecord_dir is None:
        tfrecord_dir = '../dataset/senz3d_dataset/tfrecords/color_separated'
    if log_dir is None:
        log_dir = 'test/log/CNN9'
    train(TrainClassifyCNN, CNN_9layers,
          tfrecord_dir, None, log_dir, 50, **kwargs)


# dropout_position, dropout_keep_prob
def train_CAE6_test(tfrecord_dir=None, log_dir=None, **kwargs):
    if tfrecord_dir is None:
        tfrecord_dir = '../dataset/senz3d_dataset/tfrecords/color_separated'
    if log_dir is None:
        log_dir = 'test/log/CAE6'
    train(TrainCAE, CAE_6layers, tfrecord_dir, None, log_dir, 50, **kwargs)


# endpoint, dropout_keep_prob, do_avg
def train_CAE6_classify_test(
        tfrecord_dir=None, checkpoint_dirs=None, log_dir=None, **kwargs):
    if tfrecord_dir is None:
        tfrecord_dir = '../dataset/senz3d_dataset/tfrecords/color_separated'
    if checkpoint_dirs is None:
        checkpoint_dirs = ['test/log/CAE6']
    if log_dir is None:
        log_dir = 'test/log/CAE6_classify'
    train(TrainClassifyCAE, CAE_6layers, tfrecord_dir,
          checkpoint_dirs, log_dir, 50, **kwargs)


# dropout_position, threshold, color_keep_prob
def train_fusionAE6_test(
        tfrecord_dir=None, checkpoint_dirs=None, log_dir=None,
        number_of_steps=50, **kwargs):
    if tfrecord_dir is None:
        tfrecord_dir = \
            '../dataset/senz3d_dataset/tfrecords/color_depth_separated'
    if checkpoint_dirs is None:
        checkpoint_dirs = ['test/log/CAE6', 'test/log/CAE6_depth']
    if log_dir is None:
        log_dir = 'test/log/fusionAE6'
    train(TrainFusionAE, fusion_AE_6layers, tfrecord_dir,
          checkpoint_dirs, log_dir, number_of_steps, **kwargs)


# endpoint, dropout_keep_prob
def train_fusionAE6_classify_test(
        tfrecord_dir=None, checkpoint_dirs=None, log_dir=None,
        number_of_steps=50, **kwargs):
    if tfrecord_dir is None:
        tfrecord_dir = \
            '../dataset/senz3d_dataset/tfrecords/color_depth_separated'
    if checkpoint_dirs is None:
        checkpoint_dirs = ['test/log/fusionAE6']
    if log_dir is None:
        log_dir = 'test/log/fusionAE6_classify'
    train(TrainClassifyFusion, fusion_AE_6layers, tfrecord_dir,
          checkpoint_dirs, log_dir, number_of_steps,
          endpoint='Middle', trainable_scopes=['Logits'], **kwargs)


# modality, dropout_keep_prob
def train_fusionAE6_single_classify_test(
        tfrecord_dir=None, checkpoint_dirs=None, log_dir=None,
        number_of_steps=50, **kwargs):
    if tfrecord_dir is None:
        tfrecord_dir = \
            '../dataset/senz3d_dataset/tfrecords/color_separated'
    if checkpoint_dirs is None:
        checkpoint_dirs = ['test/log/fusionAE6']
    if log_dir is None:
        log_dir = 'test/log/fusionAE6_color_classify'
    train(TrainClassifyCommonRepr, fusion_AE_6layers, tfrecord_dir,
          checkpoint_dirs, log_dir, number_of_steps, **kwargs)


# dropout_position, dropout_keep_prob
def train_CAE12_test(tfrecord_dir=None, log_dir=None, **kwargs):
    if tfrecord_dir is None:
        tfrecord_dir = '../dataset/fingerspelling5/tfrecords/color_separated'
    if log_dir is None:
        log_dir = 'test/log/CAE12'
    train(TrainCAE, CAE_12layers, tfrecord_dir, None, log_dir, 50,
          image_size=83, channels=1, **kwargs)


# endpoint, dropout_keep_prob, do_avg
def train_CAE12_classify_test(
        tfrecord_dir=None, checkpoint_dirs=None, log_dir=None, **kwargs):
    if tfrecord_dir is None:
        tfrecord_dir = '../dataset/fingerspelling5/tfrecords/color_separated'
    if checkpoint_dirs is None:
        checkpoint_dirs = ['test/log/CAE12']
    if log_dir is None:
        log_dir = 'test/log/CAE12_classify'
    train(TrainClassifyCAE, CAE_12layers, tfrecord_dir,
          checkpoint_dirs, log_dir, 50,
          image_size=83, channels=1, do_avg=True, **kwargs)


# feature_length
def train_CAE6_embedding_test(
        tfrecord_dir=None, checkpoint_dirs=None, log_dir=None, **kwargs):
    if tfrecord_dir is None:
        tfrecord_dir = \
            '../dataset/fingerspelling5/tfrecords/color_depth_separated'
    if checkpoint_dirs is None:
        checkpoint_dirs = ['test/log/CAE6', 'test/log/CAE6_depth']
    if log_dir is None:
        log_dir = 'test/log/embedding_CAE6'
    train(TrainEmbedding, CAE_6layers, tfrecord_dir,
          checkpoint_dirs, log_dir, 50, **kwargs)


# dropout_keep_prob
def train_classify_audio_test(tfrecord_dir=None, log_dir=None, **kwargs):
    if tfrecord_dir is None:
        tfrecord_dir = '../dataset/avletters/tfrecords/mfcc_sep'
    if log_dir is None:
        log_dir = 'test/log/classif_audio'
    train(TrainClassifyAudio, CNN_mfcc,
          tfrecord_dir, None, log_dir, 50, **kwargs)
