"""
Test convolutional auto-encoders on fingerspelling5 dataset

Use grayscale images (one channel) and resize to 83x83
These functions are also used for the pre-training stage of the fusion part
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from images import TrainCAE, EvaluateCAE
from images import CAE_6layers
from images import TrainClassifyImagesCAE, EvaluateClassifyImagesCAE
from images import VisualizeImages


tfrecord_dirs = {
    'color': '../dataset/fingerspelling5/tfrecords/color_separated',
    'depth': '../dataset/fingerspelling5/tfrecords/depth_separated'
}

log_dirs_CAE = {
    'color': 'test/log/CAE/CAE_color',
    'depth': 'test/log/CAE/CAE_depth'
}

log_dirs_eva = {
    'color': 'test/log/CAE/CAE_eva_color',
    'depth': 'test/log/CAE/CAE_eva_depth'
}

log_dirs_classify = {
    'color': 'test/log/CAE/classify_color',
    'depth': 'test/log/CAE/classify_depth'
}

visualize_dirs = {
    'color': 'test/log/CAE/visualize_color',
    'depth': 'test/log/CAE/visualize_depth'
}


def train_CAE(modality='color'):
    TrainCAE(CAE_6layers, image_size=83, channels=1).train(
        tfrecord_dirs[modality], None, log_dirs_CAE[modality],
        dropout_position='input', number_of_epochs=1, save_model_steps=500)


def evaluate_CAE(split_name='validation', modality='color'):
    EvaluateCAE(CAE_6layers, image_size=83, channels=1).evaluate(
        tfrecord_dirs[modality], log_dirs_CAE[modality],
        log_dirs_eva[modality], split_name=split_name,
        do_dropout=True, batch_size=50)


def train_classify_CAE(modality='color'):
    TrainClassifyImagesCAE(CAE_6layers, image_size=83, channels=1).train(
        tfrecord_dirs[modality], log_dirs_CAE[modality],
        log_dirs_classify[modality], number_of_epochs=1, save_model_steps=500)


def evaluate_classify_CAE(split_name='validation', modality='color'):
    EvaluateClassifyImagesCAE(
            CAE_6layers, image_size=83, channels=1).evaluate(
        tfrecord_dirs[modality], log_dirs_classify[modality],
        None, split_name=split_name, batch_size=None)


def visualize_CAE(split_name='validation', modality='color'):
    VisualizeImages(CAE_6layers, image_size=83, channels=1).visualize(
        tfrecord_dirs[modality], log_dirs_CAE[modality],
        visualize_dirs[modality], batch_size=5000,
        split_name=split_name, batch_stat=True)
