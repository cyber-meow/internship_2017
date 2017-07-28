from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from multimodal.gesture import fusionAE_6layers, fusion_CNN_10layers
from multimodal.gesture import TrainFusionAE
from multimodal.gesture import EvaluateFusionAE, EvaluateFusionAESingle
from multimodal.gesture import TrainClassifyCommonRepr, TrainClassifyFusion
from multimodal.gesture import EvaluateClassifyCommonRepr
from multimodal.gesture import EvaluateClassifyFusion
from multimodal.gesture import VisualizeColorOrDepth, VisualizeColorAndDepth


class FusionTest(object):

    tfrecord_dir = \
        '../dataset/fingerspelling5/tfrecords/color_depth_separated'
    tfrecord_dir_depth = \
        '../dataset/fingerspelling5/tfrecords/depth_separated'

    CAE_dir_color = 'test/log/CAE/CAE_color'
    CAE_dir_depth = 'test/log/CAE/CAE_depth'

    log_dir_fusion = 'test/log/fusion/fusionAE6'
    log_dir_classify_single = 'test/log/fusion/classify_depth'
    log_dir_classify_both = 'test/log/fusion/classify_both'
    log_dir_classify_CNN = 'test/log/fusion/classify_both_CNN'

    eva_dir_fusion_both = 'test/log/fusion/fusionAE6_eva_20_20'
    eva_dir_fusion_single = 'test/log/fusion/fusionAE6_eva_depth_input'

    visualize_dir_single = 'test/log/fusion/visualize_color_or_depth'
    visualize_dir_both = 'test/log/fusion/visualize_color_and_depth'
    visualize_dir_CNN_single = 'test/log/fusion/visualize_fusion_CNN_or'
    visualize_dir_CNN_both = 'test/log/fusion/visualize_fusion_CNN_and'

    def train_fusionAE(self):
        TrainFusionAE(fusionAE_6layers, image_size=83,
                      color_channels=1, depth_channels=1).train(
            self.tfrecord_dir, [self.CAE_dir_color, self.CAE_dir_depth],
            self.log_dir_fusion, number_of_epochs=1,
            dropout_position='input', save_model_steps=500)

    def evaluate_fusionAE_both(self):
        EvaluateFusionAE(fusionAE_6layers, image_size=83,
                         color_channels=1, depth_channels=1).evaluate(
            self.tfrecord_dir, self.log_dir_fusion,
            self.eva_dir_fusion_both, dropout_position='input',
            color_keep_prob=0.2, depth_keep_prob=0.2,
            batch_size=50, batch_stat=True)

    def evaluate_fusionAE_single(self):
        EvaluateFusionAESingle(
                fusionAE_6layers, image_size=83, channels=1).evaluate(
            self.tfrecord_dir_depth, self.log_dir_fusion,
            self.eva_dir_fusion_single,
            modality='depth', batch_size=50, batch_stat=True)

    def train_classify_fusion_single(self):
        TrainClassifyCommonRepr(
                fusionAE_6layers, image_size=83, channels=1).train(
            self.tfrecord_dir_depth, self.log_dir_fusion,
            self.log_dir_classify_single, number_of_epochs=1,
            save_model_steps=500)

    def evaluate_classify_fusion_single(self, split_name='validation'):
        EvaluateClassifyCommonRepr(
                fusionAE_6layers, image_size=83, channels=1).evaluate(
            self.tfrecord_dir_depth, self.log_dir_classify_single, None,
            split_name=split_name, batch_size=None)

    def train_classify_fusion_both(self):
        TrainClassifyFusion(fusionAE_6layers, image_size=83,
                            color_channels=1, depth_channels=1).train(
            self.tfrecord_dir, self.log_dir_fusion,
            self.log_dir_classify_both, number_of_epochs=1,
            save_model_steps=500, endpoint='Middle')

    def evaluate_classify_fusion_both(self, split_name='validation'):
        EvaluateClassifyFusion(fusionAE_6layers, image_size=83,
                               color_channels=1, depth_channels=1).evaluate(
            self.tfrecord_dir, self.log_dir_classify_both, None,
            split_name=split_name, batch_size=None, endpoint='Middle')

    def train_classify_fusion_CNN(self):
        TrainClassifyFusion(fusion_CNN_10layers, image_size=83,
                            color_channels=1, depth_channels=1).train(
            self.tfrecord_dir, None,
            self.log_dir_classify_CNN, number_of_epochs=1,
            save_model_steps=500, endpoint=None)

    def evaluate_classify_fusion_CNN(self, split_name='validation'):
        EvaluateClassifyFusion(fusion_CNN_10layers, image_size=83,
                               color_channels=1, depth_channels=1).evaluate(
            self.tfrecord_dir, self.log_dir_classify_CNN, None,
            split_name=split_name, batch_size=None)

    def visualize_fusionAE_single(self, split_name='validation'):
        VisualizeColorOrDepth(fusionAE_6layers, image_size=83,
                              color_channels=1, depth_channels=1).visualize(
            self.tfrecord_dir, self.log_dir_fusion,
            self.visualize_dir_single,
            batch_size=5000, split_name=split_name, batch_stat=True)

    def visualize_fusionAE_both(self, split_name='validation'):
        VisualizeColorAndDepth(fusionAE_6layers, image_size=83,
                               color_channels=1, depth_channels=1).visualize(
            self.tfrecord_dir, self.log_dir_fusion,
            self.visualize_dir_both,
            batch_size=5000, split_name=split_name, batch_stat=True)

    def visualize_fusion_CNN_single(self, split_name='validation'):
        VisualizeColorOrDepth(fusion_CNN_10layers, image_size=83,
                              color_channels=1, depth_channels=1).visualize(
            self.tfrecord_dir, self.log_dir_classify_CNN,
            self.visualize_dir_CNN_single, batch_size=5000,
            split_name=split_name, batch_stat=True, endpoint=None)

    def visualize_fusion_CNN_both(self, split_name='validation'):
        VisualizeColorAndDepth(fusion_CNN_10layers, image_size=83,
                               color_channels=1, depth_channels=1).visualize(
            self.tfrecord_dir, self.log_dir_classify_CNN,
            self.visualize_dir_CNN_both, batch_size=5000,
            split_name=split_name, batch_stat=True, endpoint=None)
