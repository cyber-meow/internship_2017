"""Train and visulize a common embedding.

We map linearly the represntations from two different modalities
(color/depth) in a common space and try to minimize some loss function.
For example, the distance between pairs of points representing
corresponding color and depth images. Nonetheless, the problem is in
fact more difficult and I wasn't able to find a good loss function
that allows me to learn a good embedding.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from multimodal.gesture.basics import TrainColorDepth, VisualizeColorDepth

slim = tf.contrib.slim


class TrainEmbedding(TrainColorDepth):

    @property
    def default_trainable_scopes(self):
        return ['Embedding']

    def __init__(self, architecture, **kwargs):
        super(TrainEmbedding, self).__init__(**kwargs)
        self.architecture = architecture

    def compute(self, **kwargs):
        self.color_repr, self.depth_repr = self.compute_embedding(
            self.images_color, self.images_depth, **kwargs)

    def compute_embedding(self, color_inputs, depth_inputs,
                          feature_length=512,
                          color_endpoint='Middle', depth_endpoint='Middle'):
        """Map color and depth representations to a common embedding space.

        Args:
            color_inputs, depth_inputs: `Tensors` for inputs.
            feature_length: The dimension of the embedding space.
            color_endpoint, depth_endpoint: used by `self.architecture`
                to compute representation.
        """
        with tf.variable_scope('Color'):
            color_net = self.architecture(
                color_inputs, final_endpoint=color_endpoint)
        with tf.variable_scope('Depth'):
            depth_net = self.architecture(
                depth_inputs, final_endpoint=depth_endpoint)
        color_net = slim.flatten(color_net)
        self.pre_color_repr = color_net
        depth_net = slim.flatten(depth_net)
        self.pre_depth_repr = depth_net
        with tf.variable_scope('Embedding'):
            color_net = slim.fully_connected(
                color_net, feature_length, activation_fn=None, scope='Color')
            depth_net = slim.fully_connected(
                depth_net, feature_length, activation_fn=None, scope='Depth')
            return color_net, depth_net

    def get_total_loss(self):
        """Define some loss function.

        Use cos distance between pairs of corresponding images, but
        also try to get embedding vectors uniformly distributed
        by penalizing points of the same modality that are too close.
        Finally also try to preserve cos distances between different pairs.
        """
        color_repr = slim.unit_norm(self.color_repr, 1)
        depth_repr = slim.unit_norm(self.depth_repr, 1)
        pre_color_repr = slim.unit_norm(self.pre_color_repr, 1)
        pre_depth_repr = slim.unit_norm(self.pre_depth_repr, 1)
        self.cos_loss = -10 * tf.reduce_sum(color_repr*depth_repr)
        tf.losses.add_loss(self.cos_loss)
        self.color_preserved_loss = self.preserved_distance_loss(
            pre_color_repr, color_repr, 2)
        self.depth_preserved_loss = self.preserved_distance_loss(
            pre_depth_repr, depth_repr, 0.1)
        self.color_far_loss = self.far_loss(color_repr)
        self.depth_far_loss = self.far_loss(depth_repr)
        self.total_loss = tf.losses.get_total_loss()
        return self.total_loss

    def preserved_distance_loss(self, inputs, reprs, w):
        inputs_rot = tf.concat([tf.expand_dims(inputs[-1], 0), inputs[:-1]], 0)
        old_distances = tf.reduce_sum(inputs*inputs_rot, axis=1)
        reprs_rot = tf.concat([tf.expand_dims(reprs[-1], 0), reprs[:-1]], 0)
        new_distances = tf.reduce_sum(reprs*reprs_rot, axis=1)
        distance_diff = w * tf.reduce_sum(
            tf.square(new_distances - old_distances))
        tf.losses.add_loss(distance_diff)
        return distance_diff

    def far_loss(self, inputs):
        inputs_rot = tf.concat([tf.expand_dims(inputs[-1], 0), inputs[:-1]], 0)
        far_loss = tf.reduce_sum(inputs*inputs_rot)
        tf.losses.add_loss(far_loss)
        return far_loss

    def get_summary_op(self):
        self.get_batch_norm_summary()
        tf.summary.scalar('losses/train/cos', self.cos_loss)
        tf.summary.scalar('losses/train/color_preserved',
                          self.color_preserved_loss)
        tf.summary.scalar('losses/train/depth_preserved',
                          self.depth_preserved_loss)
        tf.summary.scalar('losses/train/color_far', self.color_far_loss)
        tf.summary.scalar('losses/train/depth_far', self.depth_far_loss)
        tf.summary.scalar('losses/train/total', self.total_loss)
        tf.summary.scalar('learning_rate', self.learning_rate)
        tf.summary.image('train/color', self.images_color)
        tf.summary.image('train/depth', self.images_depth)
        self.summary_op = tf.summary.merge_all()
        return self.summary_op

    def get_test_summary_op(self):
        ls_cos = tf.summary.scalar('losses/test/cos', self.cos_loss)
        ls_tl = tf.summary.scalar('losses/test/total', self.total_loss)
        img_clr = tf.summary.image('test/color', self.images_color)
        img_dep = tf.summary.image('test/depth', self.images_depth)
        self.test_summary_op = tf.summary.merge(
            [ls_cos, ls_tl, img_clr, img_dep])
        return self.test_summary_op

    def get_init_fn(self, checkpoint_dirs):
        checkpoint_dir_color, checkpoint_dir_depth = checkpoint_dirs

        variables_color = {}
        variables_depth = {}

        for var in tf.model_variables():
            if var.op.name.startswith('Color'):
                variables_color[var.op.name[6:]] = var
            if var.op.name.startswith('Depth'):
                variables_depth[var.op.name[6:]] = var

        saver_color = tf.train.Saver(variables_color)
        saver_depth = tf.train.Saver(variables_depth)

        checkpoint_path_color = tf.train.latest_checkpoint(
            checkpoint_dir_color)
        checkpoint_path_depth = tf.train.latest_checkpoint(
            checkpoint_dir_depth)

        def restore(sess):
            saver_color.restore(sess, checkpoint_path_color)
            saver_depth.restore(sess, checkpoint_path_depth)
        return restore

    def summary_log_info(self, sess):
        self.loss, _, summaries = self.train_step(
            sess, self.train_op, self.sv.global_step, self.summary_op)
        self.sv.summary_computed(sess, summaries)

    def test_log_info(self, sess, test_use_batch):
        ls, summaries_test = sess.run(
            [self.total_loss, self.test_summary_op],
            feed_dict={self.training: False,
                       self.batch_stat: test_use_batch})
        tf.logging.info('Current Test Loss: %s', ls)
        self.sv.summary_computed(sess, summaries_test)


class TrainEmbedding2(TrainEmbedding):
    """Like `TrainEmbedding` but with a different loss function."""

    def get_total_loss(self):
        """Loss function of `TrainEmbedding` without the distance
        preserving part."""
        color_repr = slim.unit_norm(self.color_repr, 1)
        depth_repr = slim.unit_norm(self.depth_repr, 1)
        self.cos_loss = -tf.reduce_sum(color_repr*depth_repr)
        self.color_far_loss = self.far_loss(color_repr)
        self.depth_far_loss = self.far_loss(depth_repr)
        tf.losses.add_loss(self.cos_loss)
        self.total_loss = tf.losses.get_total_loss()
        return self.total_loss

    def far_loss(self, inputs):
        inputs_rot = tf.concat([tf.expand_dims(inputs[-1], 0), inputs[:-1]], 0)
        far_loss = tf.reduce_sum(inputs*inputs_rot)
        tf.losses.add_loss(far_loss)
        return far_loss

    def get_summary_op(self):
        self.get_batch_norm_summary()
        tf.summary.scalar('losses/train/cos', self.cos_loss)
        tf.summary.scalar('losses/train/color_far', self.color_far_loss)
        tf.summary.scalar('losses/train/depth_far', self.depth_far_loss)
        tf.summary.scalar('losses/train/total', self.total_loss)
        tf.summary.scalar('learning_rate', self.learning_rate)
        tf.summary.image('train/color', self.images_color)
        tf.summary.image('train/depth', self.images_depth)
        self.summary_op = tf.summary.merge_all()
        return self.summary_op

    def get_test_summary_op(self):
        ls_cos = tf.summary.scalar('losses/test/cos', self.cos_loss)
        ls_tl = tf.summary.scalar('losses/test/total', self.total_loss)
        img_clr = tf.summary.image('test/color', self.images_color)
        img_dep = tf.summary.image('test/depth', self.images_depth)
        self.test_summary_op = tf.summary.merge(
            [ls_cos, ls_tl, img_clr, img_dep])
        return self.test_summary_op


class VisualizeCommonEmbedding(VisualizeColorDepth):
    """Visualize the learned embeddding space for color and depth images."""

    def compute(self, feature_length=512, unit_normalization=True,
                color_endpoint='Middle', depth_endpoint='Middle'):

        with tf.variable_scope('Color'):
            color_net = self.architecture(
                self.images_color, final_endpoint=color_endpoint)
        with tf.variable_scope('Depth'):
            depth_net = self.architecture(
                self.images_depth, final_endpoint=depth_endpoint)
        color_net = slim.flatten(color_net)
        depth_net = slim.flatten(depth_net)

        with tf.variable_scope('Embedding'):
            color_net = slim.fully_connected(
                color_net, feature_length, activation_fn=None, scope='Color')
            depth_net = slim.fully_connected(
                depth_net, feature_length, activation_fn=None, scope='Depth')

        if unit_normalization:
            color_net = slim.unit_norm(color_net, 1)
            depth_net = slim.unit_norm(depth_net, 1)

        self.representations = tf.concat([color_net, depth_net], 0)

        self.repr_var = tf.Variable(
            tf.zeros_like(self.representations),
            name='Representation/All')
        self.repr_var_color = tf.Variable(
            tf.zeros_like(color_net), name='Representation/Color')
        self.repr_var_depth = tf.Variable(
            tf.zeros_like(depth_net), name='Representation/Depth')

        self.assign = tf.group(
            tf.assign(self.repr_var, self.representations),
            tf.assign(self.repr_var_color, color_net),
            tf.assign(self.repr_var_depth, depth_net))

        self.saver_repr = tf.train.Saver(
            [self.repr_var, self.repr_var_color, self.repr_var_depth])

    def config_embedding(self, sess, log_dir):

        _, labels = sess.run([self.assign, self.labels])
        self.saver_repr.save(sess, os.path.join(log_dir, 'repr.ckpt'))

        metadata = os.path.abspath(os.path.join(log_dir, 'metadata.tsv'))
        with open(metadata, 'w') as metadata_file:
            metadata_file.write('index\tlabel\n')
            for index, label in enumerate(labels):
                metadata_file.write('%d\tcolor[%d]\n' % (index, label))
            for index, label in enumerate(labels):
                metadata_file.write(
                    '%d\tdepth[%d]\n' % (index, label))

        metadata_single = os.path.abspath(
            os.path.join(log_dir, 'metadata_single.tsv'))
        with open(metadata_single, 'w') as metadata_file:
            metadata_file.write('index\tlabel\n')
            for index, label in enumerate(labels):
                metadata_file.write('%d\t[%d]\n' % (index, label))

        config = projector.ProjectorConfig()

        embedding = config.embeddings.add()
        embedding.tensor_name = self.repr_var.name
        embedding.metadata_path = metadata

        embedding_color = config.embeddings.add()
        embedding_color.tensor_name = self.repr_var_color.name
        embedding_color.metadata_path = metadata_single

        embedding_depth = config.embeddings.add()
        embedding_depth.tensor_name = self.repr_var_depth.name
        embedding_depth.metadata_path = metadata_single

        projector.visualize_embeddings(
            tf.summary.FileWriter(log_dir), config)
