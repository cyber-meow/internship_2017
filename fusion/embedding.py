from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from routines.train import TrainColorDepth

slim = tf.contrib.slim


class TrainEmbedding(TrainColorDepth):

    @property
    def default_trainable_scopes(self):
        return ['Embedding']

    def compute(self, **kwargs):
        self.color_repr, self.depth_repr = self.compute_embedding(
            self.images_color, self.images_depth, **kwargs)
        self.color_repr_var = tf.Variable(
            tf.zeros([24, 1536], dtype=tf.float32),
            name='Embedding/Color/representation')
        self.ag = tf.assign(self.color_repr_var, self.color_repr)

    def compute_embedding(self, color_inputs, depth_inputs,
                          feature_length=1536):
        color_net, _ = self.structure(
            color_inputs, final_endpoint='Middle', scope='Color')
        depth_net, _ = self.structure(
            depth_inputs, final_endpoint='Middle', scope='Depth')
        color_net = slim.flatten(color_net)
        depth_net = slim.flatten(depth_net)
        with tf.variable_scope('Embedding'):
            color_net = slim.fully_connected(
                color_net, feature_length, scope='Color')
            depth_net = slim.fully_connected(
                depth_net, feature_length, scope='Depth')
            return color_net, depth_net

    def get_total_loss(self):
        self.l2_loss = tf.losses.mean_squared_error(
            self.color_repr, self.depth_repr)
        self.total_loss = tf.losses.get_total_loss()
        return self.total_loss

    def get_metric_op(self):
        self.metric_op = None
        return self.metric_op

    def get_summary_op(self):
        self.get_batch_norm_summary()
        tf.summary.scalar('losses/train/l2', self.l2_loss)
        tf.summary.scalar('losses/train/total', self.total_loss)
        tf.summary.scalar('learning_rate', self.learning_rate)
        tf.summary.image('train/color', self.images_color)
        tf.summary.image('train/depth', self.images_depth)
        self.summary_op = tf.summary.merge_all()
        return self.summary_op

    def get_test_summary_op(self):
        ls_l2 = tf.summary.scalar('losses/test/l2', self.l2_loss)
        ls_tl = tf.summary.scalar('losses/test/total', self.total_loss)
        img_clr = tf.summary.image('test/color', self.images_color)
        img_dep = tf.summary.image('test/depth', self.images_depth)
        self.test_summary_op = tf.summary.merge(
            [ls_l2, ls_tl, img_clr, img_dep])
        return self.test_summary_op

    def get_init_fn(self, checkpoint_dirs):
        """Returns a function run by the chief worker to
           warm-start the training."""
        checkpoint_dir_color, checkpoint_dir_depth = checkpoint_dirs

        variables_color = {}
        variables_depth = {}

        for var in tf.model_variables():
            if var.op.name.startswith('Color'):
                variables_color['CAE'+var.op.name[5:]] = var
            if var.op.name.startswith('Depth'):
                variables_depth['CAE'+var.op.name[5:]] = var

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

    def normal_log_info(self, sess):
        self.loss, _, summaries = self.train_step(
            sess, self.train_op, self.sv.global_step, self.summary_op)
        return summaries

    def test_log_info(self, sess):
        ls, summaries_test = sess.run(
            [self.total_loss, self.test_summary_op],
            feed_dict={self.training: False})
        tf.logging.info('Current Test Loss: %s', ls)
        return summaries_test

    def final_log_info(self, sess):
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = 'Embedding/Color/representation'
        projector.visualize_embeddings(self.sv.summary_writer, config)
        sess.run(self.ag, feed_dict={self.training: False})
        tf.logging.info('Finished training. Final Loss: %s', self.loss)
        tf.logging.info('Saving model to disk now.')
