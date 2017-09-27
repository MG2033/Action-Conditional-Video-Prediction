import tensorflow as tf
from layers import conv2d, conv2d_transpose, pad, flatten, dense, action_transform, get_deconv_filter


class ACVP:
    """ACVP is implemented here!"""

    def __init__(self, args, prediction_steps, reuse=False):
        self.args = args
        self.X = None
        self.y = None
        self.network_template = None
        self.logits = None
        self.is_training = None
        self.loss = None
        self.train_op = None
        self.summaries_merged = None
        self.batch_size = self.args.batch_size if self.args.train_or_test == "train" else 1
        self.reuse = reuse

        # Learning rate changes according to the phase of training
        # with momentum of 0.9, (squared) gradient momentum of 0.95, and min squared gradient of 0.01
        self.RMSProp_params = (0.9, 0.95, 0.01)

        # Max prediction step objective
        self.prediction_steps = prediction_steps

        self.__build()

    def __init_input(self):
        with tf.variable_scope('input'):
            self.X = tf.placeholder(tf.float32, [self.batch_size, self.args.img_height, self.args.img_width,
                                                 self.args.num_stack * self.args.num_channels])
            self.y = tf.placeholder(tf.float32,
                                    [self.batch_size, self.prediction_steps, self.args.img_height, self.args.img_width,
                                     self.args.num_channels])
            self.learning_rate = tf.placeholder(tf.float32)
            self.actions = tf.placeholder(tf.float32,
                                          [self.batch_size, self.prediction_steps, self.args.num_actions])
            self.is_training = tf.placeholder(tf.bool)

    def __init_template(self, x, actions):
        with tf.variable_scope('template'):
            conv1 = conv2d('conv1', x, num_filters=64, kernel_size=(8, 8), padding='SAME', stride=(2, 2),
                           activation=tf.nn.relu, batchnorm_enabled=self.args.batchnorm_enabled,
                           l2_strength=self.args.l2_strength, bias=self.args.bias,
                           is_training=self.is_training)

            conv2 = conv2d('conv2', conv1, num_filters=128, kernel_size=(6, 6), padding='SAME', stride=(2, 2),
                           activation=tf.nn.relu, batchnorm_enabled=self.args.batchnorm_enabled,
                           l2_strength=self.args.l2_strength, bias=self.args.bias,
                           is_training=self.is_training)

            conv3 = conv2d('conv3', conv2, num_filters=128, kernel_size=(6, 6), padding='SAME', stride=(2, 2),
                           activation=tf.nn.relu, batchnorm_enabled=self.args.batchnorm_enabled,
                           l2_strength=self.args.l2_strength, bias=self.args.bias,
                           is_training=self.is_training)

            conv4 = conv2d('conv4', conv3, num_filters=128, kernel_size=(4, 4), padding='SAME', stride=(2, 2),
                           activation=tf.nn.relu, batchnorm_enabled=self.args.batchnorm_enabled,
                           l2_strength=self.args.l2_strength, bias=self.args.bias,
                           is_training=self.is_training)

            flattened = flatten(conv4)

            dense1 = dense('dense1', flattened, output_dim=2048, activation=tf.nn.relu,
                           batchnorm_enabled=self.args.batchnorm_enabled, l2_strength=self.args.l2_strength,
                           bias=self.args.bias, is_training=self.is_training)

            h, w = conv4.get_shape()[1].value, conv4.get_shape()[2].value
            features_with_actions = action_transform(dense1, actions, factors=2048,
                                                     output_dim=h * w * 128,
                                                     final_activation=tf.nn.relu,
                                                     batchnorm_enabled=self.args.batchnorm_enabled,
                                                     l2_strength=self.args.l2_strength, bias=self.args.bias,
                                                     is_training=self.is_training)

            f_a_reshaped = tf.reshape(features_with_actions, [-1, h, w, 128], 'f_a_reshaped')

            h2, w2 = conv3.get_shape()[1].value, conv3.get_shape()[2].value
            deconv1 = conv2d_transpose('deconv1', f_a_reshaped, output_shape=[self.batch_size, h2, w2, 128],
                                       kernel_size=(4, 4),
                                       padding='SAME', stride=(2, 2),
                                       l2_strength=self.args.l2_strength, bias=self.args.bias, activation=tf.nn.relu,
                                       batchnorm_enabled=self.args.batchnorm_enabled, is_training=self.is_training)

            h3, w3 = conv2.get_shape()[1].value, conv2.get_shape()[2].value
            deconv2 = conv2d_transpose('deconv2', deconv1, output_shape=[self.batch_size, h3, w3, 128],
                                       kernel_size=(6, 6),
                                       padding='SAME', stride=(2, 2),
                                       l2_strength=self.args.l2_strength, bias=self.args.bias, activation=tf.nn.relu,
                                       batchnorm_enabled=self.args.batchnorm_enabled, is_training=self.is_training)

            h4, w4 = conv1.get_shape()[1].value, conv1.get_shape()[2].value
            deconv3 = conv2d_transpose('deconv3', deconv2, output_shape=[self.batch_size, h4, w4, 128],
                                       kernel_size=(6, 6),
                                       padding='SAME', stride=(2, 2),
                                       l2_strength=self.args.l2_strength, bias=self.args.bias, activation=tf.nn.relu,
                                       batchnorm_enabled=self.args.batchnorm_enabled, is_training=self.is_training)

            deconv4 = conv2d_transpose('deconv4', deconv3,
                                       output_shape=[self.batch_size, self.args.img_height, self.args.img_width,
                                                     self.args.num_channels],
                                       kernel_size=(8, 8),
                                       padding='SAME', stride=(2, 2),
                                       l2_strength=self.args.l2_strength, bias=self.args.bias, activation=None,
                                       batchnorm_enabled=self.args.batchnorm_enabled, is_training=self.is_training)
            return deconv4

    def __init_network(self):
        with tf.variable_scope('full_network', reuse=self.reuse):
            self.network_template = tf.make_template('network_template', self.__init_template)
            all_obs = [None for _ in range(self.prediction_steps)]
            current_input = self.X
            for i in range(-1, self.prediction_steps - 1):
                if i == -1:
                    all_obs[i + 1] = self.network_template(current_input, self.actions[:, i + 1, :])
                else:
                    current_input = tf.concat([all_obs[i], current_input[:, :, :, :-3]], axis=-1)
                    all_obs[i + 1] = self.network_template(current_input, self.actions[:, i + 1, :])

            self.obs_stacked = tf.stack(all_obs[:self.prediction_steps], axis=1)

            self.loss = tf.reduce_mean((1 / self.prediction_steps) * tf.nn.l2_loss(self.obs_stacked - self.y))

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate,
                                                      momentum=self.RMSProp_params[0], decay=0.95)
                params = tf.trainable_variables()
                grads = tf.gradients(self.loss, params)
                # Minimum squared gradient is 0.01
                grads = [tf.sign(grads[i]) * tf.sqrt(tf.maximum(tf.square(grads[i]), 0.01)) for i in range(len(grads))]
                grads = list(zip(grads, params))
                self.train_op = optimizer.apply_gradients(grads)

    def __build(self):
        self.__init_global_epoch()
        self.__init_global_step()
        self.__init_input()
        self.__init_network()

    def __init_global_epoch(self):
        """
        Create a global epoch tensor to totally save the process of the training
        :return:
        """
        with tf.variable_scope('global_epoch'):
            self.global_epoch_tensor = tf.Variable(-1, trainable=False, name='global_epoch')
            self.global_epoch_input = tf.placeholder('int32', None, name='global_epoch_input')
            self.global_epoch_assign_op = self.global_epoch_tensor.assign(self.global_epoch_input)

    def __init_global_step(self):
        """
        Create a global step variable to be a reference to the number of iterations
        :return:
        """
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
            self.global_step_input = tf.placeholder('int32', None, name='global_step_input')
            self.global_step_assign_op = self.global_step_tensor.assign(self.global_step_input)
