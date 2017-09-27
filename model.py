import tensorflow as tf
from layers import conv2d, conv2d_transpose, pad, flatten, dense, action_transform


class ACVP:
    """ACVP is implemented here!"""

    def __init__(self, args):
        self.args = args
        self.X = None
        self.y = None
        self.logits = None
        self.is_training = None
        self.loss = None
        self.regularization_loss = None
        self.cross_entropy_loss = None
        self.train_op = None
        self.accuracy = None
        self.y_out_argmax = None
        self.summaries_merged = None
        self.batch_size = self.args.batch_size if self.args.train_or_test == "train" else 1

        self.__build()

    def __init_input(self):
        self.X = tf.placeholder(tf.float32, [self.batch_size, self.args.img_height, self.args.img_width,
                                             self.args.num_stack * self.args.num_channels])
        self.y = tf.placeholder(tf.float32,
                                [self.batch_size, self.args.img_height, self.args.img_width, self.args.num_channels])
        self.actions = tf.placeholder(tf.float32, [self.batch_size, self.args.num_actions])
        self.is_training = tf.placeholder(tf.bool)

    def __init_network(self):
        pad1 = pad('conv1_padding', self.X, 0, 1)
        conv1 = conv2d('conv1', pad1, num_filters=64, kernel_size=(8, 8), padding='VALID', stride=(2, 2),
                       activation=tf.nn.relu, batchnorm_enabled=self.args.batchnorm_enabled,
                       l2_strength=self.args.l2_strength, bias=self.args.bias,
                       is_training=self.is_training)

        pad2 = pad('conv2_padding', conv1, 1, 1)
        conv2 = conv2d('conv2', pad2, num_filters=128, kernel_size=(6, 6), padding='VALID', stride=(2, 2),
                       activation=tf.nn.relu, batchnorm_enabled=self.args.batchnorm_enabled,
                       l2_strength=self.args.l2_strength, bias=self.args.bias,
                       is_training=self.is_training)

        pad3 = pad('conv3_padding', conv2, 1, 1)
        conv3 = conv2d('conv3', pad3, num_filters=128, kernel_size=(6, 6), padding='VALID', stride=(2, 2),
                       activation=tf.nn.relu, batchnorm_enabled=self.args.batchnorm_enabled,
                       l2_strength=self.args.l2_strength, bias=self.args.bias,
                       is_training=self.is_training)

        pad4 = pad('conv4_padding', conv3, 0, 0)
        conv4 = conv2d('conv4', pad4, num_filters=128, kernel_size=(4, 4), padding='VALID', stride=(2, 2),
                       activation=tf.nn.relu, batchnorm_enabled=self.args.batchnorm_enabled,
                       l2_strength=self.args.l2_strength, bias=self.args.bias,
                       is_training=self.is_training)

        flattened = flatten(conv4)

        dense1 = dense('dense1', flattened, output_dim=2048, activation=tf.nn.relu,
                       batchnorm_enabled=self.args.batchnorm_enabled, l2_strength=self.args.l2_strength,
                       bias=self.args.bias, is_training=self.is_training)

        h, w = conv4.get_shape()[1].value, conv4.get_shape()[2].value
        features_with_actions = action_transform(dense1, self.actions, factors=2048,
                                                 output_dim=h * w * 128,
                                                 final_activation=tf.nn.relu,
                                                 batchnorm_enabled=self.args.batchnorm_enabled,
                                                 l2_strength=self.args.l2_strength, bias=self.args.bias,
                                                 is_training=self.is_training)

        f_a_reshaped = tf.reshape(features_with_actions, [-1, h, w, 128], 'f_a_reshaped')

        h2, w2 = conv3.get_shape()[1].value, conv3.get_shape()[2].value
        dpad1 = pad('deconv1_padding', f_a_reshaped, 0, 0)
        deconv1 = conv2d_transpose('deconv1', dpad1, output_shape=[self.batch_size, h2, w2, 128], kernel_size=(4, 4),
                                   padding='VALID', stride=(2, 2),
                                   l2_strength=self.args.l2_strength, bias=self.args.bias, activation=tf.nn.relu,
                                   batchnorm_enabled=self.args.batchnorm_enabled, is_training=self.is_training)

        h3, w3 = conv2.get_shape()[1].value, conv2.get_shape()[2].value
        dpad2 = pad('deconv2_padding', deconv1, 1, 1)
        deconv2 = conv2d_transpose('deconv2', dpad2, output_shape=[self.batch_size, h3, w3, 128], kernel_size=(6, 6),
                                   padding='VALID', stride=(2, 2),
                                   l2_strength=self.args.l2_strength, bias=self.args.bias, activation=tf.nn.relu,
                                   batchnorm_enabled=self.args.batchnorm_enabled, is_training=self.is_training)

        h4, w4 = conv1.get_shape()[1].value, conv1.get_shape()[2].value
        dpad3 = pad('deconv3_padding', deconv2, 1, 1)
        deconv3 = conv2d_transpose('deconv3', dpad3, output_shape=[self.batch_size, h4, w4, 128], kernel_size=(6, 6),
                                   padding='VALID', stride=(2, 2),
                                   l2_strength=self.args.l2_strength, bias=self.args.bias, activation=tf.nn.relu,
                                   batchnorm_enabled=self.args.batchnorm_enabled, is_training=self.is_training)

        dpad4 = pad('deconv4_padding', deconv3, 0, 1)
        deconv4 = conv2d_transpose('deconv4', dpad4,
                                   output_shape=[self.batch_size, self.args.img_height, self.args.img_width,
                                                 self.args.num_channels],
                                   kernel_size=(8, 8),
                                   padding='VALID', stride=(2, 2),
                                   l2_strength=self.args.l2_strength, bias=self.args.bias, activation=None,
                                   batchnorm_enabled=self.args.batchnorm_enabled, is_training=self.is_training)

        self.logits = deconv4

    def __init_output(self):
        pass

    def __build(self):
        self.__init_global_epoch()
        self.__init_global_step()
        self.__init_input()
        self.__init_network()
        self.__init_output()

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
