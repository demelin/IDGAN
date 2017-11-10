import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer as xi
from IDGAN.src.codebase.util import prelu, batch_norm_wrapper, layer_norm_wrapper


class EncodingsClassifier(object):
    """ A light-weight(ish) MLP classifier used as a binary discriminator within the GAN setup. """

    def __init__(self, opt, name):
        # Declare attributes
        self.opt = opt
        self.name = name
        self.int_type = tf.int32
        self.float_type = tf.float32

        # Specify normalization function, as WGAN-GP implementation uses layer normalization
        if self.opt.gan_type == 'WGANGP':
            self.normalizer = layer_norm_wrapper
        else:
            self.normalizer = batch_norm_wrapper

        # Declare input placeholders
        self.static_keep_prob = self.inputs_subgraph()

    def inputs_subgraph(self):
        """ Specifies inputs supplied to the model during graph execution. """
        with tf.name_scope('inputs'), tf.device('/cpu:0'):
            # Read-in values from the session inputs
            static_keep_prob = tf.placeholder(dtype=self.float_type, name='static_keep_prob')
        return static_keep_prob

    def __call__(self, input_data, reuse=False):
        """ Builds a feed-forward, fully connected multilayer perceptron. """
        with tf.variable_scope(self.name) as scope, tf.device('/gpu:0'):
            if reuse:
                # Reuse parameters across object instantiations
                scope.reuse_variables()

            assert (self.opt.disc_hidden_list[-1] == 1), \
                'Hidden dimension of the final layer has to equal 1 when employed as GAN discriminator!'

            def _shortcut(layer_input, layer_output, dims_in, dims_out, weight_1, weight_2, is_train):
                """ Defines shortcut connections enabled for the WGAN-GP objective, so as to increase
                the critic's modeling capacity; See arxiv.org/abs/1603.05027 for background on theory;
                See chatbotslife.com/resnets-highwaynets-and-densenets-oh-my-9bb15918ee32 for example code. """
                normalized_1 = self.normalizer(layer_input, dims_in, 'shortcut_normalized_1', is_train)
                activation_1 = tf.matmul(prelu(normalized_1, 'shortcut_activation_1', self.float_type), weight_1)
                normalized_2 = self.normalizer(activation_1, dims_out, 'shortcut_normalized_2', is_train)
                activation_2 = tf.matmul(prelu(normalized_2, 'shortcut_activation_2', self.float_type), weight_2)
                return tf.add(activation_2, layer_output, name='shortcut_output')

            def _layer(layer_input, dims_in, dims_out, layer_id):
                """ Constructs a single layer of the feed-forward network. """
                scope_id = self.name + '_layer_{:d}'.format(layer_id)
                with tf.variable_scope(scope_id):
                    # Normalize network input for GAN training; see github.com/soumith/ganhacks
                    if layer_id == 1:
                        layer_input = tf.tanh(layer_input)
                    # Define the matrix multiplication at the basis of each layer
                    layer_weight = tf.get_variable(name='layer_weight', shape=[dims_in, dims_out],
                                                   initializer=xi(uniform=False, dtype=self.float_type), trainable=True)
                    output = tf.matmul(layer_input, layer_weight)
                    # Optionally apply activation and normalization function, shortcuts, and dropout
                    if layer_id < (len(self.opt.disc_hidden_list) - 1):
                        # Normalization
                        output = self.normalizer(output, dims_out, 'normalized', self.opt.is_train)
                        output = prelu(output, scope_id, self.float_type)
                        # Shortcut connections
                        if self.opt.enable_shortcuts:
                            sc_weight_1 = tf.get_variable(name='shortcut_weight_1', shape=[dims_in, dims_out],
                                                          initializer=xi(uniform=False, dtype=self.float_type),
                                                          trainable=True)
                            sc_weight_2 = tf.get_variable(name='shortcut_weight_2', shape=[dims_out, dims_out],
                                                          initializer=xi(uniform=False, dtype=self.float_type),
                                                          trainable=True)
                            output = _shortcut(layer_input, output, dims_in, dims_out, sc_weight_1, sc_weight_2,
                                               self.opt.is_train)
                        # Dropout disabled for the final layer
                        output = tf.nn.dropout(output, self.static_keep_prob, name='dropout')
                    # Sigmoid point-wise non-linearity applied to output for standard GAN objective
                    if layer_id == (len(self.opt.disc_hidden_list) - 1) and self.opt.gan_type == 'NLLGAN':
                        output = tf.sigmoid(output)
                    return output

            # Build layers according to dimensions specified in the provided options object
            assert(len(self.opt.disc_hidden_list) >= 2), 'Discriminator construction requires the input and output ' \
                                                         'dimensions of at least one hidden layer to be specified.'
            layer_out = input_data
            for l_id in range(1, len(self.opt.disc_hidden_list)):
                layer_out = _layer(
                    layer_out, self.opt.disc_hidden_list[l_id - 1], self.opt.disc_hidden_list[l_id], l_id)
            return layer_out
