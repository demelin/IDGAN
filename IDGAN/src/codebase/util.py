""" Modified ReLU activation functions: Leaky ReLU and Parametrized ReLU, plus a batch normalisation wrapper. """

import tensorflow as tf
import numpy as np


def lrelu(input_data, multiplier, scope_id):
    """ An implementation of Leaky ReLU, outlined in 'Rectifier nonlinearities improve neural network acoustic models.'
    by Maas et al. a should be low: nmarkou.blogspot.de/2017/02/the-black-magic-of-deep-learning-tips.html """
    with tf.name_scope('lrelu_{:s}'.format(scope_id)):
        # Get boundaries
        maximum_value = tf.reduce_max(input_data, name='max_value')
        minimum_value = tf.reduce_min(input_data, name='min_value')
        # Clip values
        input_pos = tf.clip_by_value(input_data, clip_value_min=0, clip_value_max=maximum_value, name='min_clipped')
        input_neg = tf.clip_by_value(input_data, clip_value_min=minimum_value, clip_value_max=0, name='max_clipped')
        # Apply negative sloping
        out = tf.add(input_pos, tf.multiply(input_neg, multiplier), name='combined')
    return out


def prelu(input_data, scope_id, dtype):
    """ An implementation of the PReLU activation function following 'Delving Deep into Rectifiers: Surpassing Human-
    Level Performance on ImageNet Classification' by He et al. """
    # Declare adaptive multiplier parameter
    with tf.variable_scope('prelu_{:s}'.format(scope_id)):
        multiplier = tf.get_variable(
            name='multiplier', shape=[], initializer=tf.constant_initializer(0.25, dtype=dtype))
    return lrelu(input_data, multiplier, scope_id)


# Alternatively, for comparison to simplified implementation
# from tensorflow.contrib.layers.python.layers import batch_norm (might have changed with 1.0)

def batch_norm_wrapper(inputs, dims_out, scope_id, is_train, decay=0.999):
    """ A batch norm handler implmentation following r2rt.com/implementing-batch-normalization-in-tensorflow.html.
            inputs = inputs to a layer's activation function
            is_training = conditions the update
            decay = weighting of updates within the moving average and variance calculation.
    """
    # Declare parameters
    with tf.variable_scope('{:s}_batch_norm'.format(scope_id)):
        offset = tf.get_variable(name='offset', shape=[1, dims_out],
                                 initializer=tf.zeros_initializer())
        scale = tf.get_variable(name='scale', shape=[1, dims_out],
                                initializer=tf.ones_initializer())
        population_mean = tf.get_variable(name='pop_mean', shape=[dims_out],
                                          initializer=tf.zeros_initializer(), trainable=False)
        population_var = tf.get_variable(name='pop_var', shape=[dims_out],
                                         initializer=tf.ones_initializer(), trainable=False)
        # Avoids division by zero
        epsilon = 1e-3

        def _batch_bn():
            # Get batch statistics along the batch dimension
            batch_mean, batch_var = tf.nn.moments(inputs, [0])
            # Update population statistics
            train_mean = tf.assign(population_mean, tf.add(tf.multiply(population_mean, decay),
                                                           tf.multiply(batch_mean, 1 - decay)), name='pop_mean_update')
            train_var = tf.assign(population_var, tf.add(tf.multiply(population_var, decay),
                                                         tf.multiply(batch_var, 1 - decay)), name='pop_var_update')
            # Apply batch normalization using the approximation of population statistics
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(inputs, batch_mean, batch_var, offset, scale, epsilon,
                                                 name='training_batch_norm')

        def _pop_bn():
            return tf.nn.batch_normalization(inputs, population_mean, population_var, offset, scale, epsilon,
                                             name='testing_batch_norm')

        if is_train:
            norm_function = _batch_bn()
        else:
            norm_function = _pop_bn()

        return norm_function


def layer_norm_wrapper(inputs, dims_out, scope_id, _):
    """ Layer normalization implementation for use with WGAN-GP models;
    behaves identically during training and inference. """
    # see theneuralperspective.com/2016/10/27/gradient-topics
    with tf.variable_scope('{:s}_layer_norm'.format(scope_id)):
        layer_mean, layer_var = tf.nn.moments(inputs, [1], keep_dims=True)
        offset = tf.get_variable(name='offset', shape=[dims_out],
                                 initializer=tf.zeros_initializer())
        scale = tf.get_variable(name='scale', shape=[dims_out],
                                initializer=tf.ones_initializer())
        epsilon = 1e-5
        normalized = scale * (inputs - layer_mean) / tf.sqrt(layer_var + epsilon) + offset
    return normalized


class EncodingsBuffer(object):
    """ Buffer for storing sentence encodings produced by a GAN generator, used in the discriminator network updates;
    intended to reduce oscillation during GAN training as suggested in arxiv.org/pdf/1703.10593.pdf;
    implementation is comparable to github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/util/image_pool.py. """
    def __init__(self, opt, name, size_factor=50):
        self.opt = opt
        self.name = name
        self.buffer_size = self.opt.batch_size * size_factor  # Paper uses a buffer size of 50 and batch size of 1

        # Initialize the pool
        if self.buffer_size > 0:
            self.buffer = list()
            self.buffered = 0

    def query(self, batch_input):
        """ Adds supplied encodings into the current pool and returns encodings sampled from the present pool. """
        # No buffering
        if self.buffer_size == 0:
            out = batch_input
        else:
            # Split batch into individual encodings
            encodings = tf.split(batch_input, tf.shape(batch_input)[0], axis=0)
            out = list()
            for enc in encodings:
                # Fill buffer
                if self.buffered < self.buffer_size:
                    self.buffered += 1
                    self.buffer.append(enc)
                    out.append(enc)
                # Sample randomly from buffer
                else:
                    flip = np.random.uniform(0.0, 1.0)
                    if flip > 0.5:
                        replace_id = np.random.randint(0, len(self.buffer))
                        out.append(self.buffer[replace_id])
                        self.buffer[replace_id] = enc
                    else:
                        out.append(enc)
        # Return a batch tensor of the same size as the input batch
        return tf.concat(out, axis=0)
