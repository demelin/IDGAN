""" A language model for estimating surprisal scores, possibly also entropy reduction values. Architecture design
roughly follows Zoph's et al. 'Transfer Learning for Low-Resource Neural Machine Translation', 2016.
Added batch normalization and Xavier normalization to stabilize variance of the forward signal."""

import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell, DropoutWrapper, MultiRNNCell
from tensorflow.contrib.layers import xavier_initializer as xi


class CogLM(object):
    """ A multi-layer RNN-LSTM language model; non-callable variant used during pre-training only, as a callable
    implementation allows for easier and more transparent parameter sharing within the full IDGAN architecture. """

    def __init__(self, vocab, opt, name):
        # Declare attributes
        self.vocab = vocab
        self.opt = opt
        self.name = name
        self.int_type = tf.int32
        self.float_type = tf.float32

        with tf.variable_scope(self.name):
            # Build computational sub-graphs (decomposition allows for easy enabling / disabling of individual graphs)
            # inputs | properties | embeddings | lstm_rnn | projection | prediction | sampled_loss | optimization
            self.input_idx, self.labels, self.static_keep_prob, self.rnn_keep_prob, self.lr, self.rnn_state = \
                self.inputs_subgraph()
            self.batch_length, self.batch_steps, self.length_mask = self.properties_subgraph()
            self.embedding_table, self.input_data, self.output_embedding_biases = self.embeddings_subgraph()
            self.final_state, self.flat_rnn_outputs = self.lstm_rnn_subgraph()
            self.projection_weights, self.projection_biases, self.projected_rnn_outputs = self.projection_subgraph()
            self.predictions, self.final_prediction = self.prediction_subgraph()
            self.loss_avg = self.loss_subgraph()
            self.loss_regularized, self.grads, self.train_op = self.optimization_subgraph()
            # Summaries
            self.train_summaries, self.valid_summaries = self.summaries()

    def inputs_subgraph(self):
        """ Creates the placeholders used to feed data to the model. """
        with tf.name_scope('inputs'), tf.device('/cpu:0'):
            # Read-in values from session input
            input_idx = tf.placeholder(shape=[None, None], dtype=self.int_type, name='input_idx')
            labels = tf.placeholder(shape=[None, None], dtype=self.int_type, name='labels')
            static_keep_prob = tf.placeholder(dtype=self.float_type, name='static_keep_prob')
            rnn_keep_prob = tf.placeholder(dtype=self.float_type, name='rnn_keep_prob')
            lr = tf.placeholder(dtype=self.float_type, name='learning_rate')
            # LSTM state placeholder used for incremental generation
            rnn_state = tf.placeholder(
                shape=[self.opt.num_layers, 2, self.opt.batch_size, self.opt.hidden_dims],
                dtype=self.float_type, name='rnn_state')
        return input_idx, labels, static_keep_prob, rnn_keep_prob, lr, rnn_state

    def properties_subgraph(self):
        """ Returns the properties of the input data relevant to the model's operation. """
        with tf.name_scope('properties'), tf.device('/cpu:0'):
            batch_length = tf.shape(self.input_idx)[0]
            batch_steps = tf.shape(self.input_idx)[1]
            # Determine lengths of individual input sequences within the processed batch to mask RNN output and
            # exclude <EOS> tokens from loss calculation
            length_mask = tf.count_nonzero(
                tf.not_equal(self.input_idx, self.vocab.pad_id), axis=1, keep_dims=False, dtype=self.int_type,
                name='length_mask')
        return batch_length, batch_steps, length_mask

    def embeddings_subgraph(self):
        """ Initializes the embedding table and output biases; embedding table is jointly used as the projection matrix
        for projecting the RNN-generated logits into the vocabulary space for inference and loss calculation. """
        with tf.variable_scope('embeddings'), tf.device('/cpu:0'):
            embedding_table = tf.get_variable(name='embedding_table',
                                              shape=[self.vocab.n_words, self.opt.embedding_dims],
                                              dtype=self.float_type,
                                              initializer=xi(uniform=False, dtype=self.float_type),
                                              trainable=True)
            # Embed input indices
            input_data = tf.nn.embedding_lookup(embedding_table, self.input_idx, name='embeddings')
            if self.opt.is_train:
                # Optionally apply dropout (at training time only)
                input_data = tf.nn.dropout(input_data, self.static_keep_prob, name='front_dropout')
            output_embedding_biases = tf.get_variable(name='output_embedding_biases', shape=[self.vocab.n_words],
                                                      dtype=self.float_type,
                                                      initializer=tf.zeros_initializer(dtype=self.float_type),
                                                      trainable=True)
        return embedding_table, input_data, output_embedding_biases

    def lstm_rnn_subgraph(self):
        """ Defines the forward pass through the specified LSTM-RNN. """
        with tf.variable_scope('lstm_rnn'), tf.device('/gpu:0'):
            # Helper function for defining the RNN cell;
            # here, LSTMs are used
            def _lstm_cell(model_opt):
                """ Defines a basic LSTM cell to which various wrappers can be applied. """
                base_cell = BasicLSTMCell(model_opt.hidden_dims, forget_bias=2.5, state_is_tuple=True)
                if model_opt.is_train:
                    base_cell = DropoutWrapper(base_cell, output_keep_prob=self.rnn_keep_prob)
                return base_cell

            # Helper function resets RNN state between mini-batches
            def _get_zero_state(source_cell):
                """ Returns the zeroed initial state for the source LSTM cell. """
                return source_cell.zero_state(self.batch_length, self.float_type)

            # Instantiate number of layers according to value specified in options
            if self.opt.num_layers > 1:
                cell = MultiRNNCell([_lstm_cell(self.opt) for _ in range(self.opt.num_layers)])
            else:
                cell = _lstm_cell(self.opt)
            if self.opt.is_train:
                initial_state = _get_zero_state(cell)
            else:
                # During inference, RNN state is fed via placeholder at each discrete generation step
                # This is done so as to enable beam-search guided generation strategies
                state_list = tf.unstack(self.rnn_state, axis=0)
                initial_state = tuple([tf.contrib.rnn.LSTMStateTuple(
                    state_list[layer_id][0], state_list[layer_id][1]) for layer_id in range(self.opt.num_layers)])
            # Obtain RNN output - i.e. sentence encodings - for the current mini-batch
            # time-major format == [batch_size, step_num, hidden_size]
            rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, self.input_data, sequence_length=self.length_mask,
                                                         initial_state=initial_state, dtype=self.float_type,
                                                         parallel_iterations=None, swap_memory=True, time_major=False)
            # Flatten output for subsequent projection
            flat_rnn_outputs = tf.reshape(rnn_outputs, [-1, self.opt.hidden_dims], name='reshaped_rnn_outputs')
            # Optionally apply dropout (at training time only)
            if self.opt.is_train:
                flat_rnn_outputs = tf.nn.dropout(flat_rnn_outputs, self.static_keep_prob, name='back_dropout')
        return final_state, flat_rnn_outputs

    def projection_subgraph(self):
        """ Defines the weight and bias parameters used to project RNN outputs into the embedding space following
        the completion of each full pass through the RNN """
        with tf.variable_scope('projection'), tf.device('/gpu:0'):
            projection_weights = tf.get_variable(name='projection_weights',
                                                 shape=[self.flat_rnn_outputs.shape[1], self.opt.embedding_dims],
                                                 dtype=self.float_type,
                                                 initializer=xi(uniform=False, dtype=self.float_type),
                                                 trainable=True)
            projection_biases = tf.get_variable(name='projection_biases', shape=[self.opt.embedding_dims],
                                                dtype=self.float_type,
                                                initializer=tf.zeros_initializer(dtype=self.float_type),
                                                trainable=True)
            # Project RNN-output into the embedding space
            projected_rnn_outputs = tf.nn.xw_plus_b(self.flat_rnn_outputs, projection_weights, projection_biases)
            return projection_weights, projection_biases, projected_rnn_outputs

    def loss_subgraph(self):
        """ Calculates the sampled softmax candidate sampling loss used to train the LM. """
        with tf.name_scope('sampled_loss'), tf.device('/gpu:0'):
            # Candidate sampling loss speeds up training times as compared to full-softmax NL calculation
            loss = tf.nn.sampled_softmax_loss(weights=self.embedding_table, biases=self.output_embedding_biases,
                                              labels=tf.reshape(self.labels, [-1, 1]),
                                              inputs=self.projected_rnn_outputs, num_sampled=self.opt.samples,
                                              num_classes=self.vocab.n_words, num_true=1,
                                              sampled_values=self.opt.sampled_values,
                                              remove_accidental_hits=self.opt.remove_accidental_hits,
                                              partition_strategy='div', name='loss')
            loss_avg = tf.reduce_mean(loss, name='sampled_loss_avg')
        return loss_avg

    def optimization_subgraph(self):
        """ Defines the optimization procedure for the model. """
        with tf.variable_scope('optimization'), tf.device('/gpu:0'):
            # Variable tracking the optimization steps
            global_step = tf.get_variable(shape=[], name='global_step', dtype=self.int_type,
                                          initializer=tf.constant_initializer(0, dtype=self.int_type),
                                          trainable=False)
            # All trainable variables are optimized jointly
            t_vars = tf.trainable_variables()
            # Apply L2 regularization by imposing Gaussian priors on model's parameters
            loss_l2 = tf.add_n([tf.nn.l2_loss(var) for var in t_vars if len(var.shape) > 1]) * self.opt.l2_beta
            loss_regularized = tf.add(self.loss_avg, loss_l2, name='loss_regularized')
            # Calculate gradients for backpropagation with respect to the regularized loss
            grads = tf.gradients(loss_regularized, t_vars)
            clipped_grads, _ = tf.clip_by_global_norm(grads, self.opt.grad_clip_norm, name='clipped_grads')
            # Define optimization OP
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.0, epsilon=1e-09, name='optimizer')
            train_op = optimizer.apply_gradients(zip(clipped_grads, t_vars), global_step=global_step, name='train_op')
        return loss_regularized, grads, train_op

    def prediction_subgraph(self):
        """ Defines operations involved in generating sequences of word tokens on the basis of projected RNN output. """
        with tf.variable_scope('prediction'), tf.device('/gpu:0'):
            # Project RNN outputs a second time, now within the vocabulary space
            logits = tf.nn.xw_plus_b(self.projected_rnn_outputs, tf.transpose(self.embedding_table),
                                     self.output_embedding_biases, name='logits')
            # Construct predictive distributions by applying softmax function to logits
            predictions = tf.nn.softmax(logits, name='predictions')
            # Isolate the predictive distribution at the final RNN step (used for beam-search guided generation)
            final_prediction = tf.reshape(predictions, [self.batch_length, self.batch_steps, self.vocab.n_words],
                                          name='final_prediction')[:, -1, :]
        return predictions, final_prediction

    def summaries(self):
        """ Defines and compiles the summaries tracking various model parameters and outputs. """
        with tf.name_scope('summaries'), tf.device('/cpu:0'):
            # Define summaries
            lrs = tf.summary.scalar(name='learning_rate', tensor=self.lr)
            tla = tf.summary.scalar(name='training_loss_avg', tensor=self.loss_avg)
            vla = tf.summary.scalar(name='validation_loss_avg', tensor=self.loss_avg)
            lreg = tf.summary.scalar(name='loss_regularized', tensor=self.loss_regularized)
            # Track gradients
            train_list = [lrs, tla, lreg]
            valid_list = [vla]
            train_summaries = tf.summary.merge(train_list, name='train_summaries')
            valid_summaries = tf.summary.merge(valid_list, name='valid_summaries')
        return train_summaries, valid_summaries
