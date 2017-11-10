""" An implementation of the siamese RNN for sentence similarity classification outlined in Mueller et al.,
'Siamese Recurrent Architectures for Learning Sentence Similarity.' Present implementation is a modified variant of
https://github.com/dhwajraj/deep-siamese-text-similarity/blob/master/siamese_network.py"""

import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell, DropoutWrapper, MultiRNNCell
from tensorflow.contrib.layers import xavier_initializer as xi


class Encoder(object):
    """ RNN-LSTM encodes used to encode input sentences into dense, continuous representations used in
    content similarity estimation by the similarity classifier."""

    def __init__(self, vocab, opt, name, rnn_reuse):
        # Declare attributes
        self.vocab = vocab
        self.opt = opt
        self.name = name
        self.rnn_reuse = rnn_reuse
        self.float_type = tf.float32
        self.int_type = tf.int32

        with tf.variable_scope(self.name):
            # Build graph
            # inputs | properties | embeddings | lstm_rnn | attention
            self.input_idx, self.static_keep_prob, self.rnn_keep_prob = self.inputs_subgraph()
            self.batch_length, self.batch_steps, self.length_mask = self.properties_subgraph()
            self.embedding_table, self.input_data = self.embeddings_subgraph()
            self.rnn_outputs = self.lstm_rnn_subgraph()
            self.sentence_encodings = self.attention_subgraph()
            self.summary_list = self.summaries()

    def inputs_subgraph(self):
        """ Specifies inputs supplied to the model during graph execution. """
        with tf.name_scope('inputs'), tf.device('/cpu:0'):
            # Read-in input data and dropout probabilities from session input
            input_idx = tf.placeholder(shape=[None, None], dtype=self.int_type, name='input_idx')
            static_keep_prob = tf.placeholder(dtype=self.float_type, name='static_keep_prob')
            rnn_keep_prob = tf.placeholder(dtype=self.float_type, name='rnn_keep_prob')
        return input_idx, static_keep_prob, rnn_keep_prob

    def properties_subgraph(self):
        """ Returns the properties of the input data relevant to the model's operation. """
        with tf.name_scope('properties'), tf.device('/cpu:0'):
            batch_length = tf.shape(self.input_idx)[0]
            batch_steps = tf.shape(self.input_idx)[1]
            # Determine lengths of individual input sequences within the processed batch to mask RNN output and
            # exclude <EOS> and <PAD> tokens from contributing to the sentence encoding
            length_mask = tf.count_nonzero(
                tf.not_equal(self.input_idx, self.vocab.pad_id), axis=1, keep_dims=False, name='length_mask')
        return batch_length, batch_steps, length_mask

    def embeddings_subgraph(self):
        """ Instantiates the embedding table and the embedding lookup operation. """
        with tf.variable_scope('embeddings'), tf.device('/cpu:0'):
            embedding_table = tf.get_variable(name='embedding_table',
                                              shape=[self.vocab.n_words, self.opt.embedding_dims],
                                              dtype=self.float_type,
                                              initializer=xi(uniform=False, dtype=self.float_type),
                                              trainable=True)
            # Embed input indices
            input_data = tf.nn.embedding_lookup(embedding_table, self.input_idx, name='embeddings')
            # Optionally apply dropout (at training time only)
            if self.opt.is_train:
                input_data = tf.nn.dropout(input_data, self.static_keep_prob, name='front_dropout')
        return embedding_table, input_data

    def lstm_rnn_subgraph(self):
        """ Defines the forward pass through the specified LSTM-RNN. """
        with tf.variable_scope('lstm_rnn'), tf.device('/gpu:0'):
            # Helper function for defining the RNN cell;
            # here, LSTMs are used
            def _lstm_cell(model_opt, model_rnn_reuse):
                """ Defines a basic LSTM cell to which various wrappers can be applied. """
                base_cell = BasicLSTMCell(model_opt.hidden_dims, forget_bias=2.5, state_is_tuple=True,
                                          reuse=model_rnn_reuse)
                if model_opt.is_train:
                    base_cell = DropoutWrapper(base_cell, output_keep_prob=self.rnn_keep_prob)
                return base_cell

            # Helper function resets RNN stats of the forward and backward cells between mini-batches
            def _get_zero_state(source_cell_fw, source_cell_bw):
                """ Returns the zeroed initial state for the source LSTM cell. """
                zero_fw = source_cell_fw.zero_state(self.batch_length, self.float_type)
                zero_bw = source_cell_bw.zero_state(self.batch_length, self.float_type)
                return zero_fw, zero_bw

            # Instantiate number of layers according to value specified in options
            if self.opt.num_layers > 1:
                cell_fw = MultiRNNCell([_lstm_cell(self.opt, self.rnn_reuse) for _ in range(self.opt.num_layers)])
                cell_bw = MultiRNNCell([_lstm_cell(self.opt, self.rnn_reuse) for _ in range(self.opt.num_layers)])
            else:
                cell_fw = _lstm_cell(self.opt, self.rnn_reuse)
                cell_bw = _lstm_cell(self.opt, self.rnn_reuse)
            initial_state_fw, initial_state_bw = _get_zero_state(cell_fw, cell_bw)
            # Obtain RNN output for the current mini-batch
            # time-major format == [batch_size, step_num, hidden_size]
            bi_outputs, bi_states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, self.input_data,
                                                                    sequence_length=self.length_mask,
                                                                    initial_state_fw=initial_state_fw,
                                                                    initial_state_bw=initial_state_bw,
                                                                    dtype=self.float_type, parallel_iterations=False,
                                                                    swap_memory=True, time_major=False)
            # RNN output is a concatenation of the outputs obtained from the forward and backward cell(s)
            rnn_outputs = tf.concat(bi_outputs, 2, name='concatenated_rnn_outputs')
        return rnn_outputs

    def attention_subgraph(self):
        """ Designates the parameters for the self-attention mechanism used to obtain improved sentence encodings
        and applies attention to the encoder output at each time-step. """
        with tf.variable_scope('attention'), tf.device('/gpu:0'):
            projection_weights = tf.get_variable(name='projection_weights',
                                                 shape=[self.opt.hidden_dims * 2, self.opt.attention_dims],
                                                 initializer=xi(uniform=False, dtype=self.float_type),
                                                 trainable=True)
            projection_biases = tf.get_variable(name='projection_biases', shape=[self.opt.attention_dims],
                                                initializer=tf.zeros_initializer(dtype=self.float_type),
                                                trainable=True)
            context_vector = tf.get_variable(name='context_vector', shape=[self.opt.attention_dims],
                                             initializer=xi(uniform=False, dtype=self.float_type), trainable=True)
            # Publication: see www.cs.cmu.edu/~diyiy/docs/naacl16.pdf
            # Publication code: github.com/ematvey/hierarchical-attention-networks/blob/master/HAN_model.py
            # Compute attention values
            memory_values = tf.reshape(self.rnn_outputs, shape=[-1, self.opt.hidden_dims * 2], name='memory_values')
            projected_memories = tf.nn.tanh(
                tf.nn.xw_plus_b(memory_values, projection_weights, projection_biases), name='projected_memories')
            projected_memories = tf.reshape(projected_memories, shape=[self.batch_length, self.batch_steps, -1])
            # Mask out positions corresponding to padding within the input
            score_mask = tf.sequence_mask(self.length_mask, maxlen=tf.reduce_max(self.length_mask),
                                          dtype=self.float_type)
            score_mask = tf.expand_dims(score_mask, -1)
            score_mask = tf.matmul(
                score_mask, tf.ones([self.batch_length, self.opt.attention_dims, 1]), transpose_b=True)
            projected_memories = tf.where(
                tf.cast(score_mask, dtype=tf.bool), projected_memories, tf.zeros_like(projected_memories))
            # Calculate the importance of the individual encoder hidden states for the informativeness of the
            # computed sentence representation
            context_product = tf.reduce_sum(
                tf.multiply(projected_memories, context_vector, name='context_product'), axis=2, keep_dims=True)
            attention_weights = tf.nn.softmax(context_product, dim=1, name='importance_weight')
            # Weigh encoder hidden states according to the calculated importance weights
            weighted_memories = tf.multiply(projected_memories, attention_weights)
            # Sentence encodings are the importance-weighted sums of encoder hidden states / word representations
            sentence_encodings = tf.reduce_sum(weighted_memories, axis=1, name='sentence_encodings')
        return sentence_encodings

    def summaries(self):
        """ Defines and compiles the summaries tracking various model parameters and outputs. """
        with tf.name_scope('summaries'), tf.device('/cpu:0'):
            # Define summaries
            en_ase = tf.summary.histogram(name='attentive_sentence_encodings', values=self.sentence_encodings)
            # Compile summaries
            summary_list = [en_ase]
        return summary_list


class SentSimClassifier(object):
    """ Network used to calculate sentence similarity between two candidate sentences. """

    def __init__(self, vocab, opt, name):
        # Declare attributes
        self.vocab = vocab
        self.opt = opt
        self.name = name
        self.float_type = tf.float32
        self.int_type = tf.int32

        with tf.variable_scope(self.name):
            # Initialize the encoders (a single instantiation is used cross both sentences, i.e. in a 'Siamese' setup)
            self.encoder_a = Encoder(vocab, opt, 'encoder', False)
            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                self.encoder_b = Encoder(vocab, opt, 'encoder', True)
            # Build computational sub-graphs
            # inputs | prediction | loss | optimization
            self.labels, self.lr = self.inputs_subgraph()
            self.predictions = self.prediction_subgraph()
            self.loss, self.loss_avg = self.loss_subgraph()
            self.loss_regularized, self.grads, self.train_op = self.optimization_subgraph()
            # Summaries
            self.train_summaries, self.valid_summaries = self.summaries()

    def inputs_subgraph(self):
        """ Creates the placeholders used to feed data to the model. """
        with tf.name_scope('inputs'), tf.device('/cpu:0'):
            # Read-in values from session input
            labels = tf.placeholder(shape=[None, 1], dtype=self.float_type, name='labels')
            lr = tf.placeholder(dtype=self.float_type, name='learning_rate')
        return labels, lr

    def prediction_subgraph(self):
        """ Calculates the sentence similarity scores predicted by the model. Used at training and inference time. """
        with tf.name_scope('prediction'), tf.device('/gpu:0'):
            # See original publication for exact distance formulation
            predictions = tf.exp(-tf.norm(tf.subtract(
                self.encoder_a.sentence_encodings, self.encoder_b.sentence_encodings), ord=1, axis=-1, keep_dims=True),
                                 name='distance')
        return predictions

    def loss_subgraph(self):
        """ Calculates the mean squared error loss used to train the model. """
        with tf.name_scope('loss'), tf.device('/gpu:0'):
            loss = tf.pow(tf.subtract(self.predictions, self.labels), 2)
            loss_avg = tf.reduce_mean(loss, name='average_mse_loss')
        return loss, loss_avg

    def optimization_subgraph(self):
        """ Defines the optimization procedure for the model. """
        with tf.variable_scope('optimization'), tf.device('/gpu:0'):
            # Variable tracking the optimization steps
            global_step = tf.get_variable(name='global_step', shape=[], dtype=self.int_type,
                                          initializer=tf.constant_initializer(0, dtype=self.int_type), trainable=False)
            # All trainable variables are optimized jointly
            t_vars = tf.trainable_variables()
            # Apply L2 regularization by imposing Gaussian priors on model's parameters
            loss_l2 = tf.add_n([tf.nn.l2_loss(var) for var in t_vars if len(var.shape) > 1]) * self.opt.l2_beta
            loss_regularized = tf.add(self.loss_avg, loss_l2, name='loss_regularized')
            # Calculate gradients for backpropagation with respect to the regularized loss
            grads = tf.gradients(loss_regularized, t_vars)
            clipped_grads, _ = tf.clip_by_global_norm(grads, self.opt.grad_clip_norm, name='clipped_grads')
            # Define optimization OP
            optimizer = tf.train.AdamOptimizer(self.lr)
            train_op = optimizer.apply_gradients(zip(clipped_grads, t_vars), global_step=global_step, name='train_op')
        return loss_regularized, grads, train_op

    def summaries(self):
        """ Defines and compiles model summaries, combining encoder and classifier parameters. """
        with tf.name_scope('summaries'), tf.device('/cpu:0'):
            # Define summaries
            tml = tf.summary.scalar(name='training_loss', tensor=self.loss_avg)
            vml = tf.summary.scalar(name='validation_loss', tensor=self.loss_avg)
            lreg = tf.summary.scalar(name='l2_regularized_loss', tensor=self.loss_regularized)
            lr = tf.summary.scalar(name='learning_rate', tensor=self.lr)
            train_list = [tml, lreg, lr]
            valid_list = [vml]
            train_summaries = tf.summary.merge(
                self.encoder_a.summary_list + self.encoder_b.summary_list + train_list, name='train_summaries')
            valid_summaries = tf.summary.merge(valid_list, name='valid_summaries')
        return train_summaries, valid_summaries
