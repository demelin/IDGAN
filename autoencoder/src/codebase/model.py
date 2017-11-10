""" Some pointers for the raw_rnn implementation were taken from
github.com/ematvey/tensorflow-seq2seq-tutorials/blob/master/2-seq2seq-advanced.ipynb. """

import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell, DropoutWrapper, MultiRNNCell
from tensorflow.contrib.layers import xavier_initializer as xi


class Encoder(object):
    """ RNN-LSTM encoder used within the SAE-dyad to obtain dense, low-dimensional sentence
    encodings; generates sentence embeddings on the basis of sequential input; non-callable variant used during
    pre-training only, as a callable implementation allows for easier and more transparent parameter sharing within the
    full IDGAN architecture. """

    def __init__(self, vocab, opt, name):
        # Declare attributes
        self.vocab = vocab
        self.opt = opt
        self.name = name
        self.float_type = tf.float32
        self.int_type = tf.int32

        # Build computational sub-graphs
        # inputs | properties | embeddings | rnn_lstm | attention | state projection
        with tf.variable_scope(self.name):
            self.input_idx, self.static_keep_prob, self.rnn_keep_prob = self.inputs_subgraph()
            self.batch_length, self.batch_steps, self.length_mask = self.properties_subgraph()
        # Embedding parameters are shared between encoder and decoder
        self.embedding_table, self.input_data, self.output_embedding_biases = self.embeddings_subgraph()
        with tf.variable_scope(self.name):
            self.final_state, self.rnn_outputs = self.lstm_rnn_subgraph()
            if self.opt.attentive_encoding:
                # Unused, as shown to be ineffective in initial experiments
                self.sentence_encodings = self.attention_subgraph()
            self.c_state, self.h_state, self.decoder_state = self.state_projection_subgraph()

    def inputs_subgraph(self):
        """ Creates the placeholders used to feed data to the model. """
        with tf.name_scope('inputs'), tf.device('/cpu:0'):
            # Read-in values from session input
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
            length_mask = tf.count_nonzero(tf.not_equal(self.input_idx, self.vocab.pad_id), axis=1, keep_dims=False,
                                           dtype=self.int_type, name='length_mask')
        return batch_length, batch_steps, length_mask

    def embeddings_subgraph(self):
        """ Initializes the embedding table and output biases; embedding table is jointly used as the projection matrix
        for projecting the RNN-generated logits into the vocabulary space in the decoder. """
        with tf.variable_scope('embeddings'), tf.device('/cpu:0'):
            embedding_table = tf.get_variable(name='embedding_table',
                                              shape=[self.vocab.n_words, self.opt.embedding_dims],
                                              dtype=self.float_type,
                                              initializer=xi(uniform=False, dtype=self.float_type),
                                              trainable=True)
            # Embed input indices
            input_data = tf.nn.embedding_lookup(embedding_table, self.input_idx, name='embeddings')
            if self.opt.allow_dropout:
                input_data = tf.nn.dropout(input_data, self.static_keep_prob, name='enc_front_dropout')
            output_embedding_biases = tf.get_variable(name='output_embedding_biases', shape=[self.vocab.n_words],
                                                      dtype=self.float_type,
                                                      initializer=tf.zeros_initializer(dtype=self.float_type),
                                                      trainable=True)
        return embedding_table, input_data, output_embedding_biases

    def lstm_rnn_subgraph(self):
        """ Defines the forward pass through the encoder LSTM-RNN. """
        with tf.variable_scope('lstm_rnn'), tf.device('/gpu:0'):

            # Helper function for defining the RNN cell; here, LSTMs are used
            def _lstm_cell(model_opt):
                """ Defines a basic LSTM cell to which various wrappers can be applied. """
                base_cell = BasicLSTMCell(model_opt.enc_hidden_dims, forget_bias=2.5, state_is_tuple=True)
                if model_opt.allow_dropout:
                    base_cell = DropoutWrapper(base_cell, output_keep_prob=self.rnn_keep_prob)
                return base_cell

            # Helper function for resetting the RNN state between mini-batches
            def _get_zero_state(source_cell):
                """ Returns the zeroed initial state for the source LSTM cell. """
                zero_state = source_cell.zero_state(self.batch_length, self.float_type)
                return zero_state

            # Instantiate number of layers according to value specified in options
            if self.opt.enc_num_layers > 1:
                cell = MultiRNNCell([_lstm_cell(self.opt) for _ in range(self.opt.enc_num_layers)])
            else:
                cell = _lstm_cell(self.opt)

            initial_state = _get_zero_state(cell)
            # Obtain RNN output - i.e. sentence encodings - for the current mini-batch
            rnn_outputs, rnn_state = tf.nn.dynamic_rnn(cell, self.input_data,
                                                       sequence_length=self.length_mask,
                                                       initial_state=initial_state,
                                                       dtype=self.float_type, parallel_iterations=1,
                                                       swap_memory=True, time_major=False)
            # Combine the final hidden and cell states of the LSTM cells from all layers into single matrices
            if self.opt.enc_num_layers > 1:
                c_state = tf.concat([rnn_state[layer_id].c for layer_id in range(self.opt.enc_num_layers)], -1)
                h_state = tf.concat([rnn_state[layer_id].h for layer_id in range(self.opt.enc_num_layers)], -1)
            else:
                c_state = rnn_state.c
                h_state = rnn_state.h
            # Combine hidden and state matrices into a representation of the RNN-LSTM's state
            final_state = tf.concat([c_state, h_state], 0)

        return final_state, rnn_outputs

    def attention_subgraph(self):
        """ Defines the self-attention mechanism used to obtain improved sentence encodings;
        takes the hidden states of the topmost RNN layer as input;
        unused, as exploratory experiments were unable to show any positive effect on the reconstruction objective. """
        with tf.variable_scope('sentence_attention'), tf.device('/gpu:0'):
            # Publication: see www.cs.cmu.edu/~diyiy/docs/naacl16.pdf
            # Publication code: github.com/ematvey/hierarchical-attention-networks/blob/master/HAN_model.py
            # Designate attention parameters
            projection_weights = tf.get_variable(name='projection_weights',
                                                 shape=[self.opt.enc_hidden_dims, self.opt.enc_attention_dims],
                                                 initializer=xi(uniform=False, dtype=self.float_type),
                                                 trainable=True)
            projection_biases = tf.get_variable(name='projection_biases', shape=[self.opt.enc_attention_dims],
                                                initializer=tf.zeros_initializer(dtype=self.float_type),
                                                trainable=True)
            context_vector = tf.get_variable(name='context_vector', shape=[self.opt.enc_attention_dims],
                                             initializer=xi(uniform=False, dtype=self.float_type), trainable=True)
            # Compute attention values
            memory_values = tf.reshape(self.rnn_outputs, shape=[-1, self.opt.enc_hidden_dims], name='memory_values')
            projected_memories = tf.nn.tanh(
                tf.nn.xw_plus_b(memory_values, projection_weights, projection_biases), name='projected_memories')
            projected_memories = tf.reshape(projected_memories, shape=[self.batch_length, self.batch_steps, -1])
            # Mask out positions corresponding to padding within the input
            score_mask = tf.sequence_mask(
                self.length_mask, maxlen=tf.reduce_max(self.length_mask), dtype=self.float_type)
            score_mask = tf.expand_dims(score_mask, -1)
            score_mask = tf.matmul(
                score_mask, tf.ones([self.batch_length, self.opt.enc_attention_dims, 1]), transpose_b=True)
            projected_memories = tf.where(
                tf.cast(score_mask, dtype=tf.bool), projected_memories, tf.zeros_like(projected_memories))
            # Calculate the importance of the individual encoder hidden states for the informativeness of the computed
            # sentence representation
            context_product = tf.reduce_sum(
                tf.multiply(projected_memories, context_vector, name='context_product'), axis=2, keep_dims=True)
            attention_weights = tf.nn.softmax(context_product, dim=1, name='importance_weight')
            # Weigh encoder hidden states according to the calculated importance weights
            weighted_memories = tf.multiply(projected_memories, attention_weights)
            # Sentence encodings are the importance-weighted sums of encoder hidden states / word representations
            sentence_encodings = tf.reduce_sum(weighted_memories, axis=1, name='sentence_encodings')
        return sentence_encodings

    def state_projection_subgraph(self):
        """ Defines parameters for the encoder state projection, in case of a state size mismatch between
        encoder and decoder; unused if encoder and decoder states are of identical size. """
        with tf.variable_scope('state_projection'), tf.device('/gpu:0'):
            state_projection_weights = tf.get_variable(name='state_projection_weights',
                                                       shape=[self.opt.enc_hidden_dims * self.opt.enc_num_layers,
                                                              self.opt.dec_hidden_dims * self.opt.dec_num_layers],
                                                       dtype=self.float_type,
                                                       initializer=xi(uniform=False, dtype=self.float_type),
                                                       trainable=True)
            state_projection_biases = tf.get_variable(name='state_projection_biases',
                                                      shape=[self.opt.dec_hidden_dims * self.opt.dec_num_layers],
                                                      dtype=self.float_type,
                                                      initializer=tf.zeros_initializer(self.float_type),
                                                      trainable=True)

            # Unpack the final state representation
            c_state, h_state = tf.split(self.final_state, 2, axis=0)
            if self.opt.enc_num_layers == self.opt.dec_num_layers:
                # Assign encoder states to decoder states on a by layer basis
                c_states = tf.split(c_state, self.opt.enc_num_layers, axis=1)
                h_states = tf.split(h_state, self.opt.enc_num_layers, axis=1)
            elif self.opt.enc_num_layers == 1 and self.opt.dec_num_layers > 1:
                # Initialize each decoder layer with the a copy of the final encoder layer
                c_states = [c_state] * self.opt.dec_num_layers
                h_states = [h_state] * self.opt.dec_num_layers
            else:
                # Project encoder's hidden and cell states so as to match the decoder state's dimensionality
                projected_state = tf.nn.xw_plus_b(self.final_state, state_projection_weights, state_projection_biases)
                c_state, h_state = tf.split(projected_state, 2, axis=0)
                c_states = tf.split(c_state, self.opt.dec_num_layers, axis=1)
                h_states = tf.split(h_state, self.opt.dec_num_layers, axis=1)

            # Assemble the appropriate LSTM cell state tuple used to initialize the decoder
            decoder_state = tuple([tf.contrib.rnn.LSTMStateTuple(c_states[layer_id], h_states[layer_id]) for
                                   layer_id in range(self.opt.dec_num_layers)])
        return c_state, h_state, decoder_state


class Decoder(object):
    """ Attentive RNN-LSTM decoder used to decode the low-dimensional, dense sentence encodings obtained from
    the corresponding encoder network within the context of a sequence auto-encoder model. """

    def __init__(self, vocab, opt, encoder, name):
        # Declare attributes
        self.vocab = vocab
        self.opt = opt
        self.encoder = encoder
        self.name = name
        self.float_type = tf.float32
        self.int_type = tf.int32

        with tf.variable_scope(self.name):
            # Keep track of sentence completions
            # i.e. the number of sequences within the currently decoded batch that have already terminated with <EOS>
            self.eos_tracker = None
            # Build computational sub-graphs
            # inputs | properties | embeddings | projection | attention | raw_rnn
            self.input_idx, self.static_keep_prob, self.rnn_keep_prob, self.sampling_bias = self.inputs_subgraph()
            self.batch_length, self.batch_steps, self.length_mask = self.properties_subgraph()
        # Embedding parameters are shared between encoder and decoder (thus no network-specific scoping)
        self.embedding_table, self.input_data, self.output_embedding_biases = self.embeddings_subgraph()
        with tf.variable_scope(self.name):
            self.projection_weights, self.projection_biases = self.projection_subgraph()
            if self.opt.attentive_decoding:
                self.memory_key_weights, self.attention_weights, self.dec_mixture_weights = \
                    self.global_attention_subgraph()
            if self.opt.attentive_encoding:
                self.enc_mixture_weights = self.encoder_mixture_subgraph()
            self.final_state, self.flat_rnn_outputs, self.projected_rnn_outputs, self.logits = self.lstm_rnn_subgraph()

    def inputs_subgraph(self):
        """ Creates the placeholders used to feed data to the model. """
        with tf.name_scope('inputs'), tf.device('/cpu:0'):
            # Read-in values from session input
            input_idx = tf.placeholder(shape=[None, None], dtype=self.int_type, name='input_idx')
            static_keep_prob = tf.placeholder(dtype=self.float_type, name='static_keep_prob')
            rnn_keep_prob = tf.placeholder(dtype=self.float_type, name='rnn_keep_prob')
            # Read in the probability with which the decoder is fed real prediction targets as input,
            # as opposed to its own predictions from the previous time-step (i.e. 'scheduled sampling')
            sampling_bias = tf.placeholder(dtype=self.float_type, name='sampling_bias')
        return input_idx, static_keep_prob, rnn_keep_prob, sampling_bias

    def properties_subgraph(self):
        """ Returns the properties of the input data relevant to the model's operation. """
        with tf.name_scope('properties'), tf.device('/cpu:0'):
            # Same as within the encoder
            batch_length = tf.shape(self.input_idx)[0]
            batch_steps = tf.shape(self.input_idx)[1]
            # Determine lengths of individual input sequences within the processed batch to mask RNN output and
            # exclude <EOS> and <PAD> tokens from the reconstruction loss computation
            length_mask = tf.count_nonzero(tf.not_equal(self.input_idx, self.vocab.pad_id),
                                           axis=1, keep_dims=False, dtype=self.int_type, name='length_mask')
        return batch_length, batch_steps, length_mask

    def embeddings_subgraph(self):
        """ Initializes the embedding table and output biases; embedding table is jointly used as the projection matrix
        for projecting the RNN-generated logits into the vocabulary space in the decoder. """
        with tf.variable_scope('embeddings', reuse=True), tf.device('/cpu:0'):
            embedding_table = tf.get_variable(name='embedding_table',
                                              shape=[self.vocab.n_words, self.opt.embedding_dims],
                                              dtype=self.float_type,
                                              initializer=xi(uniform=False, dtype=self.float_type),
                                              trainable=True)
            input_data = tf.nn.embedding_lookup(embedding_table, self.input_idx, name='embeddings')
            if self.opt.allow_dropout:
                input_data = tf.nn.dropout(input_data, self.static_keep_prob, name='dec_front_dropout')
            output_embedding_biases = tf.get_variable(name='output_embedding_biases', shape=[self.vocab.n_words],
                                                      dtype=self.float_type,
                                                      initializer=tf.zeros_initializer(dtype=self.float_type),
                                                      trainable=True)
        return embedding_table, input_data, output_embedding_biases

    def projection_subgraph(self):
        """ Defines the weight and bias parameters used to project RNN outputs into the embedding space following
        the completion of each full pass through the RNN. """
        with tf.variable_scope('decoder_projection'), tf.device('/gpu:0'):
            projection_weights = tf.get_variable(name='projection_weights',
                                                 shape=[self.opt.dec_hidden_dims, self.opt.embedding_dims],
                                                 dtype=self.float_type,
                                                 initializer=xi(uniform=False, dtype=self.float_type),
                                                 trainable=True)
            projection_biases = tf.get_variable(name='projection_biases', shape=[self.opt.embedding_dims],
                                                dtype=self.float_type,
                                                initializer=tf.zeros_initializer(dtype=self.float_type),
                                                trainable=True)
        return projection_weights, projection_biases

    def global_attention_subgraph(self):
        """ Defines the parameters for the global 'Luong' attention mechanism used during decoding;
        Publication: arxiv.org/pdf/1508.04025.pdf; With guidance from:
        github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/seq2seq/python/ops/attention_wrapper.py """
        with tf.variable_scope('global_decoder_attention'), tf.device('/gpu:0'):
            # Projects the encoder 'memories' (hidden states at each time step) to match decoder dimensions
            memory_key_weights = tf.get_variable(name='memory_key_weights',
                                                 shape=[self.opt.enc_hidden_dims, self.opt.dec_hidden_dims],
                                                 dtype=self.float_type,
                                                 initializer=xi(uniform=False, dtype=self.float_type),
                                                 trainable=True)
            # Used in computing the attention vector describing the alignment between input and output sequences
            attention_weights = tf.get_variable(name='attention_weights',
                                                shape=[self.opt.enc_hidden_dims + self.opt.dec_hidden_dims,
                                                       self.opt.dec_attention_dims],
                                                dtype=self.float_type,
                                                initializer=xi(uniform=False, dtype=self.float_type),
                                                trainable=True)
            # Used for combining the attention information with the decoder's input during the 'input feeding' step
            dec_mixture_weights = tf.get_variable(name='mixture_weights',
                                                  shape=[self.opt.dec_hidden_dims + self.opt.dec_attention_dims,
                                                         self.opt.embedding_dims],
                                                  dtype=self.float_type,
                                                  initializer=xi(uniform=False, dtype=self.float_type),
                                                  trainable=True)
            return memory_key_weights, attention_weights, dec_mixture_weights

    def encoder_mixture_subgraph(self):
        """ Defines the mixture weights used to condition decoder outputs on sentence encodings produced by the
        encoder-side attention mechanism by combining so obtained encodings with decoder inputs at every decoding
        time-step; unused due to the observed inefficacy of encoder-side attention for the decoding process. """
        with tf.variable_scope('encoder_mixture'), tf.device('/gpu:0'):
            enc_mixture_weights = tf.get_variable(name='encoder_mixture_weights',
                                                  shape=[self.opt.embedding_dims + self.opt.enc_attention_dims,
                                                         self.opt.embedding_dims],
                                                  dtype=self.float_type,
                                                  initializer=xi(uniform=False, dtype=self.float_type),
                                                  trainable=True)
        return enc_mixture_weights

    def lstm_rnn_subgraph(self):
        """ Defines the forward pass through the decoder LSTM-RNN. """
        with tf.variable_scope('lstm_rnn', reuse=None), tf.device('/gpu:0'):
            # Same functionality as within the encoder
            def _lstm_cell(model_opt):
                """ Defines a basic LSTM cell to which various wrappers can be applied. """
                base_cell = BasicLSTMCell(model_opt.dec_hidden_dims, forget_bias=2.5, state_is_tuple=True)
                if model_opt.allow_dropout:
                    base_cell = DropoutWrapper(base_cell, output_keep_prob=self.rnn_keep_prob)
                return base_cell

            if self.opt.dec_num_layers > 1:
                cell = MultiRNNCell([_lstm_cell(self.opt) for _ in range(self.opt.dec_num_layers)])
            else:
                cell = _lstm_cell(self.opt)

            # Obtain sequences decoded from the encoder's sentence representations
            # <PAD> slice output by the decoder after each generated batch sequence has ended in <EOS>
            pad_step_idx = tf.fill([self.batch_length], self.vocab.pad_id)
            pad_step_embeds = tf.nn.embedding_lookup(self.embedding_table, pad_step_idx, name='pad_step_embeds')

            # raw_rnn expects input to be served in form of a TensorArray
            inputs_ta = tf.TensorArray(size=self.batch_steps, dtype=self.float_type) \
                .unstack(tf.transpose(self.input_data, perm=[1, 0, 2]), name='rnn_input_array')

            # Initial decoder state set equal to the final state of the encoder
            initial_state = self.encoder.decoder_state

            # Initialize tensor for tracking sentence completion
            if self.eos_tracker is None:
                self.eos_tracker = tf.zeros([self.batch_length], dtype=self.int_type)

            # Define the raw_rnn loop which allows for greater control over the generated output, as compared
            # to dynamic_rnn()
            def loop_fn(time, cell_output, cell_state, loop_state):
                """ Defines the loop function utilized by the raw_rnn. """

                # Helper function for obtaining the output embeddings
                def _get_predictions():
                    """ Projects the likeliest raw_rnn output predictions into the embedding space. """
                    # Flatten RNN output to two dimensions
                    flat_step_outputs = tf.reshape(
                        cell_output, [-1, self.opt.dec_hidden_dims])
                    projected_step_outputs = tf.nn.xw_plus_b(
                        flat_step_outputs, self.projection_weights, self.projection_biases)
                    step_logits = tf.nn.xw_plus_b(projected_step_outputs, tf.transpose(self.embedding_table),
                                                  self.output_embedding_biases, name='logits')
                    # Isolate highest-probability predictions
                    predicted_scores = tf.nn.softmax(step_logits, -1)
                    idx_predictions = tf.cast(tf.argmax(predicted_scores, axis=-1), dtype=self.int_type)
                    # Embed predicted word indices
                    embedded_predictions = tf.nn.embedding_lookup(self.embedding_table, idx_predictions)
                    return idx_predictions, embedded_predictions

                def _attend():
                    """ Executes the decoding-with-attention mechanism utilizing global 'Luong' attention. """
                    # Project encoder hidden states, 'memories', to match the dimensionality of the decoder,
                    # i.e. target, hidden states
                    memory_values = self.encoder.rnn_outputs
                    flat_values = tf.reshape(memory_values, [-1, tf.shape(memory_values)[-1]])
                    flat_keys = tf.matmul(flat_values, self.memory_key_weights)
                    memory_keys = tf.reshape(
                        flat_keys, [self.encoder.batch_length, self.encoder.batch_steps, self.opt.dec_hidden_dims])

                    # Apply length to the memory keys so as to restrict attention to non-padded positions
                    score_mask = tf.sequence_mask(
                        self.encoder.length_mask, maxlen=tf.reduce_max(self.encoder.length_mask), dtype=self.float_type)
                    score_mask = tf.expand_dims(score_mask, -1)
                    score_mask = tf.matmul(score_mask,
                                           tf.ones([self.encoder.batch_length, self.opt.dec_hidden_dims, 1]),
                                           transpose_b=True)
                    memory_keys = tf.where(tf.cast(score_mask, dtype=tf.bool), memory_keys, tf.zeros_like(memory_keys))

                    # Obtain target query, i.e. the current decoder hidden state
                    target_hidden_state = cell_state[-1][-1]
                    target_query = tf.expand_dims(target_hidden_state, 1)

                    # Compute alignments globally, by attending to all encoder states at once
                    score = tf.matmul(target_query, memory_keys, transpose_b=True)
                    score = tf.squeeze(score, [1])
                    alignments = tf.nn.softmax(score)

                    # Compute the context vector by applying calculated alignments to encoder states
                    expanded_alignments = tf.expand_dims(alignments, 1)
                    context = tf.matmul(expanded_alignments, memory_values)
                    context = tf.squeeze(context, [1])

                    # Compute the attentional vector by combining encoder context with decoder query
                    attention = tf.tanh(tf.matmul(tf.concat([context, target_hidden_state], -1),
                                                  self.attention_weights))
                    return attention

                # Initialize the loop function
                emit_output = cell_output  # no output is emitted during initialization
                next_loop_state = None
                # Check if to terminate the loop;
                # length slack denotes how much longer the output sequence is allowed to be than the input
                elements_finished = tf.greater_equal(time, self.length_mask + self.opt.length_slack)
                # Once stopping conditions are met for all batch elements, terminate loop
                finished = tf.reduce_all(elements_finished)

                if cell_output is None:  # i.e. during initialization only
                    # Set initial values
                    self.eos_tracker *= 0
                    next_cell_state = initial_state
                    next_input = inputs_ta.read(0)

                # At time-step 1+
                else:
                    # Pass on the cell state
                    next_cell_state = cell_state
                    # Get predictions from previous time-step
                    predicted_idx, predicted_embeds = _get_predictions()
                    # Check if stopping conditions are met
                    # 1. Check if all decoded batch items contain an <EOS> prediction
                    self.eos_tracker += tf.cast(tf.equal(predicted_idx, self.vocab.eos_id), self.int_type)
                    # 2. Check if all decoded batch items are equal in length to corresponding encoder inputs
                    boundary_reached = tf.greater_equal(time, self.length_mask)
                    if not self.opt.is_train or not self.opt.use_reconstruction_objective:
                        # Extended stopping criterion during inference,
                        # as output length is allowed to exceed input length via the slack_length parameter
                        self.eos_tracker += tf.cast(tf.equal(predicted_idx, self.vocab.eos_id), self.int_type)
                        elements_finished = tf.logical_or(tf.greater(self.eos_tracker, 0),
                                                          tf.greater_equal(time,
                                                                           (self.length_mask + self.opt.length_slack)))
                        finished = tf.reduce_all(elements_finished)

                    # Scheduled sampling: If flip value is smaller than sampling probability, the output of the
                    # decoder at the current time-step is fed as input to the decoder at the subsequent time-step
                    flip = tf.random_uniform(shape=[], minval=0.0, maxval=1.0)
                    input_tensor = tf.cond(tf.logical_or(
                        tf.less(self.sampling_bias, flip), tf.reduce_all(boundary_reached)),
                        lambda: predicted_embeds, lambda: inputs_ta.read(time))
                    # If stopping conditions have been met, output a <PAD> slice, then terminate loop
                    next_input = tf.cond(finished, lambda: pad_step_embeds, lambda: input_tensor)

                    if self.opt.attentive_decoding:
                        # Input feeding: Combine attentive information with the input to the decoder at the
                        # subsequent time-step (either target tokens or predictions from the current time-step)
                        attentional_hidden_state = _attend()
                        next_input = tf.matmul(tf.concat([next_input, attentional_hidden_state], -1),
                                               self.dec_mixture_weights)

                if self.opt.attentive_encoding:
                    # Unused
                    next_input = tf.matmul(tf.concat([next_input, self.encoder.sentence_encodings], -1),
                                           self.enc_mixture_weights)

                return elements_finished, next_input, next_cell_state, emit_output, next_loop_state

            # Get RNN outputs
            rnn_outputs_tensor_array, final_state, _ = tf.nn.raw_rnn(cell, loop_fn)
            rnn_outputs = rnn_outputs_tensor_array.stack()
            rnn_outputs = tf.transpose(rnn_outputs, perm=[1, 0, 2])
            flat_rnn_outputs = tf.reshape(rnn_outputs, [-1, self.opt.enc_hidden_dims],
                                          name='reshaped_rnn_outputs')
            # Project RNN outputs into the embedding space, followed by the projection into vocabulary space
            projected_rnn_outputs = tf.nn.xw_plus_b(flat_rnn_outputs, self.projection_weights, self.projection_biases)
            logits = tf.nn.xw_plus_b(
                projected_rnn_outputs, tf.transpose(self.encoder.embedding_table), self.output_embedding_biases,
                name='logits')

        return final_state, flat_rnn_outputs, projected_rnn_outputs, logits


class SeqAE(object):
    """ Sequence autoencoder object linking together the encoder and decoder networks for training and inference. """

    def __init__(self, vocab, opt, name):
        # Declare attributes
        self.vocab = vocab
        self.opt = opt
        self.name = name
        self.float_type = tf.float32
        self.int_type = tf.int32

        with tf.variable_scope(self.name):
            # Initialize component networks; variable scoping enforces sharing of embedding parameters
            self.encoder = Encoder(vocab, opt, 'encoder')
            with tf.variable_scope(tf.get_variable_scope()):
                self.decoder = Decoder(vocab, opt, self.encoder, 'decoder')

            # Build computational sub-graph
            # inputs | prediction| nce_loss | cross_entropy_loss | optimization
            self.lr, self.label_idx = self.inputs_subgraph()
            self.predicted_scores, self.predicted_idx, self.predicted_idx_eos, self.last_prediction = \
                self.prediction_subgraph()
            sampled_loss_avg = self.sampled_loss_subgraph()
            cross_ent_loss_avg = self.cross_ent_loss_subgraph()
            if self.opt.use_candidate_sampling:
                self.loss_avg = sampled_loss_avg
            else:
                self.loss_avg = cross_ent_loss_avg
            self.validation_loss = cross_ent_loss_avg
            self.loss_regularized, self.grads, self.train_op, self.vars_and_grads = self.optimization_subgraph()
            # Summaries
            self.train_summaries, self.valid_summaries = self.summaries()

    def inputs_subgraph(self):
        """ Creates the placeholders used to feed data to the model. """
        with tf.name_scope('inputs'), tf.device('/cpu:0'):
            # Read in values from session input
            lr = tf.placeholder(dtype=self.float_type, name='learning_rate')
            label_idx = tf.placeholder(shape=[None, None], dtype=self.int_type, name='label_idx')
        return lr, label_idx

    def sampled_loss_subgraph(self):
        """ Calculates the sampled softmax candidate sampling loss used to train the SAE in a sequence-to-sequence
        manner. """
        with tf.name_scope('sampled_loss'), tf.device('/gpu:0'):
            # Flatten labels to two dimensions
            flat_labels = tf.reshape(self.label_idx, [-1, 1])
            # 'div' partition strategy assures consistency with use of softmax at inference time
            sampled_loss = tf.nn.sampled_softmax_loss(weights=self.encoder.embedding_table,
                                                      biases=self.decoder.output_embedding_biases,
                                                      labels=flat_labels,
                                                      inputs=self.decoder.projected_rnn_outputs,
                                                      num_sampled=self.opt.samples,
                                                      num_classes=self.vocab.n_words, num_true=1,
                                                      sampled_values=self.opt.sampled_values,
                                                      remove_accidental_hits=self.opt.remove_accidental_hits,
                                                      partition_strategy='div', name='sampled_loss')
            # RNN predictions corresponding to padded input positions are masked out,
            # so as to not affect the incurred loss
            loss_mask = tf.cast(tf.not_equal(tf.reshape(flat_labels, [-1]), self.vocab.pad_id), self.float_type)
            masked_loss = tf.multiply(sampled_loss, loss_mask, name='masked_sampled_loss')
            sampled_loss_avg = tf.reduce_mean(masked_loss, name='sampled_loss_avg')
        return sampled_loss_avg

    def cross_ent_loss_subgraph(self):
        """ Calculates the cross-entropy loss used to evaluate the model during validation steps. """
        with tf.name_scope('cross_entropy_loss'), tf.device('/gpu:0'):
            labels_flat = tf.reshape(self.label_idx, [-1])

            cross_ent_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_flat,
                                                                            logits=self.decoder.logits,
                                                                            name='cross_entropy_loss')
            # Masking is applied as for sampled softmax loss calculation
            loss_mask = tf.cast(tf.not_equal(labels_flat, self.vocab.pad_id), self.float_type)
            masked_loss = tf.multiply(cross_ent_loss, loss_mask, name='masked_cross_ent_loss')
            cross_ent_loss_avg = tf.reduce_mean(masked_loss, name='cross_ent_loss_avg')
        return cross_ent_loss_avg

    def optimization_subgraph(self):
        """ Defines the optimization procedure for the entire model. """
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
            # Define optimization OP
            optimizer = tf.train.AdamOptimizer(self.lr, name='optimizer')
            train_op = optimizer.apply_gradients(zip(grads, t_vars), global_step=global_step, name='train_op')
        return loss_regularized, grads, train_op, (t_vars, grads)

    def prediction_subgraph(self):
        """ Defines operations involved in generating sequences of word tokens on the basis of projected RNN output. """
        # Sub-graph contains some level of redundancy, as the decoder performs projection steps already
        with tf.variable_scope('prediction'), tf.device('/gpu:0'):
            # Isolate highest-probability RNN predictions for every time-step
            predicted_scores = tf.reshape(tf.nn.softmax(self.decoder.logits, -1, name='predictions'),
                                          [self.decoder.batch_length, -1, self.vocab.n_words])
            predicted_idx = tf.cast(tf.argmax(predicted_scores, axis=-1), dtype=self.int_type)
            # Predictions at last step only are used for beam-search prediction
            last_prediction = predicted_scores[:, -1, :]

            # Mask output past the <EOS> boundary, optionally also masking the <EOS> tag itself
            # First, identify <EOS> positions within each predicted sequence
            check_eos = tf.cast(tf.equal(predicted_idx, self.vocab.eos_id), dtype=self.int_type)
            # Flatten to one dimension, as required by tf.where()
            check_range = tf.range(tf.shape(check_eos)[-1], 0, -1, dtype=self.int_type)
            range_eos = tf.multiply(check_eos, check_range)
            eos_idx = tf.cast(tf.argmax(range_eos, axis=-1), dtype=self.int_type)

            # Assemble outputs masks
            max_sent_lens = tf.cast(tf.fill(tf.shape(eos_idx), tf.shape(predicted_idx)[1]), dtype=self.int_type)
            no_eos = tf.where(tf.equal(eos_idx, 0), max_sent_lens, eos_idx)
            # no_eos masks <EOS> tags
            shift_non_final = tf.cast(tf.not_equal(no_eos, max_sent_lens), dtype=self.int_type)
            final_eos = tf.add(no_eos, shift_non_final)
            # final_eos preserves sentence-final <EOS> tags
            no_eos_mask = tf.sequence_mask(no_eos, tf.shape(predicted_idx)[1], dtype=self.int_type)
            final_eos_mask = tf.sequence_mask(final_eos, tf.shape(predicted_idx)[1], dtype=self.int_type)
            # Apply masks to the RNN predictions
            pre_pad = tf.subtract(predicted_idx, self.vocab.pad_id)
            predicted_idx = tf.add(tf.multiply(pre_pad, no_eos_mask), self.vocab.pad_id)
            predicted_idx_eos = tf.add(tf.multiply(pre_pad, final_eos_mask), self.vocab.pad_id)
        return predicted_scores, predicted_idx, predicted_idx_eos, last_prediction

    def summaries(self):
        """ Defines and compiles the summaries tracking various model parameters and outputs. """
        with tf.name_scope('summaries'), tf.device('/cpu:0'):
            # Define summaries
            lrs = tf.summary.scalar(name='learning_rate', tensor=self.lr)
            tla = tf.summary.scalar(name='training_loss', tensor=self.loss_avg)
            vla = tf.summary.scalar(name='validation_loss', tensor=self.validation_loss)
            la_reg = tf.summary.scalar(name='loss_regularized', tensor=self.loss_regularized)
            train_list = [lrs, tla, la_reg]
            valid_list = [vla]
            train_summaries = tf.summary.merge(train_list, name='train_summaries')
            valid_summaries = tf.summary.merge(valid_list, name='valid_summaries')
        return train_summaries, valid_summaries
