""" Interface object defined for the sentence similarity classifier. Limited to the definition of the inference step
taken by the model when predicting the similarity between two word sequences. """

import tensorflow as tf


class SentSimClassInterface(object):
    """ A small interface for the evaluation of sentence similarity classifier; extendable. """

    def __init__(self, model, vocab, session, opt):
        self.model = model
        self.vocab = vocab
        self.session = session
        self.opt = opt
        self.model_saver = tf.train.Saver()

    def infer_step(self, batch_input):
        """ Performs a single inference step. """
        # Variable values passed to the model graph
        feed_dict = {
            self.model.encoder_a.input_idx: batch_input[0][0],
            self.model.encoder_b.input_idx: batch_input[0][1],
            self.model.labels: batch_input[1],
            self.model.encoder_a.static_keep_prob: 1.0,
            self.model.encoder_a.rnn_keep_prob: 1.0,
            self.model.encoder_b.static_keep_prob: 1.0,
            self.model.encoder_b.rnn_keep_prob: 1.0
        }
        # OPs called within the model graph
        ops = [self.model.predictions, self.model.loss]
        # OP output is returned as numpy arrays
        predictions, prediction_error = self.session.run(ops, feed_dict=feed_dict)
        return predictions, prediction_error
