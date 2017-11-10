""" Trainer object defined for the cognitive language model. Trains the model either for a predefined maximum
number of training epochs or until some early stopping condition is met. """

import os
import time
import logging
import numpy as np
import tensorflow as tf

from tensorflow.contrib.tensorboard.plugins import projector

from cognitive_language_model.src.codebase.batching import DataServer
from shared.util import save_model


class CogLMTrainer(object):
    """ Runs the model for the purposes of training and validation. Primary training engine. """

    def __init__(self, vocab, opt, model, session, train_data, valid_data):
        # Declare arguments
        self.vocab = vocab
        self.opt = opt
        self.model = model
        self.session = session
        self.train_data = train_data
        self.valid_data = valid_data

        # Initialize variables used to guide the training process
        self.current_lr = self.opt.learning_rate
        self.best_perplexity = float('inf')
        self.training_best = float('inf')

        self.stagnant_epochs = 0
        self.final_epoch = 0
        self.best_epoch = 0
        self.stopped_early = False
        self.train_global_step = 0
        self.valid_global_step = 0

        # Initialize writer objects needed for the construction of TensorBoard summaries
        self.train_writer = tf.summary.FileWriter(
            self.opt.log_dir + '/{:s}_train'.format(self.model.name), session.graph)
        self.valid_writer = tf.summary.FileWriter(
            self.opt.log_dir + '/{:s}_valid'.format(self.model.name), session.graph)
        # Initialize a saver object for the creation of model checkpoints
        self.model_saver = tf.train.Saver()

        # Extraneous summary for tracking a value calculated outside of the graph
        self.perplexity_summary = tf.Summary()
        self.start_time = None

        # Set up a saver object responsible for visualization of word embeddings
        self.embedding_saver = tf.train.Saver([self.model.embedding_table])
        # Metadata file is used to assign word tokens to corresponding word embeddings during the visualization step
        self.metadata_path = os.path.join(self.opt.log_dir + '/{:s}_train'.format(self.model.name), 'metadata.tsv')
        with open(self.metadata_path, 'w') as in_file:
            for idx in range(self.vocab.n_words):
                in_file.write('{:s}\n'.format(self.vocab.index_to_word[idx]))

    def train_model(self):
        """ Defines the primary training loop. """
        self.start_time = time.time()

        for e in range(self.opt.num_epochs):
            # For each epoch, track training losses, number of word tokens processed, and duration
            train_batch_losses = list()
            words_processed = 0
            epoch_start = time.time()
            # Re-initialize the training data loader at the start of each training epoch;
            # Epoch concludes once training data has been exhausted
            train_loader = DataServer(self.train_data, self.vocab, self.opt)
            for i, train_data in enumerate(train_loader):
                # At each training step, sample a training set mini-batch and feed it to the LM
                train_input, train_labels = train_data
                batch_length, batch_steps, train_batch_loss = self.train_step(i, train_input, train_labels)
                # Monitor step-wise losses
                train_batch_losses.append(train_batch_loss)
                words_processed += batch_length * batch_steps
                if i % self.opt.report_freq == 0 and i != 0:
                    logging.info('[TRAINING] Epoch: {:d} | Step: {:d} | Running loss avg: {:.4f}'
                                 .format(e, i, sum(train_batch_losses[-self.opt.report_freq:]) / self.opt.report_freq))
                self.train_global_step += 1
            # Calculate and display epoch statistics
            train_epoch_loss = sum(train_batch_losses) / len(train_batch_losses)
            epoch_wps = words_processed / (time.time() - epoch_start)
            logging.info('[TRAINING] Epoch {:d} concluded | Average epoch loss: {:.4f} | Average speed: {:.2f} wps'
                         .format(e, train_epoch_loss, epoch_wps))

            # Perform validation steps at the end of each training epoch after the specified warm-up period
            if e >= self.opt.start_early_stopping:
                valid_batch_losses = list()
                valid_epoch_words = 0
                epoch_log_prob = 0.0
                # Re-initialize the validation data loader at the start of each validation epoch
                valid_loader = DataServer(self.valid_data, self.vocab, self.opt)
                for j, valid_data in enumerate(valid_loader):
                    # Sample a validation set mini-batch and feed it to the LM
                    valid_input, valid_labels = valid_data
                    batch_length, batch_steps, valid_batch_loss, model_predictions = self.valid_step(
                        j, valid_input, valid_labels)
                    # Monitor step-wise losses
                    valid_batch_losses.append(valid_batch_loss)
                    valid_epoch_words += batch_length * batch_steps
                    epoch_log_prob += self.get_batch_prob(model_predictions, valid_labels)
                    if j % self.opt.report_freq == 0 and j != 0:
                        logging.info('[VALIDATION] Epoch: {:d} | Step: {:d} | Running loss avg: {:.4f}'
                                     .format(e, j,
                                             sum(valid_batch_losses[-self.opt.report_freq:]) / self.opt.report_freq))
                    self.valid_global_step += 1
                # Calculate and display epoch statistics
                valid_epoch_loss = sum(valid_batch_losses) / len(valid_batch_losses)
                # Rough estimate, as due to the nature of computation shuffling order affects the perplexity outcome
                valid_epoch_perplexity = np.power(2, -(epoch_log_prob / valid_epoch_words))
                # Add perplexity estimate to the validation summary
                self.perplexity_summary.value.add(
                    tag='validation_model_perplexity', simple_value=valid_epoch_perplexity)
                self.valid_writer.add_summary(self.perplexity_summary, global_step=self.valid_global_step)
                logging.info(
                    '[VALIDATION] Epoch {:d} concluded | Validation epoch loss: {:.4f} | Epoch perplexity: {:.4f}'
                    .format(e, valid_epoch_loss, valid_epoch_perplexity))

                # Keep track of validation losses to identify best-performing epoch
                if valid_epoch_perplexity < self.best_perplexity:
                    self.best_perplexity = valid_epoch_perplexity
                    self.training_best = train_epoch_loss
                    self.best_epoch = e
                    save_model(self.session, self.model, self.model_saver, self.opt.save_dir, 'best')
                    self.save_embeddings('best')
                    self.stagnant_epochs = 0
                else:
                    # If validation performance did not improve, increment number of 'stagnant' epochs
                    self.stagnant_epochs += 1

                # Optionally trigger early stopping after the specified number of validation epochs during which model
                # performance did not improve
                if self.opt.enable_early_stopping and self.stagnant_epochs >= self.opt.patience:
                    logging.info('Training terminated early after {:d} stagnant epochs | Final epoch: {:d}.'
                                 .format(self.stagnant_epochs, e))
                    self.final_epoch = e
                    self.stopped_early = True
                    break

                # Reduce the training rate by a set amount after the specified number of 'stagnant' validation epochs
                if self.stagnant_epochs % self.opt.annealing_step == 0 and self.stagnant_epochs >= \
                        self.opt.annealing_step:
                    old_lr = self.current_lr
                    self.current_lr *= self.opt.annealing_factor
                    logging.info('Learning rate reduced from {:.8f} to {:.8f} after {:d} stagnant epochs'
                                 .format(old_lr, self.current_lr, self.stagnant_epochs))

            # Optionally save model parameters periodically throughout the training process
            if self.opt.save_freq is not None:
                if e % self.opt.save_freq == 0 and e != 0:
                    save_model(self.session, self.model, self.model_saver, self.opt.save_dir, e)

        # Save the final set of learned parameters after the conclusion of the training loop
        save_model(self.session, self.model, self.model_saver, self.opt.save_dir, 'final')

        # Final report
        if self.stopped_early:
            logging.info('Training procedure terminated after {:d} epochs total.'
                         'Best validated epoch: {:d} | Best perplexity: {:.4f} | Best training loss: {:.4f}'
                         .format(self.final_epoch, self.best_epoch, self.best_perplexity, self.training_best))
        else:
            logging.info('Training procedure finished after {:d} epochs total.'
                         'Best validated epoch: {:d} | Best perplexity: {:.4f} | Best training loss: {:.4f}'
                         .format(self.opt.num_epochs, self.best_epoch, self.best_perplexity, self.training_best))

    def train_step(self, step, batch_input, batch_labels):
        """ Performs a single training step. """
        # Variable values passed to the model graph
        feed_dict = {
            self.model.input_idx: batch_input,
            self.model.labels: batch_labels,
            self.model.static_keep_prob: self.opt.static_keep_prob,
            self.model.rnn_keep_prob: self.opt.rnn_keep_prob,
            self.model.lr: self.current_lr,
        }
        # OPs called within the model graph
        ops = [self.model.batch_length, self.model.batch_steps, self.model.loss_avg, self.model.train_op]
        # Extend OP list with summary ops to be called periodically throughout the training
        ops_plus_summaries = ops + [self.model.train_summaries]
        if step % self.opt.summary_freq == 0 and step != 0:
            # Call summary OPs at specified increments
            batch_length, batch_steps, train_batch_loss, _, summary = self.session.run(
                ops_plus_summaries, feed_dict=feed_dict)
            # Write collected step-wise summaries
            self.train_writer.add_summary(summary=summary, global_step=self.train_global_step)
        else:
            # Execute training OPs
            batch_length, batch_steps, train_batch_loss, _ = self.session.run(ops, feed_dict=feed_dict)
        return batch_length, batch_steps, train_batch_loss

    def valid_step(self, step, batch_input, batch_labels):
        """ Performs a single validation step. """
        # Variable values passed to the model graph
        feed_dict = {
            self.model.input_idx: batch_input,
            self.model.labels: batch_labels,
            self.model.static_keep_prob: 1.0,
            self.model.rnn_keep_prob: 1.0,
        }
        # Summary acquisition and OP calls are performed in a manner identical to the training step
        ops = [self.model.batch_length, self.model.batch_steps, self.model.loss_avg, self.model.predictions]
        ops_plus_summaries = ops + [self.model.valid_summaries]
        # As the validation set is smaller than the training set, summaries are collected more frequently
        if step % (self.opt.summary_freq // 3) == 0 and step != 0:
            batch_length, batch_steps, valid_batch_loss, model_predictions, summary = self.session.run(
                ops_plus_summaries, feed_dict=feed_dict)
            self.valid_writer.add_summary(summary=summary, global_step=self.valid_global_step)
        else:
            batch_length, batch_steps, valid_batch_loss, model_predictions, = self.session.run(
                ops, feed_dict=feed_dict)
        return batch_length, batch_steps, valid_batch_loss, model_predictions

    @staticmethod
    def get_batch_prob(model_predictions, batch_labels):
        """ Calculates the log probability of a validation batch for the estimation of model perplexity. """
        # See web.mit.edu/6.863/www/fall2012/lectures/lecture2&3-notes12.pdf for notes on perplexity
        flat_predictions = np.reshape(model_predictions, [-1])
        flat_labels = np.arange(
            0, model_predictions.shape[0]) * model_predictions.shape[1] + np.reshape(batch_labels, [-1])
        # Isolate probability predictions corresponding to target labels
        step_probs = np.take(flat_predictions, flat_labels)
        batch_log_prob = np.sum(np.log2(step_probs))
        return batch_log_prob

    def save_embeddings(self, epoch):
        """ Saves the embedding table learned by the model for post-hoc visualization of learned word embeddings. """
        # Declare the target path for the embeddings checkpoint file
        train_dir = self.opt.log_dir + '/{:s}_train'.format(self.model.name)
        embeddings_name = '{:s}_{:s}_embeddings.ckpt'.format(str(epoch), self.model.name)
        embeddings_path = self.embedding_saver.save(self.session, os.path.join(train_dir, embeddings_name))
        # Set up the TensorBoard projector
        projector_config = projector.ProjectorConfig()
        embedding_projection = projector_config.embeddings.add()
        embedding_projection.tensor_name = self.model.embedding_table.name
        # Add metadata and save the projector configuration file
        embedding_projection.metadata_path = self.metadata_path
        projector.visualize_embeddings(tf.summary.FileWriter(train_dir), projector_config)
        logging.info('Model embeddings have been saved in file {:s}'.format(embeddings_path))
