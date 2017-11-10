""" Session defined for the sentence similarity classifier model. Includes the training process and an evaluation
method for estimating the accuracy of the trained models. """

import os
import time
import codecs
import logging
import numpy as np
import tensorflow as tf

from shared.util import make_pickle, load_pickle, load_model
from sentence_similarity_classifier.src.codebase.preprocessing import prepare_data
from sentence_similarity_classifier.src.codebase.batching import DataServer
from sentence_similarity_classifier.src.codebase.util import extend_true_corpora, generate_fake_corpus
from sentence_similarity_classifier.src.codebase.model import SentSimClassifier
from sentence_similarity_classifier.src.codebase.trainer import SentSimClassTrainer
from sentence_similarity_classifier.src.codebase.interface import SentSimClassInterface
# from sentence_similarity_classifier.src.options.pretrain_options import PreTrainOptions
from sentence_similarity_classifier.src.options.train_options import TrainOptions
from sentence_similarity_classifier.src.options.test_options import TestOptions

# Suppress TensorFlow info-level messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Initialize options (uncomment the preferred set of training options)
# train_opt = PreTrainOptions()
train_opt = TrainOptions()
test_opt = TestOptions()

# Training is logged to allow for post-hoc examination of possible error sources
logging_path = os.path.join(train_opt.out_dir, 'console_log.txt')
logging.basicConfig(filename=logging_path, level=logging.INFO)
# The logs are also printed to console for immediate evaluation
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

# Specify training, validation, and test sets
data_dir = os.path.join(train_opt.data_dir, 'similarity')
# 'Components' directory contains the SemEval corpus as well as SICK corpus and its extensions
component_dir = os.path.join(data_dir, 'components')
if train_opt.pre_train:
    full_name, train_name, valid_name, test_name = 'joint_sim_all', 'joint_sim_train', 'joint_sim_valid', \
                                                   'joint_sim_test'
else:
    # Europarl similarity corpus comprises synthetic similarity pairs derived from the 90k Europarl corpus
    full_name, train_name, valid_name, test_name = 'europarl_sim_all', 'europarl_sim_train', 'europarl_sim_valid', \
                                                   'europarl_sim_test'

# Construct paths pointing to source files
vocab_source = os.path.join(data_dir, '{:s}.txt'.format(train_name))
train_source = os.path.join(data_dir, '{:s}.txt'.format(train_name))
valid_source = os.path.join(data_dir, '{:s}.txt'.format(valid_name))
test_source = os.path.join(data_dir, '{:s}.txt'.format(test_name))

if not os.path.exists(train_source):
    if 'europarl' in train_source:
        # Derive fake Europarl similarity samples from the split 90k Europarl corpus
        source_train_name, source_valid_name, source_test_name = 'europarl_v7_train', 'europarl_v7_valid', \
                                                                 'europarl_v7_test'
        generate_fake_corpus(train_opt, data_dir, source_train_name, train_name, shuffle_mixtures=True)
        generate_fake_corpus(train_opt, data_dir, source_valid_name, valid_name, shuffle_mixtures=True)
        generate_fake_corpus(train_opt, data_dir, source_test_name, test_name, shuffle_mixtures=True)
    else:
        # Pre-process and extend human-annotated similarity corpora
        filtered_name, extended_name, semeval_name = 'filtered_sick', 'extended_sick', 'sem_eval'
        extend_true_corpora(data_dir, component_dir, filtered_name, extended_name, semeval_name, full_name,
                            train_name, valid_name, test_name, [0.8, 0.15, 0.05])

# Construct paths pointing to pickles containing corpus items and vocabulary
vocab_pickle = os.path.join(data_dir, '{:s}_vocab.pkl'.format(train_name))
train_pickle = os.path.join(data_dir, '{:s}_data.pkl'.format(train_name))
valid_pickle = os.path.join(data_dir, '{:s}_data.pkl'.format(valid_name))
test_pickle = os.path.join(data_dir, '{:s}_data.pkl'.format(test_name))

# Create pickles - this way, data pre-processing has only to be done once
if not os.path.isfile(train_pickle):
    make_pickle(train_opt, prepare_data, train_name, train_source, train_pickle, vocab_path=vocab_pickle, is_train=True)
    make_pickle(train_opt, prepare_data, valid_name, valid_source, valid_pickle, is_valid=True)
    make_pickle(train_opt, prepare_data, test_name, test_source, test_pickle, is_test=True)

# Load in the model-internal vocabulary; during the fine-tuning/ domain adoptation step,
# load in the 90k Europark vocabulary used by all of IDGAN's component models
if not train_opt.pre_train:
    vocab_pickle = os.path.join(train_opt.root_dir, 'data/europarl/europarl_v7_train_vocab.pkl')
vocab = load_pickle(vocab_pickle)

# Write vocabulary contents to file for manual inspection
vocab_log_path = os.path.join(train_opt.out_dir, '{:s}_vocab_log.txt'.format(train_name))
with codecs.open(vocab_log_path, 'w', encoding='utf8') as vocab_file:
    for key, value in vocab.index_to_word.items():
        vocab_file.write('{:s}, {:s}\n'.format(str(key), value))
print('Vocab log written.')

# Define TensorFlow session configuration - active during all calls to session.py (uncomment desired settings)
config = tf.ConfigProto(allow_soft_placement=True)
# config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
# config.gpu_options.allow_growth = True

# ======================================================================================================


def train_session():
    """ Executes a training session on the SAE model. """
    # Clear the default graph within which the model graph is constructed
    tf.reset_default_graph()
    # Load data
    train_data = load_pickle(train_pickle)
    valid_data = load_pickle(valid_pickle)
    # Construct the model graph
    sent_sim_class = SentSimClassifier(vocab, train_opt, 'ssc')
    all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    # Declare OP for initializing of model variables
    init_op = tf.global_variables_initializer()

    # During domain adoptation, restore learned SSC parameters with the exception of the embeddings,
    # which are extracted from the pre-trained IDGAN-internal LM
    restored_vars = [var for var in all_vars if 'embedding_table' not in var.name]
    pre_train_saver = tf.train.Saver(restored_vars)
    embeddings_ssc_keys = [var.name for var in all_vars if var not in restored_vars and 'optimization' not in var.name]
    embedding_lm_keys = list()
    # Handle scoping discrepancies between SSC and LM checkpoints, to make LM variables compatible with the SSC graph
    for k in embeddings_ssc_keys:
        k = k.replace('ssc', 'cog_lm')
        k = k.replace('encoder/embeddings', 'embeddings')
        k = k_replace('Adam', 'optimizer')
	k = k.split(':')[0]
        embedding_lm_keys.append(k)
    embeddings_dir = os.path.join(train_opt.root_dir, 'cognitive_language_model/src/checkpoints/')
    embeddings_epoch = 'best'
    # Map SSC embedding variables to LM embedding variables,
    # so that the former may be initialized with values extracted from the latter
    embeddings_dict = {embedding_lm_keys[i]: [v for v in tf.global_variables() if v.name == embeddings_ssc_keys[i]][0]
                       for i in range(len(embedding_lm_keys))}
    # Declare saver object for initializing SAE's embedding table with embeddings learned by IDGAN's LM
    embeddings_saver = tf.train.Saver(embeddings_dict)
    # Time training duration
    starting_time = time.time()

    with tf.Session(config=config) as train_sess:
        if train_opt.pre_train:
            # Initialize variables
            train_sess.run(init_op)
        else:
            # Restore pre-trained model parameters for domain adaptation (sans embedding table)
            load_model(train_sess, pre_train_saver, os.path.join(train_opt.save_dir, 'pre_training'), 'best')
            # Restore embedding parameters from the specified LM checkpoint
            load_model(train_sess, embeddings_saver, embeddings_dir, embeddings_epoch)

        # Initialize SSC trainer
        trainer = SentSimClassTrainer(vocab, train_opt, sent_sim_class, train_sess, train_data, valid_data)
        # Train model (either for a predefined number of epochs or until early stopping)
        print('+++TRAINING+++')
        trainer.train_model()

    # Report training duration
    elapsed = time.time() - starting_time
    logging.info('Training took {:d} hours, {:d} minutes, and {:.4f} seconds.'.format(
        int(elapsed // 3600), int((elapsed % 3600)) // 60, elapsed % 60))

# ======================================================================================================


def test_session(target_epoch='best'):
    """ Evaluates the accuracy of the learned SSC model by using it to predict the similarity score of
    sentence pairs contained within the specified test set. """
    # Clear the default graph
    tf.reset_default_graph()
    # Load data
    test_data = load_pickle(test_pickle)
    # Build model graph
    sent_sim_class = SentSimClassifier(vocab, test_opt, 'ssc')
    # Declare saver
    test_saver = tf.train.Saver()
    save_dir = train_opt.save_dir

    # Initiate testing session
    with tf.Session(config=config) as test_sess:
        # Load learned model parameters
        load_model(test_sess, test_saver, save_dir, target_epoch)
        # Initialize model interface
        interface = SentSimClassInterface(sent_sim_class, vocab, test_sess, test_opt)
        # Initialize a loader object to pre-process and serve items drawn from the source corpus
        sample_loader = DataServer(test_data, vocab, test_opt)
        # Evaluate model's performance on a withheld test corpus to estimate its capacity for generalization beyond
        # seen data
        # Track prediction accuracy and the divergence of predicted similarity scores from target values
        total_error = 0.0
        total_differential = 0.0
        total_items = 0

        for i, test_batch in enumerate(sample_loader):
            # Obtain model predictions for the current test batch
            predictions, prediction_error = interface.infer_step(test_batch)
            total_error += np.sum(np.abs(prediction_error))
            try:
                for j in range(test_opt.batch_size):
                    cj = total_items + j
                    differential = np.abs(np.subtract(float(test_data[1][cj]), predictions[j][0]))
                    total_differential += differential
                    # Report model prediction and error
                    print('Sentence 1: {:s}\nSentence 2: {:s}\n'
                          'True score: {:.4f} | Model Prediction: {:.4f} | Differential: {:.4f}'
                          .format(test_data[0][cj][0], test_data[0][cj][1], float(test_data[1][cj]), predictions[j][0],
                                  differential))
                    print('-' * 10)
                total_items += test_opt.batch_size
            except IndexError:
                break
        # Report test corpus statistics
        print('Total model error: {:.4f} | Average model error: {:.4f} | Average prediction error: {:.4f}'.format(
            total_error, total_error / total_items, total_differential / total_items))

    print('-' * 10 + '\n')
    print('=' * 10 + '\n')
    print('Sentence similarity classifier evaluation completed!')
