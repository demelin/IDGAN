""" Session defined for the sequence autoencoder model. Includes the training process, a quick evaluation method that
uses sampled test items, and a more comprehensive evaluation on the entirety of the test set. """

import os
import time
import codecs
import logging
import numpy as np
import tensorflow as tf

# PyCharm imports
from shared.util import make_pickle, load_pickle, load_model
from autoencoder.src.codebase.preprocessing import prepare_data
from autoencoder.src.codebase.batching import DataServer
from autoencoder.src.codebase.model import SeqAE
from autoencoder.src.codebase.trainer import SeqAETrainer
from autoencoder.src.codebase.interface import SeqAEInterface
# from autoencoder.src.options.small_options import TrainOptionsSmall
from autoencoder.src.options.train_options import TrainOptions
from autoencoder.src.options.test_options import TestOptions

# Suppress TensorFlow info-level messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Initialize options (uncomment the preferred set of training options)
# train_opt = TrainOptionsSmall()
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
if train_opt.is_local:
    # Toy sets for quick identification of syntactic and semantic errors
    data_dir = os.path.join(train_opt.data_dir, 'toy')
    train_name, valid_name, test_name = 'toy_train', 'toy_valid', 'toy_test'
else:
    # Available Europarl corpora: low-ID, high-ID, and full (combining sentences from both ID-specific corpora)
    data_dir = os.path.join(train_opt.data_dir, 'europarl')
    if train_opt.train_id == 'source':
        train_name, valid_name, test_name = 'europarl_v7_high_train', 'europarl_v7_high_valid', 'europarl_v7_high_test'
    elif train_opt.train_id == 'target':
        train_name, valid_name, test_name = 'europarl_v7_low_train', 'europarl_v7_low_valid', 'europarl_v7_low_test'
    else:
        train_name, valid_name, test_name = 'europarl_v7_train', 'europarl_v7_valid', 'europarl_v7_test'

# Construct paths pointing to source files
vocab_source = os.path.join(data_dir, '{:s}.txt'.format(train_name))
train_source = os.path.join(data_dir, '{:s}.txt'.format(train_name))
valid_source = os.path.join(data_dir, '{:s}.txt'.format(valid_name))
test_source = os.path.join(data_dir, '{:s}.txt'.format(test_name))

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

if not train_opt.is_local:
    # Full vocabulary is used for ID-specific corpora, as well, to encourage greater output variance
    vocab_pickle = os.path.join(train_opt.root_dir, 'data/europarl/europarl_v7_train_vocab.pkl')
vocab = load_pickle(vocab_pickle)

# Write vocabulary contents to file for manual inspection
vocab_log_path = os.path.join(train_opt.out_dir, '{:s}_vocab_log.txt'.format(train_name))
with codecs.open(vocab_log_path, 'w', encoding='utf8') as in_file:
    for key, value in vocab.index_to_word.items():
        in_file.write('{:s}, {:s}\n'.format(str(key), value))
print('Vocab log written.')

# Define TensorFlow session configuration - active during all calls to session.py (uncomment desired settings)
config = tf.ConfigProto(allow_soft_placement=True)
# config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.5

# ======================================================================================================


def train_session():
    """ Executes a training session on the SAE model. """
    # Clear the default graph within which the model graph is constructed
    tf.reset_default_graph()
    # Load data
    train_data = load_pickle(train_pickle)
    valid_data = load_pickle(valid_pickle)
    # Construct the model graph
    sae_name = 'seq_ae' + '_{:s}'.format(train_opt.train_id)
    autoencoder = SeqAE(vocab, train_opt, sae_name)
    all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    all_init_op = tf.global_variables_initializer()

    # Extract pre-trained word embeddings from the IDGAN-internal LM and use them to initialize the SAE
    initialized_vars = [var for var in all_vars if 'embedding_table' not in var.name or 'optimization' in var.name]
    embeddings_sae_keys = [var.name for var in all_vars if var not in initialized_vars]
    embedding_lm_keys = list()
    # Handle scoping discrepancies between SAE graph and LM checkpoints, 
    # to make LM variables compatible with the instantiated graph
    for k in embeddings_sae_keys:
        k = k.replace(sae_name, 'cog_lm')
	k = k.split(':')[0]
        embedding_lm_keys.append(k)
    embeddings_dir = os.path.join(train_opt.root_dir, 'cognitive_language_model/src/checkpoints/')
    embeddings_epoch = 'best'
    # Map SAE embedding variables to LM embedding variables,
    # so that the former may be initialized with values extracted from the latter
    embeddings_dict = {embedding_lm_keys[i]: [v for v in tf.global_variables() if v.name == embeddings_sae_keys[i]][0]
                       for i in range(len(embedding_lm_keys))}
    # Declare saver object for initializing SAE's embedding table with embeddings learned by IDGAN's LM
    embeddings_saver = tf.train.Saver(embeddings_dict)
    # Declare OP for initializing other SAE parameters randomly
    no_embeds_init_op = tf.variables_initializer(initialized_vars)
    # Time training duration
    starting_time = time.time()

    with tf.Session(config=config) as train_sess:
        # Initialize variables
        if train_opt.is_local:
            # No pre-trained embeddings are loaded for experiments on the toy set
            train_sess.run(all_init_op)
        else:
            load_model(train_sess, embeddings_saver, embeddings_dir, embeddings_epoch)
            train_sess.run(no_embeds_init_op)

        # Initialize SAE interface and trainer, used for inference and training steps, respectively
        interface = SeqAEInterface(autoencoder, vocab, train_sess, test_opt)
        trainer = SeqAETrainer(vocab, train_opt, autoencoder, train_sess, train_data, valid_data, test_opt, interface)
        # Train model (either for a predefined number of epochs or until early stopping)
        print('+++TRAINING+++')
        trainer.train_model()

    # Report training duration
    elapsed = time.time() - starting_time
    logging.info('Training took {:d} hours, {:d} minutes, and {:.4f} seconds.'.format(
        int(elapsed // 3600), int((elapsed % 3600)) // 60, elapsed % 60))


# ======================================================================================================


def test_session(batch_size=1, target_epoch='best', beam_decoding=False):
    """ Executes a quick test session on the SAE model by sampling a small quantity of items from the test set
    and using the model to first compress them into a meaningful representation and subsequently reconstruct them. """
    # Clear the default graph
    tf.reset_default_graph()
    # Declare the batch size, if left unspecified in test options
    if train_opt.batch_size is None:
        train_opt.batch_size = batch_size
    # Load test data
    test_data = load_pickle(test_pickle)
    # Build model graph
    autoencoder = SeqAE(vocab, test_opt, 'seq_ae' + '_{:s}'.format(train_opt.train_id))
    # Declare saver object for restoring learned model parameters
    test_saver = tf.train.Saver()

    # Initiate testing session
    with tf.Session(config=config) as test_sess:
        # Load learned model parameters
        load_model(test_sess, test_saver, test_opt.save_dir, target_epoch)
        # Initialize model interface containing inference methods
        interface = SeqAEInterface(autoencoder, vocab, test_sess, test_opt)
        # Sample candidate sentences from the test set
        samples = np.random.choice(test_data, test_opt.num_samples).tolist()
        while max([len(sample.split()) for sample in samples]) > 100:
            samples = np.random.choice(test_data, test_opt.num_samples).tolist()
        # Initialize a loader object to pre-process the sampled sentences
        sample_loader = DataServer(samples, vocab, test_opt)
        samples_read = 0

        print('Sampled sentences:')
        for i, s in enumerate(samples):
            print('{:d}: {:s}'.format(i, s))
        print('-' * 10 + '\n')

        if not beam_decoding:
            # Perform greedy encoding-decoding
            print('Greedy decoding:')
            for i, sample_data in enumerate(sample_loader):
                _, enc_input, dec_input = sample_data
                generated = interface.greedy_generation(enc_input, dec_input)
                for j in range(test_opt.batch_size):
                    print('Encoded: {:s}\nDecoded: {:s}\n'.format(samples[samples_read + j], generated[j]))
                samples_read += test_opt.batch_size
        else:
            # Perform encoding-decoding with beam-search (limited use for reconstruction)
            assert (test_opt.batch_size == 1), 'Beam search not defined for batches with more than one element.'
            print('Beam search decoding:')
            for i, sample_data in enumerate(sample_loader):
                _, enc_input, _ = sample_data
                print('Encoded: {:s}'.format(samples[i]))
                interface.beam_generation(enc_input, print_results=True)

    print('-' * 10 + '\n')
    print('=' * 10 + '\n')
    print('Auto-encoder evaluation completed!')


# ======================================================================================================


def test_to_file(batch_size=1, target_epoch='best', beam_decoding=False):
    """ Executes a comprehensive test session on the entire test corpus;
    output is written to file for BLEU score calculation. """

    def _reconstruct_input(input_array):
        """ Reconstructs input sentences from numpy arrays; used to derive an accurate representation of the
        pre-processed, encoded sequences. """
        # Convert input array to list of lists of word indices
        input_idx = [np.squeeze(array).tolist() for array in np.split(input_array, input_array.shape[0], axis=0)]
        # Translate indices into corresponding word tokens; truncated after sentence-final <EOS>
        input_boundaries = [idx_list.index(vocab.eos_id) if vocab.eos_id in idx_list else len(idx_list)
                            for idx_list in input_idx]
        input_sentences = [[vocab.index_to_word[idx] for idx in input_idx[j][:input_boundaries[j]]]
                           for j in range(len(input_idx))]
        input_sentences = [' '.join(word_list) + '.' for word_list in input_sentences]
        return input_sentences

    if beam_decoding:
        assert (test_opt.batch_size == 1), \
            'Function is defined for a batch size of 1 due to the nature of beam search implementation.'

    # Clear the default graph
    tf.reset_default_graph()
    # Declare the batch size
    if train_opt.batch_size is None:
        train_opt.batch_size = batch_size
    # Load test data
    test_data = load_pickle(test_pickle)
    # Build model graph
    autoencoder = SeqAE(vocab, train_opt, 'seq_ae' + '_{:s}'.format(train_opt.train_id))
    # Declare saver object for restoring learned model parameters
    test_saver = tf.train.Saver()

    # Declare paths pointing to locations of output files (i.e. reference and translations sets for BLEU)
    encoded_path = os.path.join(test_opt.out_dir, '{:s}_encoded_test_corpus_beam_{:s}.txt'
                                .format(test_opt.train_id, str(beam_decoding)))
    decoded_path = os.path.join(test_opt.out_dir, '{:s}_decoded_test_corpus_beam_{:s}.txt'
                                .format(test_opt.train_id, str(beam_decoding)))

    # Initiate testing session
    with tf.Session(config=config) as test_sess:
        # Load learned model parameters
        load_model(test_sess, test_saver, test_opt.save_dir, target_epoch)
        # Initialize the model interface containing inference methods
        interface = SeqAEInterface(autoencoder, vocab, test_sess, test_opt)
        # Initialize a loader object to pre-process the test corpus
        test_loader = DataServer(test_data, vocab, test_opt)

        with open(encoded_path, 'w') as enc_file:
            with open(decoded_path, 'w') as dec_file:
                if not beam_decoding:
                    # Perform greedy encoding-decoding;
                    # write encoded and decoded sequences to respective files
                    print('Greedy decoding:')
                    for i, test_items in enumerate(test_loader):
                        labels, enc_input, dec_input = test_items
                        generated = interface.greedy_generation(enc_input, dec_input)
                        enc_file.write(_reconstruct_input(labels)[0] + '\n')
                        dec_file.write(generated[0] + '\n')
                        if i % 10 == 0 and i > 0:
                            print('{:d} sentences written to file.'.format(i * test_opt.batch_size))

                else:
                    # Perform encoding-decoding with beam-search;
                    # write encoded and decoded sequences to respective files
                    assert (test_opt.batch_size == 1), 'Beam search not defined for batches with more than one element.'
                    print('Beam search decoding:')
                    for i, test_items in enumerate(test_loader):
                        labels, enc_input, _ = test_items
                        generated = interface.beam_generation(enc_input, print_results=False)
                        # Write best beam result only
                        enc_file.write(_reconstruct_input(labels)[0] + '\n')
                        dec_file.write(generated[0][0] + '\n')
                        if i % 10 == 0 and i > 0:
                            print('{:d} sentences written to file.'.format(i))

    print('-' * 10 + '\n')
    print('=' * 10 + '\n')
    print('Auto-encoder documented evaluation completed!')
