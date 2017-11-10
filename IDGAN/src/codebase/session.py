""" Session defined for the IDGAN system. Includes the training process, a quick evaluation method that
uses sampled test items, and a more comprehensive evaluation on the entirety of the test set. Significant
cross-over with SAE session, as the evaluated system component is the IDGAN generator, instantiated as an SAE. """

import os
import time
import codecs
import logging
import numpy as np
import tensorflow as tf

from shared.util import make_pickle, load_pickle, load_model
from autoencoder.src.codebase.preprocessing import prepare_data
from autoencoder.src.codebase.batching import DataServer

from IDGAN.src.codebase.seq_gan import IDGAN
from IDGAN.src.codebase.gan_trainer import IDGANTrainer
from IDGAN.src.codebase.interface import IDGANInterface

# Options
from cognitive_language_model.src.options.test_options import TestOptions as LMTestOpt
from sentence_similarity_classifier.src.options.test_options import TestOptions as SSCTestOpt
from autoencoder.src.options.train_options import TrainOptions as AETrainOpt
from IDGAN.src.options.train_options import TrainOptions as GANTrainOpt
# from IDGAN.src.options.small_options import TrainOptionsSmall as GANTrainOptSmall
from IDGAN.src.options.test_options import TestOptions as GANTestOpt

# Suppress TensorFlow info-level messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Initialize options (uncomment the preferred set of IDGAN training options)
# Component options are initialized from component-specific option classes
lm_gan_opt = LMTestOpt()
ssc_gan_opt = SSCTestOpt()
ae_gan_opt = AETrainOpt()
train_opt = GANTrainOpt()
# train_opt = GANTrainOptSmall()
test_opt = GANTestOpt()
opts = [train_opt, ae_gan_opt, lm_gan_opt, ssc_gan_opt]

# Training is logged to allow for post-hoc examination of possible error sources
logging_path = os.path.join(train_opt.out_dir, 'console_log.txt')
logging.basicConfig(filename=logging_path, level=logging.INFO)
# The logs are also printed to console for immediate evaluation
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

# Specify training, validation, and test sets
if train_opt.use_toy:
    # Toy sets for quick identification of syntactic and semantic errors
    data_dir = os.path.join(train_opt.data_dir, 'toy')
    source_train_name, source_valid_name, source_test_name = 'toy_train', 'toy_valid', 'toy_test'
    target_train_name, target_valid_name, target_test_name = 'toy_train', 'toy_valid', 'toy_test'
else:
    # ID-variant Europarl corpora provide the input data for IDGAN training
    data_dir = os.path.join(train_opt.data_dir, 'europarl')
    source_train_name, source_valid_name, source_test_name = 'europarl_v7_high_train', 'europarl_v7_high_valid', \
                                                             'europarl_v7_high_test'
    target_train_name, target_valid_name, target_test_name = 'europarl_v7_low_train', 'europarl_v7_low_valid', \
                                                             'europarl_v7_low_test'

corpus_names = [source_train_name, source_valid_name, source_test_name, target_train_name, target_valid_name,
                target_test_name]
# Construct paths pointing to source files
source_paths = [os.path.join(data_dir, '{:s}.txt'.format(name)) for name in corpus_names]
# Construct paths pointing to pickles containing corpus items and vocabulary
pickle_paths = [os.path.join(data_dir, '{:s}_data.pkl'.format(name)) for name in corpus_names]
# Create pickles - this way, data pre-processing has only to be done once
if not os.path.isfile(pickle_paths[0]):
    for i in range(len(corpus_names)):
        make_pickle(train_opt, prepare_data, corpus_names[i], source_paths[i], pickle_paths[i], vocab_path=None,
                    is_train='train' in corpus_names[i], is_valid='valid' in corpus_names[i],
                    is_test='test' in corpus_names[i])

# Load the vocabulary pickle shared among all of IDGAN's components
if train_opt.use_toy:
    vocab_pickle = os.path.join(train_opt.root_dir, 'data/toy/toy_train_vocab.pkl')
else:
    vocab_pickle = os.path.join(train_opt.root_dir, 'data/europarl/europarl_v7_train_vocab.pkl')
vocab = load_pickle(vocab_pickle)

# Write vocabulary contents to file for manual inspection
vocab_log_path = os.path.join(train_opt.out_dir, '{:s}_vocab_log.txt'.format('shared'))
with codecs.open(vocab_log_path, 'w', encoding='utf8') as in_file:
    for key, value in vocab.index_to_word.items():
        in_file.write('{:s}, {:s}\n'.format(str(key), value))
print('Vocab log written.')

# Define TensorFlow session configuration - active during all calls to session.py (uncomment desired settings)
config = tf.ConfigProto(allow_soft_placement=True)


# config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.5

# ======================================================================================================


def get_ckpt_vars(ckpt_dir):
    """ Returns the names of variables stored within a TensorFlow checkpoint; used in the initialization
    of IDGAN's components with pre-trained parameters. """
    ckpt_vars = list()
    with tf.Session():
        for var_name, _ in tf.contrib.framework.list_variables(ckpt_dir):
            ckpt_vars.append(var_name)
    return ckpt_vars


# ======================================================================================================


def train_session():
    """ Executes a training session on the IDGAN system. """
    # Clear the default graph within which the model graph is constructed
    tf.reset_default_graph()

    # Load data
    source_train_data = load_pickle(pickle_paths[0])
    source_valid_data = load_pickle(pickle_paths[1])
    target_train_data = load_pickle(pickle_paths[3])
    target_valid_data = load_pickle(pickle_paths[4])

    # Construct the system graph (component-specific graphs are constructed within the IDGAN graph)
    seq_gan = IDGAN(opts, vocab, 'IDGAN')

    # Initialize IDGAN's component models with pre-trained parameters
    # Declare paths pointing to checkpoints containing desired parameter values
    component_ckpt_dir = os.path.join(train_opt.local_dir, 'checkpoints/components')
    lm_dir = os.path.join(component_ckpt_dir, 'lm')
    source_encoder_dir = os.path.join(component_ckpt_dir, 'source')
    if train_opt.cross_dec:
        source_decoder_dir = os.path.join(component_ckpt_dir, 'source_decoder')
    else:
        source_decoder_dir = os.path.join(component_ckpt_dir, 'source')  # NO crossing
    target_dir = os.path.join(component_ckpt_dir, 'target')
    chosen_epoch = 'best'

    # Isolate parameters to be loaded into the IDGAN's system
    # Excludes optimization variables as well as variables connected to the training of 'frozen' IDGAN components
    # Get lists of variables contained within component checkpoint files
    lm_vars_plus_optimization = get_ckpt_vars(lm_dir)
    source_encoder_vars_plus_optimization = get_ckpt_vars(source_encoder_dir)
    source_decoder_vars_plus_optimization = get_ckpt_vars(source_decoder_dir)
    target_vars_plus_optimization = get_ckpt_vars(target_dir)
    # Exclude training-specific variables from IDGAN initialization
    lm_vars = [var_name for var_name in lm_vars_plus_optimization if 'optimization' not in var_name]
    # To enable the 'crossed decoder' training condition, separate the encoder and decoder variables
    # of the SAE pre-trained on the high-ID corpus ('translator SAE' within IDGAN)
    source_encoder_vars = \
        [var_name for var_name in source_encoder_vars_plus_optimization if 'optimization' not in var_name]
    source_encoder_vars = [var_name for var_name in source_encoder_vars if 'encoder' in var_name]
    source_decoder_vars = \
        [var_name for var_name in source_decoder_vars_plus_optimization if 'optimization' not in var_name]
    source_decoder_vars = [var_name for var_name in source_decoder_vars if 'decoder' in var_name]
    target_vars = [var_name for var_name in target_vars_plus_optimization if 'optimization' not in var_name]
    # Obtain list of all variables which have to be initialized (either randomly or from pre-trained values)
    # within the IDGAN system
    all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    # Check for matches between variables found within the pre-trained checkpoints and IDGAN variables
    lm_parameters = [var for var in all_vars if var.name.split(':')[0] in lm_vars]
    source_encoder_parameters = [var for var in all_vars if var.name.split(':')[0] in source_encoder_vars]
    source_decoder_parameters = [var for var in all_vars if var.name.split(':')[0] in source_decoder_vars]
    target_parameters = [var for var in all_vars if var.name.split(':')[0] in target_vars]
    # Load matching variables from corresponding checkpoints
    loaded_parameters = lm_parameters + source_encoder_parameters + source_decoder_parameters + target_parameters
    # Rest is initialized randomly
    initialized_parameters = [var for var in all_vars if var not in loaded_parameters]

    # Initialize saver objects tasked with loading in the pre-trained parameters
    lm_saver = tf.train.Saver(lm_parameters)
    source_encoder_saver = tf.train.Saver(source_encoder_parameters)
    source_decoder_saver = tf.train.Saver(source_decoder_parameters)
    target_saver = tf.train.Saver(target_parameters)
    # Declare random initialization OP
    init_op = tf.variables_initializer(initialized_parameters)

    # Time training duration
    starting_time = time.time()

    with tf.Session(config=config) as train_sess:
        # Load pre-trained parameters into the IDGAN graph
        load_model(train_sess, lm_saver, lm_dir, chosen_epoch)
        load_model(train_sess, source_encoder_saver, source_encoder_dir, chosen_epoch)
        load_model(train_sess, source_decoder_saver, source_decoder_dir, chosen_epoch)
        load_model(train_sess, target_saver, target_dir, chosen_epoch)
        # Initialize the rest
        train_sess.run(init_op)

        # Initialize IDGAN interface and trainer, used for inference and training steps, respectively
        interface = IDGANInterface(seq_gan, vocab, train_sess, test_opt)
        trainer = IDGANTrainer(vocab, train_opt, seq_gan, train_sess, source_train_data, source_valid_data,
                               target_train_data, target_valid_data, test_opt, interface, verbose=True)
        # Train system (either for a predefined number of epochs or until early stopping)
        print('+++TRAINING+++')
        trainer.train_gan()

    # Report training duration
    elapsed = time.time() - starting_time
    logging.info('Training took {:d} hours, {:d} minutes, and {:.4f} seconds.'.format(
        int(elapsed // 3600), int((elapsed % 3600)) // 60, elapsed % 60))


# ======================================================================================================


def test_session(batch_size=1, target_epoch='best', beam_decoding=False):
    """ Executes a quick test session on the IDGAN system by sampling a small quantity of items from the test set
    and using the model to first compress them into a meaningful representation and subsequently reconstruct them;
    the evaluation process focuses exclusively on the translator SAE. """
    # Clear the default graph
    tf.reset_default_graph()
    # Declare the batch size, if left unspecified in test options
    if train_opt.batch_size is None:
        train_opt.batch_size = batch_size
    # Load data
    source_test_data = load_pickle(pickle_paths[2])
    # Build system graph
    seq_gan = IDGAN(opts, vocab, 'IDGAN')
    # Declare saver object for restoring learned IDGAN parameters
    test_saver = tf.train.Saver()

    # Initiate testing session
    with tf.Session(config=config) as test_sess:
        # Load learned model parameters
        load_model(test_sess, test_saver, test_opt.save_dir, target_epoch)
        # Initialize system interface containing inference methods
        interface = IDGANInterface(seq_gan, vocab, test_sess, test_opt)
        # Sample candidate sentences from the test set
        samples = np.random.choice(source_test_data, test_opt.num_samples).tolist()
        while max([len(sample.split()) for sample in samples]) > 10:
            samples = np.random.choice(source_test_data, test_opt.num_samples).tolist()
        # Initialize a loader object to pre-process the sampled sentences
        sample_loader = DataServer(samples, vocab, test_opt)
        samples_read = 0

        print('Sampled sentences:')
        for s_id, s in enumerate(samples):
            print('{:d}: {:s}'.format(s_id, s))
        print('-' * 10 + '\n')

        if not beam_decoding:
            # Perform greedy ID-reduction on the sampled sentences
            print('Greedy decoding:')
            for _, sample_data in enumerate(sample_loader):
                enc_labels, enc_inputs, dec_inputs = sample_data
                generated = interface.greedy_generation(enc_labels, enc_inputs, dec_inputs)
                for j in range(test_opt.batch_size):
                    print('Encoded: {:s}\nDecoded: {:s}\n'.format(samples[samples_read + j], generated[j]))
                samples_read += test_opt.batch_size
        else:
            # Perform greedy ID-reduction with beam-search on the sampled sentences
            assert (test_opt.batch_size == 1), 'Beam search not defined for batches with more than one element.'
            print('Beam search decoding:')
            for _, sample_data in enumerate(sample_loader):
                enc_labels, enc_input, _ = sample_data
                print('Encoded: {:s}'.format(samples[i]))
                interface.beam_generation(enc_labels, enc_input, print_results=True)

    print('-' * 10 + '\n')
    print('=' * 10 + '\n')
    print('IDGAN evaluation completed!')


# ======================================================================================================


def test_to_file(batch_size=1, target_epoch='best', beam_decoding=True):
    """ Executes a comprehensive test session on the entire test corpus;
    output is written to file for the calculation of the achieved corpus-wide ID reduction and
    the BLEU score between source sentences and their ID-reduced translations. """
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

    assert (test_opt.batch_size == 1), \
        'Function is defined for a batch size of 1 due to the nature of beam search implementation.'

    # Clear the default graph
    tf.reset_default_graph()
    # Declare the batch size
    if train_opt.batch_size is None:
        train_opt.batch_size = batch_size
    # Load test data from the high-ID corpus
    source_test_data = load_pickle(pickle_paths[2])
    # Build model graph
    seq_gan = IDGAN(opts, vocab, 'IDGAN')
    # Declare saver object for restoring learned model parameters
    test_saver = tf.train.Saver()

    # Declare paths pointing to locations of output files (i.e. reference and translations sets for BLEU)
    encoded_path = os.path.join(test_opt.out_dir, 'source_encoded_test_corpus_beam_{:s}.txt'.format(str(beam_decoding)))
    decoded_path = os.path.join(test_opt.out_dir, 'source_decoded_test_corpus_beam_{:s}.txt'.format(str(beam_decoding)))

    # Initiate testing session
    with tf.Session(config=config) as test_sess:
        # Load learned model parameters
        load_model(test_sess, test_saver, test_opt.save_dir, target_epoch)
        # Initialize the model interface containing inference methods
        interface = IDGANInterface(seq_gan, vocab, test_sess, test_opt)
        # Initialize a loader object to pre-process the test corpus
        test_loader = DataServer(source_test_data, vocab, test_opt)

        with open(encoded_path, 'w') as enc_file:
            with open(decoded_path, 'w') as dec_file:
                if not beam_decoding:
                    # Perform greedy ID-reduction on the sampled sentences
                    print('Greedy decoding:')
                    for s_id, test_items in enumerate(test_loader):
                        enc_labels, enc_inputs, dec_inputs = test_items
                        generated = interface.greedy_generation(enc_labels, enc_inputs, dec_inputs)
                        enc_file.write(_reconstruct_input(enc_labels)[0] + '\n')
                        dec_file.write(generated[0] + '\n')
                        if s_id % 10 == 0 and s_id > 0:
                            print('{:d} sentences written to file.'.format(s_id))

                else:
                    # Perform greedy ID-reduction with beam-search on the sampled sentences
                    assert (test_opt.batch_size == 1), 'Beam search not defined for batches with more than one element.'
                    print('Beam search decoding:')
                    for s_id, test_items in enumerate(test_loader):
                        enc_labels, enc_input, _ = test_items
                        generated = interface.beam_generation(enc_labels, enc_input, print_results=False)
                        # Write best beam result only
                        enc_file.write(_reconstruct_input(enc_labels)[0] + '\n')
                        dec_file.write(generated[0][0] + '\n')
                        if s_id % 10 == 0 and s_id > 0:
                            print('{:d} sentences written to file.'.format(s_id))

    print('-' * 10 + '\n')
    print('=' * 10 + '\n')
    print('IDGAN documented evaluation completed!')
