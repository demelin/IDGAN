""" Session defined for the cognitive language model. Includes the training process, scripts for the annotation of
corpora with the model's output (perplexity and ID-relevant measures), the entirety of the corpus construction pipeline
utilized in the preparation for IDGAN's training, and an evaluation method designed to estimate the model's performance
on sentences sampled from the test set. """

import os
import time
import codecs
import logging
import numpy as np
import tensorflow as tf

from shared.util import make_pickle, load_pickle, load_model, train_valid_test_split, shrink_domain, id_split
from cognitive_language_model.src.codebase.preprocessing import prepare_data
from cognitive_language_model.src.codebase.model import CogLM
from cognitive_language_model.src.codebase.trainer import CogLMTrainer
from cognitive_language_model.src.codebase.interface import CogLMInterface
# from cognitive_language_model.src.options.small_options import TrainOptionsSmall
from cognitive_language_model.src.options.train_options import TrainOptions
from cognitive_language_model.src.options.test_options import TestOptions

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
# Toy sets for quick identification of syntactic and semantic errors
if train_opt.is_local:
    data_dir = os.path.join(train_opt.data_dir, 'toy')
    full_name, train_name, valid_name, test_name = 'toy_full', 'toy_train', 'toy_valid', 'toy_test'
    scored_name, shrunk_name, annotated_name, low_name, high_name = 'toy_scored', 'toy_shrunk', 'toy_annotated', \
                                                                    'toy_low', 'toy_high'
else:
    # LM is trained on the 100k Europarl corpus first, then re-trained on the truncated 90k corpus
    data_dir = os.path.join(train_opt.data_dir, 'europarl')
    full_name, train_name, valid_name, test_name = 'europarl_v7_all', 'europarl_v7_train', 'europarl_v7_valid', \
                                                   'europarl_v7_test'
    scored_name, shrunk_name, annotated_name, low_name, high_name = 'europarl_v7_scored', 'europarl_v7_shrunk', \
                                                                    'europarl_v7_annotated', 'europarl_v7_low', \
                                                                    'europarl_v7_high'
# Construct paths pointing to source files
full_source = os.path.join(data_dir, '{:s}.txt'.format(full_name))
vocab_source = os.path.join(data_dir, '{:s}.txt'.format(train_name))
train_source = os.path.join(data_dir, '{:s}.txt'.format(train_name))
valid_source = os.path.join(data_dir, '{:s}.txt'.format(valid_name))
test_source = os.path.join(data_dir, '{:s}.txt'.format(test_name))

# Construct paths pointing to pickles containing corpus items and vocabulary
full_pickle = os.path.join(data_dir, '{:s}_data.pkl'.format(full_name))
restricted_pickle = os.path.join(data_dir, '{:s}_data.pkl'.format(full_name))
vocab_pickle = os.path.join(data_dir, '{:s}_vocab.pkl'.format(train_name))
train_pickle = os.path.join(data_dir, '{:s}_data.pkl'.format(train_name))
valid_pickle = os.path.join(data_dir, '{:s}_data.pkl'.format(valid_name))
test_pickle = os.path.join(data_dir, '{:s}_data.pkl'.format(test_name))

# Create pickles - this way, data pre-processing has only to be done once
if not os.path.isfile(train_pickle):
    make_pickle(train_opt, prepare_data, full_name, full_source, full_pickle)
    make_pickle(train_opt, prepare_data, train_name, train_source, train_pickle, vocab_path=vocab_pickle, is_train=True)
    make_pickle(train_opt, prepare_data, valid_name, valid_source, valid_pickle, is_valid=True)
    make_pickle(train_opt, prepare_data, test_name, test_source, test_pickle, is_test=True)

# Write vocabulary contents to file for manual inspection
vocab = load_pickle(vocab_pickle)
vocab_log_path = os.path.join(train_opt.out_dir, '{:s}_vocab_log.txt'.format(train_name))
with codecs.open(vocab_log_path, 'w', encoding='utf8') as vocab_file:
    for key, value in vocab.index_to_word.items():
        vocab_file.write('{:s}, {:s}\n'.format(str(key), value))
print('Vocab log written.')

# Define TensorFlow session configuration - active during all calls to session.py (uncomment desired settings)
config = tf.ConfigProto(allow_soft_placement=True)
# config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.5

# Designate a general 'notes' file for logging of miscellaneous information
lm_notes = os.path.join(train_opt.out_dir, '{:s}_notes.txt'.format(train_name))


# ======================================================================================================


def train_session():
    """ Executes a training session on the LM. """
    # Clear the default graph within which the model graph is constructed
    tf.reset_default_graph()
    # Load data
    train_data = load_pickle(train_pickle)
    valid_data = load_pickle(valid_pickle)
    # Construct the model graph and
    cog_lm = CogLM(vocab, train_opt, 'cog_lm')
    # Declare OP for initializing of model variables
    init_op = tf.global_variables_initializer()
    # Time training duration
    starting_time = time.time()

    with tf.Session(config=config) as train_sess:
        # Initialize variables
        train_sess.run(init_op)
        # Initialize LM trainer
        trainer = CogLMTrainer(vocab, train_opt, cog_lm, train_sess, train_data, valid_data)
        # Train model (either for a predefined number of epochs or until early stopping)
        print('+++TRAINING+++')
        trainer.train_model()

    # Report training duration
    elapsed = time.time() - starting_time
    logging.info('Training took {:d} hours, {:d} minutes, and {:.4f} seconds.'.format(
        int(elapsed // 3600), int((elapsed % 3600)) // 60, elapsed % 60))


# ======================================================================================================


def score_corpus():
    """ Executes a session during which the source corpus is annotated with sentence-wise model perplexity scores. """
    # Clear the default graph
    tf.reset_default_graph()
    # Declare path leading to corpus to be scored
    scored_path = os.path.join(data_dir, '{:s}.txt'.format(scored_name))
    ppx_scores = list()
    # Load data
    full_data = load_pickle(full_pickle)
    # Build model graph
    cog_lm = CogLM(vocab, test_opt, 'cog_lm')
    # Declare saver object for restoring learned model parameters
    sort_saver = tf.train.Saver()
    # Time the duration of the scoring process
    starting_time = time.time()

    with tf.Session(config=config) as sort_sess:
        # Load learned model parameters
        load_model(sort_sess, sort_saver, test_opt.save_dir, 'best')
        # Initialize LM interface
        interface = CogLMInterface(cog_lm, vocab, sort_sess, test_opt)
        # Run the scoring loop
        pos = 0
        with codecs.open(scored_path, 'w') as in_file:
            while pos < len(full_data) - 1:
                # Fill a single batch of sentences to be scored
                try:
                    batch = full_data[pos: pos + test_opt.batch_size]
                    pos += test_opt.batch_size
                except IndexError:
                    batch = full_data[pos: len(full_data) - 1]
                    pos = len(full_data) - 1
                # Get sentence-wise model perplexity scores
                batch_ppx = interface.get_sequence_perplexity(batch)
                # Write the scored sentences to file
                for i in range(len(batch)):
                    sentence_ppx = batch_ppx[i, :].tolist()[0]
                    scored_sent = '{:s}\t{:.4f}\n'.format(batch[i], sentence_ppx)
                    in_file.write(scored_sent)
                    # Keep track of corpus-wide statistics
                    ppx_scores.append(sentence_ppx)

    # Archive corpus statistics
    with open(lm_notes, 'w') as notes_file:
        notes_file.write('------------ Scored Corpus Statistics -------------\n')
        notes_file.write('Metric\tMean\tMedian\n')
        notes_file.write('Senetence Perplexity\t{:.4f}\t{:.4f}\n'.
                         format(np.mean(ppx_scores), np.median(ppx_scores)))

    # Report scoring duration
    elapsed = time.time() - starting_time
    print('Scoring took {:d} hours, {:d} minutes, and {:.4f} seconds.'.format(
        int(elapsed // 3600), int((elapsed % 3600)) // 60, elapsed % 60))


def shrink_scored_corpus():
    """ Shrinks the scored corpus with the aim of restrict the corpus domain,
    by removing outliers as determined by the model perplexity scores. """
    scored_path = os.path.join(data_dir, '{:s}.txt'.format(scored_name))
    print('Shrink corpus domain ... ')
    reduced_path = os.path.join(data_dir, '{:s}.txt'.format(shrunk_name))
    shrink_domain(scored_path, reduced_path, keep_fraction=0.9)
    print('Done!')


def shrunk_split():
    """ Splits the shrunk corpus into the training, validation, and test sets. """
    shrunk_source_path = os.path.join(data_dir, '{:s}.txt'.format(shrunk_name))
    shrunk_train_path = os.path.join(data_dir, '{:s}.txt'.format(shrunk_name + '_train'))
    shrunk_valid_path = os.path.join(data_dir, '{:s}.txt'.format(shrunk_name + '_valid'))
    shrunk_test_path = os.path.join(data_dir, '{:s}.txt'.format(shrunk_name + '_test'))
    split_fractions = [0.8, 0.15, 0.05]
    train_valid_test_split(shrunk_source_path, shrunk_train_path, shrunk_valid_path, shrunk_test_path, split_fractions)
    print('Done!')

# ======================================================================================================


def annotate_corpus():
    """ Executes a session during which the shrunk source corpus is annotated with ID-relevant measures. """
    # Clear the default graph
    tf.reset_default_graph()
    # Declare path leading to corpus to be annotated
    annotated_path = os.path.join(data_dir, '{:s}_annotated.txt'.format(train_name.split('_')[0]))
    # Values assigned per sentence are tracked for subsequent computation of corpus-wide statistics
    corpus_stats = {
        'Total_surprisal': list(),
        'Per_word_surprisal': list(),
        'Normalized_surprisal': list(),
        'Total_UID_divergence': list(),
        'Per_word_UID_divergence': list(),
        'Normalized_UID_divergence': list()
    }

    # Load data
    full_data = load_pickle(full_pickle)
    # Build model graph
    cog_lm = CogLM(vocab, test_opt, 'cog_lm')
    # Declare saver object for restoring learned model parameters
    annotate_saver = tf.train.Saver()
    # Time annotation duration
    starting_time = time.time()

    with tf.Session(config=config) as annotate_sess:
        # Load learned model parameters
        load_model(annotate_sess, annotate_saver, test_opt.save_dir, 'best')
        # Initialize LM interface
        interface = CogLMInterface(cog_lm, vocab, annotate_sess, test_opt)
        # Run the annotation loop
        pos = 0
        with codecs.open(annotated_path, 'w') as in_file:
            while pos < len(full_data) - 1:
                # Fill a single batch of sentences to be annotated
                try:
                    batch = full_data[pos: pos + test_opt.batch_size]
                    pos += test_opt.batch_size
                except IndexError:
                    batch = full_data[pos: len(full_data) - 1]
                    pos = len(full_data) - 1
                # Obtain ID-values via LM's interface
                total_surp, per_word_surp, norm_surp, total_uiddiv, per_word_uiddiv, norm_uiddiv = \
                    interface.get_surprisal(batch)
                # Write annotated sentences to file
                for i in range(len(batch)):
                    # For per-word annotations, exclude values associated with <EOS> and <PAD> tags
                    cut_off = len(batch[i].split())
                    item_ts = total_surp[i, :].tolist()[0]
                    # Surprisal
                    item_pws_floats = per_word_surp[i, :].tolist()[:cut_off]
                    item_pws = ';'.join(['{:.4f}'.format(pws) for pws in item_pws_floats])
                    item_ns = norm_surp[i, :].tolist()[0]
                    item_tu = total_uiddiv[i, :].tolist()[0]
                    # UID divergence
                    item_pwu_floats = per_word_uiddiv[i, :].tolist()[:cut_off]
                    item_pwu = ';'.join(['{:.4f}'.format(pwu) for pwu in item_pwu_floats])
                    item_nu = norm_uiddiv[i, :].tolist()[0]
                    # Construct annotated sample
                    scored_sent = '{:s}\t{:.4f}\t{:s}\t{:.4f}\t{:.4f}\t{:s}\t{:4f}\n'. \
                        format(batch[i], item_ts, item_pws, item_ns, item_tu, item_pwu, item_nu)
                    # Write to file
                    in_file.write(scored_sent)
                    # Update corpus stats dictionary
                    corpus_stats['Total_surprisal'].append(item_ts)
                    corpus_stats['Per_word_surprisal'].extend(item_pws_floats)
                    corpus_stats['Normalized_surprisal'].append(item_ns)
                    corpus_stats['Total_UID_divergence'].append(item_tu)
                    corpus_stats['Per_word_UID_divergence'].extend(item_pwu_floats)
                    corpus_stats['Normalized_UID_divergence'].append(item_nu)

    # Archive corpus statistics
    with open(lm_notes, 'a') as notes_file:
        notes_file.write('\n')
        notes_file.write('------------ Annotated Corpus Statistics -------------\n')
        notes_file.write('Metric\tMean\tMedian\tLowest\tHighest\n')
        for k, v in corpus_stats.items():
            notes_file.write('{:s}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n'
                             .format(k, np.mean(v), np.median(v), np.min(v), np.max(v)))

    # Report scoring duration
    elapsed = time.time() - starting_time
    print('Annotation took {:d} hours, {:d} minutes, and {:.4f} seconds.'.format(
        int(elapsed // 3600), int((elapsed % 3600)) // 60, elapsed % 60))


def split_annotated_corpus():
    """ Splits the ID-annotated corpus into low-ID and high-ID halves. """
    annotated_path = os.path.join(data_dir, '{:s}_annotated.txt'.format(train_name.split('_')[0]))
    print('Splitting source corpus in ID-variant sub-corpora ...')
    low_path = os.path.join(data_dir, low_name)
    high_path = os.path.join(data_dir, high_name)
    id_split(annotated_path, low_path, high_path)
    print('Done!')


def low_split():
    """ Splits the low-ID corpus into the training, validation, and test sets. """
    low_source_path = os.path.join(data_dir, '{:s}.txt'.format(low_name))
    low_train_path = os.path.join(data_dir, '{:s}.txt'.format(low_name + '_train'))
    low_valid_path = os.path.join(data_dir, '{:s}.txt'.format(low_name + '_valid'))
    low_test_path = os.path.join(data_dir, '{:s}.txt'.format(low_name + '_test'))
    split_fractions = [0.8, 0.15, 0.05]
    train_valid_test_split(low_source_path, low_train_path, low_valid_path, low_test_path, split_fractions)


def high_split():
    """ Splits the high-ID corpus into the training, validation, and test sets. """
    high_source_path = os.path.join(data_dir, '{:s}.txt'.format(high_name))
    high_train_path = os.path.join(data_dir, '{:s}.txt'.format(high_name + '_train'))
    high_valid_path = os.path.join(data_dir, '{:s}.txt'.format(high_name + '_valid'))
    high_test_path = os.path.join(data_dir, '{:s}.txt'.format(high_name + '_test'))
    split_fractions = [0.8, 0.15, 0.05]
    train_valid_test_split(high_source_path, high_train_path, high_valid_path, high_test_path, split_fractions)

# ======================================================================================================


def test_session(target_epoch='best', calculate_er=False, generate=True, gen_cycles=1):
    """ Executes a quick test session on the language model by sampling a small quantity of items from the test set
    and scoring them along various metrics. """
    # Tests are defined for a batch size of 1
    assert (test_opt.batch_size == 1), 'Model tests require batch size to equal 1.'
    # Clear the default graph
    tf.reset_default_graph()
    # Load data
    test_data = load_pickle(test_pickle)
    # Build model graph
    cog_lm = CogLM(vocab, test_opt, 'cog_lm')
    # Declare saver object for restoring learned model parameters
    test_saver = tf.train.Saver()

    with tf.Session(config=config) as test_sess:
        # Load learned model parameters
        load_model(test_sess, test_saver, test_opt.save_dir, target_epoch)
        # Initialize LM interface
        interface = CogLMInterface(cog_lm, vocab, test_sess, test_opt)
        # Sample sentences to be forwarded to the model
        samples = np.random.choice(test_data, test_opt.num_samples).tolist()

        print('Sampled sentences:')
        for i, s in enumerate(samples):
            print('{:d}: {:s}'.format(i, s))
        print('-' * 10 + '\n')

        # Get sentence probabilities
        print('Probabilities:')
        for i, s in enumerate(samples):
            total, prob_array, _ = interface.get_probability(s)
            # Mask <EOS> and <PAD> tag values
            cut_off = len(s.split())
            print('{:d}: {:s} | Total probability: {:.10f}'.format(i, s, total[0][0]))
            print('Per-word probabilities:')
            print('\t'.join(s.split()))
            print('\t'.join(['{:.4}'.format(score) for score in prob_array[0][:cut_off]]))
        print('-' * 10 + '\n')

        # Get sentence log-probabilities
        print('Log-probabilities:')
        for i, s in enumerate(samples):
            total, prob_array, _ = interface.get_log_probability(s)
            cut_off = len(s.split())
            print('{:d}: {:s} | Total log-probability: {:.4f}'.format(i, s, total[0][0]))
            print('Per-word log-probabilities:')
            print('\t'.join(s.split()))
            print('\t'.join(['{:.4}'.format(score) for score in prob_array[0]][:cut_off]))
        print('-' * 10 + '\n')

        # Get surprisal
        print('Surprisal and UID:')
        for i, s in enumerate(samples):
            total_s, s_array, norm_s, total_ud, ud_array, norm_ud = interface.get_surprisal(s)
            cut_off = len(s.split())
            tabbed_sent = '\t'.join(s.split())
            print('{:d}: {:s} | Total surprisal: {: .4f} | Normalized surprisal: {: .4f}'
                  .format(i, s, total_s[0][0], norm_s[0][0]))
            print('Per-word surprisal:')
            print(tabbed_sent)
            print('\t'.join(['{: .4}'.format(score) for score in s_array[0][:cut_off]]))
            print('{:d}: {:s} | Absolute UID: {:.4f} | Normalized UID: {: .4f}'
                  .format(i, s, total_ud[0][0], norm_ud[0][0]))
            print('Per-word UID:')
            print(tabbed_sent)
            print('\t'.join(['{: .4}'.format(score) for score in ud_array[0][:cut_off]]))
        print('-' * 10 + '\n')

        # Get approximate entropy reduction (computationally expensive!)
        if calculate_er:
            print('Approximate entropy reduction:')
            for i, s in enumerate(samples):
                total, array, norm = interface.get_entropy_reduction(samples)
                cut_off = len(s.split())
                print('{:d}: {:s} | Total ER: {: .4f} | Normalized ER: {: .4f}'.format(
                    i, s, total[0][0], norm[0][0]))
                print('Per-word ER:')
                print('\t'.join(s.split()))
                print('\t'.join(['{: .4}'.format(score) for score in array[0][:cut_off]]))
            print('-' * 10 + '\n')

            # Get cognitive load score (weighted sum of normalized surprisal and entropy reduction scores)
            print('Combined cognitive load:')
            for i, s in enumerate(samples):
                total, array, norm = interface.get_cognitive_load(samples)
                cut_off = len(s.split())
                print('{: d}: {:s} | Total CL: {: .4f} | Normalized CL: {: .4f}'.format(
                    i, s, total[0][0], norm[0][0]))
                print('Per-word CL:')
                print('\t'.join(s.split()))
                print('\t'.join(['{: .4}'.format(score) for score in array[0][:cut_off]]))
            print('-' * 10 + '\n')

        # Get model perplexity for the entire test set
        print('Model perplexity: {: .4f}'.format(interface.get_model_perplexity(test_data)[0][0]))
        print('-' * 10 + '\n')

        # Evaluate generative capability of the trained LM
        if generate:
            # Generate greedliy from scratch
            print('Sentences generated from scratch:')
            for c in range(gen_cycles):
                interface.generate(prefix=None, print_results=True)
            print('-' * 10 + '\n')

            # Generate greedily from some sentence prefix (i.e. a sentence completion test)
            print('Sentences generated from prefix:')
            for i, s in enumerate(samples):
                sent_list = s.split(' ')
                # Generate a sentence prefix of random length (at most 1/2 of the source sentence)
                cut_off = np.random.randint(1, len(sent_list) // 2)
                prefix = ' '.join(sent_list[:cut_off])
                print('Prefix: {:s} | Source: {:s}'.format(prefix, s))
                generated = interface.generate(prefix=prefix, print_results=False)
                for tpl in generated:
                    print('{:s} | Probability: {:.10f}'.format(tpl[0], tpl[1]))
                print('\n')

    print('-' * 10 + '\n')
    print('=' * 10 + '\n')
    print('Language model evaluation completed!')
