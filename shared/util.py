""" Defines a number of diverse, system-wide helper functions.

Contents:
1. Pickling
2. Graph saving and loading
3. Reporting
4. Corpus processing
5. Math functions

"""

import os
import sys
import time
import pickle
import codecs
import random
import logging
import numpy as np
import pandas as pd
import tensorflow as tf


# =========================================== Pickling ===========================================

def make_pickle(opt, data_processor, corpus_name, source_path, data_path, vocab_path=None, is_train=False,
                is_valid=False, is_test=False):
    """ Pickles corpus information for fast re-use; tracks the duration of the performed operations for
    rudimentary estimation of processing efficiency. """
    # Vocabulary objects are created for training sets only
    if not is_train:
        # Display the appropriate feedback to user
        if is_valid:
            print('Processing the validation data ...')
        elif is_test:
            print('Processing the test data ...')
        else:
            print('Processing full corpus data ...')

        g_st = time.time()
        sentences = data_processor(opt, source_path, corpus_name)
        g_diff = time.time() - g_st
        print('Data generation took {:d} minutes and {:.4f} seconds!'.format(int(g_diff // 60), g_diff % 60))

        p_st = time.time()
        with open(data_path, 'wb') as in_file:
            pickle.dump(sentences, in_file)
        p_diff = time.time() - p_st
        print('Pickling took {:d} minutes and {:.4f} seconds!'.format(int(p_diff // 60), p_diff % 60))

    else:
        print('Processing training vocab and data ...')
        g_st = time.time()
        vocab, sentences = data_processor(opt, source_path, corpus_name, zipf_sort=True, generate_vocab=True)
        g_diff = time.time() - g_st
        print('Data generation took {:d} minutes and {:.4f} seconds!'.format(int(g_diff // 60), g_diff % 60))

        if vocab_path is not None:
            pv_st = time.time()
            with open(vocab_path, 'wb') as in_file:
                pickle.dump(vocab, in_file)
            pv_diff = time.time() - pv_st
            print('Vocab pickling took {:d} minutes and {:.4f} seconds!'.format(int(pv_diff // 60), pv_diff % 60))

        pd_st = time.time()
        with open(data_path, 'wb') as in_file:
            pickle.dump(sentences, in_file)
        pd_diff = time.time() - pd_st
        print('Data pickling took {:d} minutes and {:.4f} seconds!'.format(int(pd_diff // 60), pd_diff % 60))


def load_pickle(pickle_path):
    """ Un-pickles corpus information (or any pickle, in general). """
    with open(pickle_path, 'rb') as out_file:
        return pickle.load(out_file)


# =========================================== Graph saving and loading ===========================================


def save_model(session, model, model_saver, save_dir, source_epoch):
    """ Saves the model to the specified save directory. """
    # Epoch designations are limited to 'best', 'final', and time-stamps
    unique = ['best', 'final']
    # Generate the appropriate checkpoint name
    if source_epoch in unique:
        file_name = '{:s}_{:s}.ckpt'.format(str(source_epoch), model.name)
    else:
        time_tuple = time.localtime(time.time())
        time_stamp = '{:d}.{:d}.{:d}_{:d}:{:d}:{:d}' \
            .format(time_tuple[2], time_tuple[1], time_tuple[0], time_tuple[3], time_tuple[4], time_tuple[5])
        file_name = '{:s}_{:s}_{:s}.ckpt'.format(str(source_epoch), time_stamp, model.name)
    # Save
    save_path = model_saver.save(session, os.path.join(save_dir, file_name))
    # Report
    logging.info('{:s} model {:s} has been saved in file {:s}'.format(model.name, file_name, save_path))


def load_model(session, model_saver, save_dir, target_epoch):
    """ Loads the specified checkpoint from the designated save directory. """
    # Retrieve the correct checkpoint file
    checkpoints = [
        ckpt for ckpt in os.listdir(save_dir) if os.path.isfile(os.path.join(save_dir, ckpt)) and 'meta' in ckpt]
    if target_epoch is None:
        load_from = [ckpt for ckpt in checkpoints if ckpt.startswith('best')]
    else:
        load_from = [ckpt for ckpt in checkpoints if ckpt.startswith(str(target_epoch))]
    file_name = '.'.join(load_from[0].split('.')[:-1])
    file_path = os.path.join(save_dir, file_name)
    # Load
    model_saver.restore(session, file_path)
    # Report
    logging.info('Model restored from {:s}'.format(file_name))

# =========================================== Reporting ===========================================


def print_off():
    """ Suppresses print output; see: stackoverflow.com/questions/8391411/suppress-calls-to-print-python"""
    sys.stdout = open(os.devnull, 'w')


def print_on():
    """ Re-enables print output; same source as above."""
    sys.stdout = sys.__stdout__


# =========================================== Corpus processing ===========================================


def clean_europarl(source_path, clean_path, keep_above=2):
    """ Removes lines of length <= 2 from the corpus so as to make the training more stable. """
    line_count = 0
    word_count = 0
    with codecs.open(source_path, 'r', encoding='utf8') as out_file:
        with open(clean_path, 'w') as in_file:
            for line in out_file:
                line_length = len(line.split())
                if line_length > keep_above:
                    in_file.write(line)
                    line_count += 1
                    word_count += line_length
    # Report the outcome of the cleaning process
    print('Corpus cleaned. Cleaned corpus contains {:d} lines, totaling up to {:d} words.'.
          format(line_count, word_count))


def truncate_europarl(source_path, truncated_path, truncated_length=100000):
    """ Truncates the Europarl v7 monolingual English (or any) corpus to the specified length. """
    with codecs.open(source_path, 'r', encoding='utf8') as out_file:
        word_count = 0
        with open(truncated_path, 'w') as in_file:
            for i, line in enumerate(out_file):
                if i < truncated_length:
                    in_file.write(line)
                    word_count += len(line.split())
    # Report the scope of the truncated corpus
    print('Corpus truncated to {:d} lines, totaling up to {:d} words.'.format(truncated_length, word_count))


def train_valid_test_split(source_path, train_path, valid_path, test_path, split_fractions):
    """ Splits the specified source corpus in training, validation, and testing sub-corpora according to
    the proportions specified in split_factions. """
    with open(source_path, 'r') as out_file:
        # Read in the full corpus
        all_lines = out_file.readlines()
        # Shuffle to increase the diversity of sentences contained in each of the split sets
        random.shuffle(all_lines)
        source_len = len(all_lines)
        # Determine cut-off points for each split
        train_bound = int(source_len * split_fractions[0])
        valid_bound = int(source_len * (split_fractions[0] + split_fractions[1]))
        # Split source corpus in train/ valid/ test sets
        with open(train_path, 'w') as train_file:
            for line in all_lines[: train_bound]:
                train_file.write(line.strip() + '\n')
        with open(valid_path, 'w') as valid_file:
            for line in all_lines[train_bound: valid_bound]:
                valid_file.write(line.strip() + '\n')
        with open(test_path, 'w') as test_file:
            for line in all_lines[valid_bound:]:
                test_file.write(line.strip() + '\n')
    print('Train-valid-test-split successfully completed.')


def shrink_domain(scored_path, reduced_path, keep_fraction=0.9):
    """ Prunes 1.0 - keep_faction of the total source corpus size in outliers, as determined by model perplexity scores
    assigned to each of the corpus sentences by a trained language model. """
    # Read in the source corpus
    df_full = pd.read_table(scored_path, header=None, names=['Sentence', 'Sentence_Perplexity'], skip_blank_lines=True)
    # Sort dataframe by sentence-wise model perplexity scores, ascending
    df_full = df_full.sort_values('Sentence_Perplexity', ascending=True)
    # Prune the lowest 1.0 - keep_fraction of the dataframe
    df_shrunk = df_full.iloc[0: int(len(df_full) * keep_fraction), 0]
    # Shuffle the retained dataframe and write the result to file
    df_shrunk = df_shrunk.iloc[np.random.permutation(len(df_shrunk))]
    with open(reduced_path, 'w') as in_file:
        for entry_id in range(len(df_shrunk)):
            line = df_shrunk.iloc[entry_id].strip() + '\n'
            in_file.write(line)
    print('Corpus domain successfully restricted.')


def id_split(annotated_path, low_path, high_path):
    """ Splits the annotated corpus into low-ID and a high-ID sub-corpora, each containing an identical
    number of samples; the so obtained corpora are used in both stages of IDGAN training. """
    # Read in the source corpus
    df_annotated = pd.read_table(annotated_path, header=None,
                                 names=['Sentence', 'Total_surprisal', 'Per_word_surprisal', 'Normalized_surprisal',
                                        'Total_UID_divergence', 'Per_word_UID_divergence', 'Normalized_UID_divergence'],
                                 skip_blank_lines=True)
    # Sort dataframe along sentence-wise normalized surprisal scores
    df_annotated = df_annotated.loc[:, ['Sentence', 'Normalized_surprisal']]
    df_annotated = df_annotated.sort_values('Normalized_surprisal', ascending=True)
    # Split dataframe along the median surprisal value
    median_row = len(df_annotated) // 2
    df_low = df_annotated.iloc[: median_row, :]
    df_high = df_annotated.iloc[median_row:, :]
    id_variant_corpora = [(df_low, low_path), (df_high, high_path)]
    # Shuffle the derived corpora and write the result to file
    for tpl in id_variant_corpora:
        # Calculate ID-related corpus statistics
        corpus_name = tpl[1].split('/')[-1]
        ns_mean = tpl[0]['Normalized_surprisal'].mean()
        ns_median = tpl[0]['Normalized_surprisal'].median()
        ns_min = tpl[0]['Normalized_surprisal'].min()
        ns_max = tpl[0]['Normalized_surprisal'].max()
        sent_lens = 0
        corpus = tpl[0].iloc[np.random.permutation(len(tpl[0]))]
        with open(tpl[1], 'w') as in_file:
            for entry_id in range(len(corpus)):
                line = corpus.iloc[entry_id][0].strip()
                in_file.write(line + '\n')
                sent_lens += len(line.split())
        mean_sent_len = sent_lens / len(corpus)
        # Report ID-relevant statistics
        print('{:s} corpus: Mean NS: {:.4f} | Median NS: {:.4f} | Min NS: {:.4f} | Max NS: {:.4f} | '
              'Mean sentence length: {:.4f}'
              .format(corpus_name, ns_mean, ns_median, ns_min, ns_max, mean_sent_len))
    print('Corpus successfully subdivided according to the chosen ID criterion.')


# =========================================== Math functions ===========================================


def padded_log(input_tensor):
    """ Prevents NaNs during log computations, see github.com/AYLIEN/IDGAN-intro/blob/master/IDGAN.py. """
    return tf.log(tf.maximum(input_tensor, 1e-5))
