""" A suite of functions for evaluating ID-variant corpora along (mostly) linguistic criteria.
Results are logged in text form and visualized in a format best-suited to the nature of the collected data. """

import os
import codecs

import numpy as np
import spacy as sc
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from nltk import ngrams
from string import digits
from string import ascii_uppercase
from scipy import stats
from cognitive_language_model.src.codebase.preprocessing import Indexer, read_text_file


def read_file(corpus_path, corpus_name, target_dir, opt):
    """ Generates a frequency-sorted vocabulary from a corpus file. """
    print('Generating the vocab file for the {:s} corpus ...'.format(corpus_name))
    corpus_sents, _ = read_text_file(corpus_path, None, lower=True)
    # Create and populate an Indexer object holing the target vocabulary
    corpus_vocab = Indexer(opt, corpus_name, zipf_sort=True)
    for i in range(len(corpus_sents)):
        corpus_vocab.add_sentence(corpus_sents[i])
    corpus_vocab.map_ids()
    vocab_log_path = os.path.join(target_dir, '{:s}_raw_vocab.txt'.format(corpus_name))
    # Write so obtained vocabulary to file, including word rank, identity, and frequency
    with codecs.open(vocab_log_path, 'w', encoding='utf8') as vocab_file:
        rank = 1
        for value in corpus_vocab.index_to_word.values():
            try:
                vocab_file.write('{:d}\t{:s}\t{:d}\n'.format(rank, value, corpus_vocab.word_to_freq[value]))
                rank += 1
            except KeyError:
                continue
        # Calculate corpus statistics
        vocab_file.write('=' * 10 + '\n')
        vocab_file.write('Word frequency mean: {:.4f}\n'.format(np.mean(list(corpus_vocab.word_to_freq.values()))))
        vocab_file.write('Word frequency standard deviation: {:.4f}\n'
                         .format(np.std(list(corpus_vocab.word_to_freq.values()))))
    print('Done.')
    return corpus_sents, corpus_vocab


def get_length_stats(corpus_sents, corpus_name, target_dir):
    """ Collects sentence length counts for the specified corpus. """
    # Collect individual sentence lengths associated with sentences within the corpus
    sent_lens = [len(sent.split()) for sent in corpus_sents]
    unique_lens = set(sent_lens)
    # Count length frequencies
    len_counts = [(a_len, sent_lens.count(a_len)) for a_len in unique_lens]
    len_counts_sorted = sorted(len_counts, reverse=True, key=lambda x: x[1])
    lens_log_path = os.path.join(target_dir, '{:s}_sentence_lengths.txt'.format(corpus_name))
    # Write length counts to file
    with codecs.open(lens_log_path, 'w', encoding='utf8') as len_file:
        for i in range(len(len_counts_sorted)):
            len_file.write('{:d}\t{:d}\n'.format(len_counts_sorted[i][0], len_counts_sorted[i][1]))
        # Calculate corpus statistics
        len_file.write('=' * 10 + '\n')
        len_file.write('Sentence length max: {:d}\n'.format(np.max(sent_lens)))
        len_file.write('Sentence length min: {:d}\n'.format(np.min(sent_lens)))
        len_file.write('Sentence length mean: {:.4f}\n'.format(np.mean(sent_lens)))
        len_file.write('Sentence length standard deviation: {:.4f}\n'.format(np.std(sent_lens)))
    print('Done.')


def get_ngrams(corpus_sents, gram, corpus_name, target_dir):
    """ Generates a set of n-grams for the specified granularity, corpus-wise;
    here: used for 2-grams and 3-grams. """
    print('Generating the {:d}-gram file for the {:s} corpus ...'.format(gram, corpus_name))
    # Collect n-grams present within the corpus
    ngram_lists = [list(ngrams(sent.split(), gram)) for sent in corpus_sents]
    flat_ngrams = [ngram for ngram_list in ngram_lists for ngram in ngram_list]
    # Assemble n-gram frequency dictionary
    ngram_dict = dict()
    for ngram in flat_ngrams:
        if ngram in ngram_dict.keys():
            ngram_dict[ngram] += 1
        else:
            ngram_dict[ngram] = 1
    # Count the occurrences of unique n-grams
    ngram_counts = list(ngram_dict.items())
    # Sort n-grams by frequency
    ngram_counts_sorted = sorted(ngram_counts, reverse=True, key=lambda x: x[1])
    # Write n-gram distribution to file
    ngram_log_path = os.path.join(target_dir, '{:s}_{:d}-gram_counts.txt'.format(corpus_name, gram))
    with codecs.open(ngram_log_path, 'w', encoding='utf8') as ngram_file:
        for i in range(len(ngram_counts_sorted)):
            ngram_file.write('{:d}\t{}\t{:d}\n'.format(i + 1, ngram_counts_sorted[i][0], ngram_counts_sorted[i][1]))
    print('Done.')


def get_parses(corpus_sents, corpus_name, target_dir):
    """ Annotates a set of sentences with POS tags and dependency parses,
    tracking tag frequencies and dependency arc lengths. """
    print('Generating the parse files for the {:s} corpus ...'.format(corpus_name))
    # Define POS tag inventory separated along the open/ closed class axis;
    # exclude tags not associated with either class (such as 'filler' words), due to their relatively low frequency
    # and low relevance for the contrastive analysis of the two ID-variant corpora
    open_class_tags = ['FW', 'GW', 'JJ', 'JJR', 'JJS', 'NN', 'NNP', 'NNPS', 'NNS', 'PDT', 'RB', 'RBR', 'RBS', 'UH',
                       'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WRB']
    closed_class_tags = ['AFX', 'BES', 'CC', 'CD', 'DT', 'EX', 'HVS', 'IN', 'MD', 'POS', 'PRP', 'PRP$', 'RP', 'SYM',
                         'TO', 'WDT', 'WP', 'WP$']
    all_tags = open_class_tags + closed_class_tags

    # Parse corpus contents with SpaCy
    model = sc.load('en')
    parses = [model(sent) for sent in corpus_sents]
    # Obtain tag counts for the specified tag inventory
    flat_tags = [parse.tag_ for parsed_sent in parses for parse in parsed_sent]
    unique_tags = set(flat_tags)
    tag_counts = sorted([(tag, flat_tags.count(tag)) for tag in unique_tags if tag in all_tags],
                        reverse=True, key=lambda x: x[1])

    # Calculate open class fraction (total and top 50%), to determine whether open classes are distributed differently
    # in sentences of varying ID; intuitively one may expect high-ID sentences to contain a greater portion of open
    # class words, as they exhibit greater variation and are therefore less predictable in sentential context
    top_open = [tag_tpl[1] for tag_tpl in tag_counts[: len(tag_counts) // 2] if tag_tpl[0] in open_class_tags]
    top_closed = [tag_tpl[1] for tag_tpl in tag_counts[: len(tag_counts) // 2] if tag_tpl[0] in closed_class_tags]
    top_all = [tag_tpl[1] for tag_tpl in tag_counts[: len(tag_counts) // 2]]
    top_open_fraction = sum(top_open) / sum(top_all)
    top_closed_fraction = sum(top_closed) / sum(top_all)

    full_open = [tag_tpl[1] for tag_tpl in tag_counts if tag_tpl[0] in open_class_tags]
    full_closed = [tag_tpl[1] for tag_tpl in tag_counts if tag_tpl[0] in closed_class_tags]
    full_all = [tag_tpl[1] for tag_tpl in tag_counts]
    full_open_fraction = sum(full_open) / sum(full_all)
    full_closed_fraction = sum(full_closed) / sum(full_all)

    # Write tag counts to file
    tag_log_path = os.path.join(target_dir, '{:s}_tag_counts.txt'.format(corpus_name))
    with codecs.open(tag_log_path, 'w', encoding='utf8') as tag_file:
        for i in range(len(tag_counts)):
            tag_file.write('{:s}\t{:d}\n'.format(tag_counts[i][0], tag_counts[i][1]))
        # Calculate corpus statistics
        tag_file.write('=' * 10 + '\n')
        tag_file.write('Open class fraction of most frequent 50% POS tags: {:.4f}\n'.format(top_open_fraction))
        tag_file.write('Closed class fraction of most frequent 50% POS tags: {:.4f}\n'.format(top_closed_fraction))
        tag_file.write('Open class fraction of all identified POS tags: {:.4f}\n'.format(full_open_fraction))
        tag_file.write('Closed class fraction of all identified POS tags: {:.4f}'.format(full_closed_fraction))
    print('Done with POS-tagging.')

    # Perform dependency parsing related analysis
    def _get_dlt(_parent, _children):
        """ Computes the integration cost at the head of dependency relations identified within the input sentence,
        according to the Dependency Locality Theory. """
        dlt_cost = 0
        for child in _children:
            # Determine the span length between the child and parent node
            left = min(_parent.i, child.i)
            right = max(_parent.i, child.i)
            for j in range(left + 1, right):
                # Identify discourse referents present within the determined span
                if 'NN' in parse[j].tag_ or 'VB' in parse[j].tag_:
                    dlt_cost += 1
            # Check if the parent node is also occupied by a new discourse referent
            if 'NN' in _parent.tag_ or 'VB' in _parent.tag_:
                dlt_cost += 1
        return dlt_cost

    corpus_spans = list()
    corpus_costs = list()
    # Compute the mean dependency arc length and DLT integration cost for each sentence within the corpus
    for parse in parses:
        sent_spans = list()
        sent_costs = list()
        for parent in parse:
            children = [w for w in parent.lefts] + [w for w in parent.rights]
            if len(children) == 0:
                continue
            parent_spans = [abs(parent.i - child.i) for child in children]
            sent_spans += parent_spans
            sent_costs += [_get_dlt(parent, children)]
        # Collect means
        corpus_spans += [np.mean(sent_spans)]
        corpus_costs += [np.mean(sent_costs)]

    # Calculate SVO fraction (ultimately did not yield any interesting insights)
    clause_triples = list()
    svo_count = 0
    other_count = 0
    for parse in parses:
        # Identify subjects, predicates, and objects
        subjects = [[word.i, word.head.i] for word in parse if 'subj' in word.dep_ and word.head.pos_ == 'VERB']
        objects = [[word.head.i, word.i] for word in parse if 'obj' in word.dep_]
        for subj_list in subjects:
            for obj_list in objects:
                if subj_list[-1] == obj_list[0]:
                    clause_triple = subj_list + obj_list[-1:]
                    clause_triples.append(clause_triple)
                    # Check if isolated triples are in the SVO order, increment counter if so
                    if clause_triple[0] < clause_triple[1] < clause_triple[2]:
                        svo_count += 1
                    else:
                        other_count += 1
    # Compute word order fractions
    svo_fraction = svo_count / len(clause_triples)
    other_fraction = other_count / len(clause_triples)

    # Write mean sentence-wise dependency arc lengths and DLT integration costs to file
    parse_log_path = os.path.join(target_dir, '{:s}_parse_stats.txt'.format(corpus_name))
    with codecs.open(parse_log_path, 'w', encoding='utf8') as parse_file:
        # Document mean sentence dependency arc length and mean sentence DLT integration cost
        for i in range(len(corpus_spans)):
            parse_file.write('{:.4f}\t{:.4f}\n'.format(corpus_spans[i], corpus_costs[i]))
        # Calculate corpus statistics
        parse_file.write('=' * 10 + '\n')
        parse_file.write('Span length max: {:.4f}\n'.format(np.max(corpus_spans)))
        parse_file.write('Span length min: {:.4f}\n'.format(np.min(corpus_spans)))
        parse_file.write('Span length mean: {:.4f}\n'.format(np.mean(corpus_spans)))
        parse_file.write('Span length standard deviation: {:.4f}\n'.format(np.std(corpus_spans)))
        parse_file.write('=' * 10 + '\n')
        parse_file.write('DLT cost max: {:.4f}\n'.format(np.max(corpus_costs)))
        parse_file.write('DLT cost min: {:.4f}\n'.format(np.min(corpus_costs)))
        parse_file.write('DLT cost mean: {:.4f}\n'.format(np.mean(corpus_costs)))
        parse_file.write('DLT cost standard deviation: {:.4f}\n'.format(np.std(corpus_costs)))
        # Document word order distribution
        parse_file.write('=' * 10 + '\n')
        parse_file.write('SVO clauses count: {:d}\n'.format(svo_count))
        parse_file.write('SVO clauses fraction: {:.4f}\n'.format(svo_fraction))
        parse_file.write('Other clauses count: {:d}\n'.format(other_count))
        parse_file.write('Other clauses fraction: {:.4f}'.format(other_fraction))
    print('Done with dependency parsing.')


def get_vocab_overlap(vocab_a, vocab_b, freq_bound, corpus_name_a, corpus_name_b, target_dir):
    """ Calculates the overlap of the vocabularies provided, total and among words occurring with a frequency
    greater than the specified threshold; no immediately interpretable results could be obtained. """
    print('Comparing corpora-specific vocabularies ...')

    # Keep a list of individual word types for each vocabulary
    word_list_a = list(vocab_a.word_to_index.keys())
    word_list_b = list(vocab_b.word_to_index.keys())
    # Extract 'high-frequency' words (frequency bound is set arbitrarily)
    bound_list_a = [word for word in word_list_a if word in vocab_a.word_to_freq.keys() and
                    vocab_a.word_to_freq[word] >= freq_bound]
    bound_list_b = [word for word in word_list_b if word in vocab_b.word_to_freq.keys() and
                    vocab_b.word_to_freq[word] >= freq_bound]

    # Calculate total word type overlap
    total_shared_vocab = [word for word in word_list_a if word in word_list_b]
    total_shared_words = len(total_shared_vocab)
    vocab_a_total_unique_words = len(word_list_a) - total_shared_words
    vocab_b_total_unique_words = len(word_list_b) - total_shared_words
    # Calculate frequency-bounded word type overlap
    bound_shared_vocab = [word for word in bound_list_a if word in bound_list_b]
    bound_shared_words = len(bound_shared_vocab)
    vocab_a_bound_unique_words = len(bound_list_a) - bound_shared_words
    vocab_b_bound_unique_words = len(bound_list_b) - bound_shared_words

    # Calculate overlap fractions
    vocab_a_total_overlap_fraction = total_shared_words / len(word_list_a)
    vocab_a_bound_overlap_fraction = bound_shared_words / len(bound_list_a)
    vocab_b_total_overlap_fraction = total_shared_words / len(word_list_b)
    vocab_b_bound_overlap_fraction = bound_shared_words / len(bound_list_b)

    # Write collected information to file
    overlap_log_path = os.path.join(target_dir, '{:s}_{:s}_vocab_overlap.txt'.format(corpus_name_a, corpus_name_b))
    with codecs.open(overlap_log_path, 'w', encoding='utf8') as overlap_file:
        overlap_file.write('Compared corpora: {:s} and {:s}\n'.format(corpus_name_a, corpus_name_b))
        overlap_file.write('=' * 10 + '\n')
        overlap_file.write('Total {:s} vocabulary size: {:d} words\n'.format(corpus_name_a, len(word_list_a)))
        overlap_file.write('Total {:s} vocabulary size: {:d} words\n'.format(corpus_name_b, len(word_list_b)))
        overlap_file.write('Bound {:s} vocabulary size for bound {:d}: {:d} words\n'
                           .format(corpus_name_a, freq_bound, len(bound_list_a)))
        overlap_file.write('Bound fraction for {:s} corpus: {:.4f} words\n'
                           .format(corpus_name_a, len(bound_list_a) / len(word_list_a)))
        overlap_file.write('Bound {:s} vocabulary for bound {:d} size: {:d} words\n'
                           .format(corpus_name_b, freq_bound, len(bound_list_b)))
        overlap_file.write('Bound fraction for {:s} corpus: {:.4f} words\n'
                           .format(corpus_name_b, len(bound_list_b) / len(word_list_b)))
        overlap_file.write('=' * 10 + '\n')
        overlap_file.write('Total overlap size: {:d} words\n'.format(total_shared_words))
        overlap_file.write('Total unique words for corpus {:s}: {:d}\n'
                           .format(corpus_name_a, vocab_a_total_unique_words))
        overlap_file.write('Total unique words for corpus {:s}: {:d}\n'
                           .format(corpus_name_b, vocab_b_total_unique_words))
        overlap_file.write('Frequency-bound overlap size for bound {:d}: {:d} words\n'.
                           format(freq_bound, bound_shared_words))
        overlap_file.write('Frequency-bound unique words for corpus {:s}: {:d}\n'
                           .format(corpus_name_a, vocab_a_bound_unique_words))
        overlap_file.write('Frequency-bound unique words for corpus {:s}: {:d}\n'
                           .format(corpus_name_b, vocab_b_bound_unique_words))
        overlap_file.write('=' * 10 + '\n')
        overlap_file.write('Total overlap fraction for corpus {:s}: {:.4f}\n'
                           .format(corpus_name_a, vocab_a_total_overlap_fraction))
        overlap_file.write('Frequency-bound overlap fraction for corpus {:s} for bound {:d}: {:.4f}\n'
                           .format(corpus_name_a, freq_bound, vocab_a_bound_overlap_fraction))
        overlap_file.write('=' * 10 + '\n')
        overlap_file.write('Total overlap fraction for corpus {:s}: {:.4f}\n'
                           .format(corpus_name_b, vocab_b_total_overlap_fraction))
        overlap_file.write('Frequency-bound overlap fraction for corpus {:s} for bound {:d}: {:.4f}\n'
                           .format(corpus_name_b, freq_bound, vocab_b_bound_overlap_fraction))
    print('Done.')


def get_ngram_overlap(ngram_file_a, ngram_file_b, freq_bound, corpus_name_a, corpus_name_b, target_dir):
    """ Calculates the overlap of the ngram-lists provided, total and among ngrams occurring with a frequency
    greater than the specified threshold; no immediately interpretable results could be obtained."""
    print('Comparing n-gram inventories ...')

    # Read in corpus files
    df_ngram_counts_a = pd.read_table(ngram_file_a, header=None, names=['Rank', 'Ngram', 'Counts'],
                                      skip_blank_lines=True)
    df_ngram_counts_b = pd.read_table(ngram_file_b, header=None, names=['Rank', 'Ngram', 'Counts'],
                                      skip_blank_lines=True)
    # Build n-gram inventories
    ngram_list_a = [df_ngram_counts_a.iloc[row_id, 1] for row_id in range(len(df_ngram_counts_a))]
    ngram_list_b = [df_ngram_counts_b.iloc[row_id, 1] for row_id in range(len(df_ngram_counts_b))]
    bound_list_a = [df_ngram_counts_a.iloc[row_id, 1] for row_id in range(len(df_ngram_counts_a))
                    if int(df_ngram_counts_a.iloc[row_id, 2]) >= freq_bound]
    bound_list_b = [df_ngram_counts_b.iloc[row_id, 1] for row_id in range(len(df_ngram_counts_b))
                    if int(df_ngram_counts_b.iloc[row_id, 2]) >= freq_bound]

    # Calculate total unique n-gram overlap
    total_shared_ngrams = [ngram for ngram in ngram_list_a if ngram in ngram_list_b]
    total_shared_count = len(total_shared_ngrams)
    total_unique_ngrams_a = len(ngram_list_a) - total_shared_count
    total_unique_ngrams_b = len(ngram_list_b) - total_shared_count

    # Calculate frequency-bounded unique ngram overlap
    bound_shared_ngrams = [ngram for ngram in bound_list_a if ngram in bound_list_b]
    bound_shared_count = len(bound_shared_ngrams)
    bound_unique_ngrams_a = len(bound_list_a) - bound_shared_count
    bound_unique_ngrams_b = len(bound_list_b) - bound_shared_count

    # Calculate overlap fractions
    total_overlap_fraction_a = total_shared_count / len(ngram_list_a)
    bound_overlap_fraction_a = bound_shared_count / len(bound_list_a)
    total_overlap_fraction_b = total_shared_count / len(ngram_list_b)
    bound_overlap_fraction_b = bound_shared_count / len(bound_list_b)

    # Write collected information to file
    gram_size = len(total_shared_ngrams[0].split())
    overlap_log_path = os.path.join(target_dir, '{:s}_{:s}_{:d}-gram_overlap.txt'
                                    .format(corpus_name_a, corpus_name_b, gram_size))
    with codecs.open(overlap_log_path, 'w', encoding='utf8') as overlap_file:
        overlap_file.write('Compared corpora: {:s} and {:s}\n'.format(corpus_name_a, corpus_name_b))
        overlap_file.write('=' * 10 + '\n')
        overlap_file.write('Total {:s} vocabulary size: {:d} entries\n'.format(corpus_name_a, len(ngram_list_a)))
        overlap_file.write('Total {:s} vocabulary size: {:d} entries\n'.format(corpus_name_b, len(ngram_list_b)))
        overlap_file.write('Bound {:s} vocabulary size for bound {:d}: {:d} entries\n'
                           .format(corpus_name_a, freq_bound, len(bound_list_a)))
        overlap_file.write('Bound fraction for {:s} corpus: {:.4f} entries\n'
                           .format(corpus_name_a, len(bound_list_a) / len(ngram_list_a)))
        overlap_file.write('Bound {:s} vocabulary for bound {:d} size: {:d} entries\n'
                           .format(corpus_name_b, freq_bound, len(bound_list_b)))
        overlap_file.write('Bound fraction for {:s} corpus: {:.4f} entries\n'
                           .format(corpus_name_b, len(bound_list_b) / len(ngram_list_b)))
        overlap_file.write('=' * 10 + '\n')
        overlap_file.write('Total overlap size: {:d} entries\n'.format(total_shared_count))
        overlap_file.write('Total unique {:d}-grams for corpus {:s}: {:d}\n'
                           .format(gram_size, corpus_name_a, total_unique_ngrams_a))
        overlap_file.write('Total unique {:d}-grams for corpus {:s}: {:d}\n'
                           .format(gram_size, corpus_name_b, total_unique_ngrams_b))
        overlap_file.write('Frequency-bound overlap size for bound {:d}: {:d} entries\n'.
                           format(freq_bound, bound_shared_count))
        overlap_file.write('Frequency-bound unique {:d}-grams for corpus {:s}: {:d}\n'
                           .format(gram_size, corpus_name_a, bound_unique_ngrams_a))
        overlap_file.write('Frequency-bound unique {:d}-grams for corpus {:s}: {:d}\n'
                           .format(gram_size, corpus_name_b, bound_unique_ngrams_b))
        overlap_file.write('=' * 10 + '\n')
        overlap_file.write('Total overlap fraction for corpus {:s}: {:.4f}\n'
                           .format(corpus_name_a, total_overlap_fraction_a))
        overlap_file.write('Frequency-bound overlap fraction for corpus {:s} for bound {:d}: {:.4f}\n'
                           .format(corpus_name_a, freq_bound, bound_overlap_fraction_a))
        overlap_file.write('=' * 10 + '\n')
        overlap_file.write('Total overlap fraction for corpus {:s}: {:.4f}\n'
                           .format(corpus_name_b, total_overlap_fraction_b))
        overlap_file.write('Frequency-bound overlap fraction for corpus {:s} for bound {:d}: {:.4f}\n'
                           .format(corpus_name_b, freq_bound, bound_overlap_fraction_b))
    print('Done.')


def construct_annotated_corpora(extraction_path, id_variant_path, corpus_name, target_dir):
    """ Compiles ID-variant corpora annotated with evaluation-relevant information, i.e. normalized surprisal,
    normalized UID, and sentence length, by extracting low-ID and high-ID entries from the annotated 90k Europarl
    corpus. """
    # Read in main ID-annotated file
    df_annotated = pd.read_table(extraction_path, header=None,
                                 names=['Sentence', 'Total_surprisal', 'Per_word_surprisal', 'Normalized_surprisal',
                                        'Total_UID_divergence', 'Per_word_UID_divergence', 'Normalized_UID_divergence'],
                                 skip_blank_lines=True)
    if id_variant_path is not None:
        # Extract ID-specific sentences from the reference corpus
        df_variant = pd.read_table(id_variant_path, header=None, names=['Sentence'], skip_blank_lines=True)
        target_list = df_variant.iloc[:, 0].tolist()
        target_list = [sent.strip() for sent in target_list]
    else:
        # No extraction, entire reference corpus is considered for further steps
        target_list = df_annotated.iloc[:, 0].tolist()
        target_list = [sent.strip() for sent in target_list]

    # Isolate evaluation-relevant features
    df_features = df_annotated.loc[:, ['Sentence', 'Normalized_surprisal', 'Normalized_UID_divergence']]
    surprisals = list()
    uid_divs = list()

    # Write the normalized surprisal and UID divergence distributions to file
    features_log_path = os.path.join(target_dir, '{:s}_ID_features.txt'.format(corpus_name))
    print('Writing to {:s} ...'.format(features_log_path))
    with open(features_log_path, 'w') as id_file:
        for line_id in range(len(df_features)):
            sent = df_features.iloc[line_id][0]
            sent_ns = df_features.iloc[line_id][1]
            sent_nud = df_features.iloc[line_id][2]

            if sent in target_list:
                id_file.write('{:f}\t{:f}\n'.format(sent_ns, sent_nud))
                surprisals += [float(sent_ns)]
                uid_divs += [float(sent_nud)]
        # Calculate corpus statistics
        id_file.write('=' * 10 + '\n')
        id_file.write('Surprisal max: {:.4f}\n'.format(np.max(surprisals)))
        id_file.write('Surprisal min: {:.4f}\n'.format(np.min(surprisals)))
        id_file.write('Surprisal mean: {:.4f}\n'.format(np.mean(surprisals)))
        id_file.write('Surprisal standard deviation: {:.4f}\n'.format(np.std(surprisals)))
        id_file.write('=' * 10 + '\n')
        id_file.write('UID divergence max: {:.4f}\n'.format(np.max(uid_divs)))
        id_file.write('UID divergence min: {:.4f}\n'.format(np.min(uid_divs)))
        id_file.write('UID divergence mean: {:.4f}\n'.format(np.mean(uid_divs)))
        id_file.write('UID divergence standard deviation: {:.4f}\n'.format(np.std(uid_divs)))
    print('Done.')


def plot_dist(data_source, column_id, x_label, y_label, title=None, dtype=float):
    """ Plots the histogram and the corresponding kernel density estimate for checking whether the data distribution is
    approximately normal. """

    def _filter_data(raw_data):
        """ Filters plot-able data, by excluding lines containing corpus statistics and related information. """
        legal_inventory = digits + '.'
        filtered_data = list()
        # Only retain numeric information
        for data_point in raw_data:
            skip = False
            for symbol in list(str(data_point)):
                if symbol not in legal_inventory:
                    skip = True
            if not skip:
                filtered_data.append(dtype(data_point))
        return np.array(filtered_data)

    # Set plot parameters
    sns.set_style('whitegrid')
    sns.set_context('paper')
    # Read in source dataframe
    df_features = pd.read_table(data_source, header=None, skip_blank_lines=True)
    # Designate data to be visualized within the data base
    if len(column_id) > 1:
        # Transform data into a format better suited for a density plot
        entries = _filter_data(df_features.iloc[:, column_id[0]].values)
        counts = _filter_data(df_features.iloc[:, column_id[1]].values)
        data_source = list()
        for i in range(entries.shape[0]):
            data_source += [entries[i]] * counts[i]
        data_source = np.array(data_source)
    else:
        data_source = _filter_data(df_features.iloc[:, column_id[0]].values)

    assert (type(data_source) == np.ndarray and len(data_source.shape) == 1), \
        'Expected a one-dimensional numpy array.'

    # Make plot
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 6)
    sns.distplot(data_source, kde=False, ax=ax, hist_kws=dict(edgecolor='w', linewidth=1))
    # Adjust visuals
    sns.despine()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if title is not None:
        plt.title(title)
    plt.show()


def plot_linear(data_source, column_ids, log_scale, x_label, y_label, title=None, dtype=int):
    """ Visualizes a linear relationship between the provided data points. """

    def _filter_data(raw_data):
        """ Filters plot-able data. """
        legal_inventory = digits + '.'
        filtered_data = list()
        # Only retain numeric information
        for data_point in raw_data:
            skip = False
            for symbol in list(str(data_point)):
                if symbol not in legal_inventory:
                    skip = True
            if not skip:
                filtered_data.append(dtype(data_point))
        return np.array(filtered_data)

    # Set plot parameters
    sns.set_style('whitegrid')
    sns.set_context('paper')

    # Read in the source dataframe and isolate columns to be plotted
    df_features = pd.read_table(data_source, header=None, skip_blank_lines=True)
    x_data = _filter_data(df_features.iloc[:, column_ids[0]].values)
    y_data = _filter_data(df_features.iloc[:, column_ids[1]].values)

    assert (type(x_data) == np.ndarray and len(x_data.shape) == 1), \
        'Expected a one-dimensional numpy array.'
    assert (type(y_data) == np.ndarray and len(y_data.shape) == 1), \
        'Expected a one-dimensional numpy array.'

    # Optionally, use log axis (e.g. for plotting ranked word type frequencies)
    if log_scale:
        fig, ax = plt.subplots()
        ax.set(xscale="log", yscale="log")
    else:
        fig, ax = plt.subplots()
        ax = None

    fig.set_size_inches(8, 6)
    sns.regplot(x=x_data, y=y_data, ax=ax, fit_reg=False)
    sns.despine()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if title is not None:
        plt.title(title)
    plt.show()


def plot_bar(source_files, column_ids, column_names, normalize, sort, plot_difference, freq_bound, title=None,
             dtype=int):
    """ Produces bar plots on the basis of the provided data, useful for comparing discrete quantities of distinct
    entities. """

    def _filter_data(raw_data, numerical):
        """ Filters plot-able data. """
        # Retain numeric information
        legal_count_inventory = digits + '.'
        # Retain POS tags, also
        legal_entry_inventory = ascii_uppercase + '$'
        filtered_data = list()
        for data_point in raw_data:
            skip = False
            for symbol in list(str(data_point)):
                if symbol not in legal_count_inventory and symbol not in legal_entry_inventory:
                    skip = True
            if not skip:
                if numerical:
                    filtered_data.append(dtype(data_point))
                else:
                    filtered_data.append(data_point)
        # Optionally normalize count values, resulting in a proportion plot
        if numerical and normalize:
            filtered_data = filtered_data / np.sum(filtered_data)
        return np.array(filtered_data)

    # Set plot parameters
    sns.set_style('whitegrid')
    sns.set_context('paper')

    # Compile data to be plotted within a new dataframe
    # Not necessary, but convenient when plotting with seaborn
    source_dict = dict()
    # Read in data and sort alphanumeric features (e.g. POS tags) alphabetically
    df_features = pd.read_table(source_files[0], header=None, names=['Tag', 'Count'], skip_blank_lines=True)
    df_features = df_features.sort_values('Tag', ascending=True)
    df_reference = pd.read_table(source_files[1], header=None, names=['Tag', 'Count'], skip_blank_lines=True)
    df_reference = df_reference.sort_values('Tag', ascending=True)
    # Isolate columns to be plotted
    entries = _filter_data(df_features.iloc[:, column_ids[0]].values, False)
    counts = _filter_data(df_features.iloc[:, column_ids[1]].values, True)  # e.g. counts from corpus A
    reference_counts = _filter_data(df_reference.iloc[:, column_ids[1]].values, True)  # e.g. counts from corpus B
    # Construct dataframe to be visualized
    source_dict[column_names[0]] = entries
    source_dict['reference_counts'] = reference_counts
    # Generate frequency mask to exclude low-frequency features from the plot
    # Optional; results in a clearer, better readable visualization
    frequency_mask = np.array(
        [int(counts[i] >= freq_bound or reference_counts[i] >= freq_bound) for i in range(counts.shape[0])])
    source_dict['frequency_mask'] = frequency_mask
    # Calculate per-feature count differences (i.e. target counts vs. reference counts), if specified
    if plot_difference:
        diffs = counts - reference_counts
        source_dict[column_names[1]] = diffs
    else:
        source_dict[column_names[1]] = counts
    features = pd.DataFrame.from_dict(source_dict)
    # Sort by count value and apply frequency mask
    if sort:
        features = features.sort_values(column_names[0], ascending=True)
    if freq_bound > 0:
        features = features.drop(features[features.frequency_mask == 0].index)

    # Make plot
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 6)
    if plot_difference:
        colors = ['coral' if feature >= 0 else 'skyblue' for feature in features[column_names[1]]]
        sns.barplot(x=column_names[1], y=column_names[0], data=features, ax=ax, palette=colors)
    else:
        sns.barplot(x=column_names[1], y=column_names[0], data=features, ax=ax, palette='Set2')
    sns.despine()
    if title is not None:
        plt.title(title)
    plt.show()


def plot_grid_dist(source_files, data_ids, col_ids, column_names, title=None, dtype=float, count_values=False):
    """ Combines a set of histograms within one shared visualization. """

    def _filter_data(raw_data, _dtype=dtype):
        """ Filters plot-able data. """
        # Same functionality as for non-facet plots
        legal_inventory = digits + '.'
        filtered_data = list()
        for data_point in raw_data:
            skip = False
            for symbol in list(str(data_point)):
                if symbol not in legal_inventory:
                    skip = True
            if not skip:
                filtered_data.append(_dtype(data_point))
        return filtered_data

    # Combine data from multiple sources
    source_dict = {column_names[i]: list() for i in range(len(column_names))}
    assert (len(source_files) == len(data_ids) == len(col_ids)), \
        'Input lists should be of the same length.'
    for i in range(len(source_files)):
        df_features = pd.read_table(source_files[i], header=None, skip_blank_lines=True)
        features = _filter_data(df_features.iloc[:, data_ids[i]].values)
        # Change data format to be better suited for distribution plots
        if count_values:
            # Assumes feature frequencies are given in a column immediately to the right of the feature column
            counts = _filter_data(df_features.iloc[:, data_ids[i] + 1].values, _dtype=int)
            expanded_features = list()
            for j in range(len(features)):
                expanded_features += [features[j]] * counts[j]
            features = expanded_features
        source_dict[column_names[0]] += features
        source_dict[column_names[1]] += [col_ids[i]] * len(features)
    # Convert values into array-like form for dataframe creation
    for key in source_dict.keys():
        source_dict[key] = np.array(source_dict[key])
    # Compile dataframe to be plotted
    features = pd.DataFrame.from_dict(source_dict)

    # Set plot parameters
    sns.set_style('whitegrid')
    sns.set_context('paper')

    # Make facet-plot
    fgrid = sns.FacetGrid(features, col=column_names[1], size=4.5)
    fgrid.map(sns.distplot, column_names[0], kde=True, bins=50, hist_kws=dict(edgecolor='w', linewidth=1))
    plt.subplots_adjust(top=0.85)
    if title is not None:
        fgrid.fig.suptitle(title)
    sns.despine()
    plt.show()


def plot_grid_linear(source_files, x_data_ids, y_data_ids, col_ids, row_ids, column_names, title=None, dtype=int,
                     log_scale=True):
    """ Combines a set of linear plots within one shared visualization. """

    def _filter_data(raw_data):
        """ Filters plotable data. """
        # Same functionality as for non-facet plots
        legal_inventory = digits + '.'
        filtered_data = list()
        for data_point in raw_data:
            skip = False
            for symbol in list(str(data_point)):
                if symbol not in legal_inventory:
                    skip = True
            if not skip:
                filtered_data.append(dtype(data_point))
        return filtered_data

    # Combine data from multiple sources
    source_dict = {column_names[i]: list() for i in range(len(column_names))}
    assert (len(source_files) == len(x_data_ids) == len(y_data_ids) == len(col_ids) == len(row_ids)), \
        'Input lists should be of the same length.'
    for i in range(len(source_files)):
        # Read in features to be visualized from each source file
        df_features = pd.read_table(source_files[i], header=None, skip_blank_lines=True)
        x_features = _filter_data(df_features.iloc[:, x_data_ids[i]].values)
        y_features = _filter_data(df_features.iloc[:, y_data_ids[i]].values)
        # Add features to the joint dictionary and denote their source
        source_dict[column_names[0]] += x_features
        source_dict[column_names[1]] += y_features
        source_dict[column_names[2]] += [col_ids[i]] * len(x_features)
        source_dict[column_names[3]] += [row_ids[i]] * len(x_features)
    # Convert values into array-like form for dataframe creation
    for key in source_dict.keys():
        source_dict[key] = np.array(source_dict[key])
    # Compile dataframe
    features = pd.DataFrame.from_dict(source_dict)

    # Set plot parameters
    sns.set_style('whitegrid')
    sns.set_context('paper')

    # Make facet plot
    fgrid = sns.FacetGrid(features, col=column_names[2], row=column_names[3], size=3.5)
    # Optionally use log axes
    if log_scale:
        ax = fgrid.axes[0][0]
        ax.set_xscale('log')
        ax.set_yscale('log')
    fgrid.map(sns.regplot, column_names[0], column_names[1], fit_reg=False)
    plt.subplots_adjust(top=0.9)
    if title is not None:
        fgrid.fig.suptitle(title)
    sns.despine()
    plt.show()


def plot_grid_bar(source_files, col_ids, column_ids, column_names, normalize, freq_bound, grouped, title=None,
                  dtype=int):
    """ Combines a set of bar plots within one shared visualization. """

    def _filter_data(raw_data, numerical):
        """ Filters plotable data. """
        # Same functionality as for non-facet plots
        legal_count_inventory = digits + '.'
        legal_entry_inventory = ascii_uppercase + '$'
        filtered_data = list()
        for data_point in raw_data:
            skip = False
            for symbol in list(str(data_point)):
                if symbol not in legal_count_inventory and symbol not in legal_entry_inventory:
                    skip = True
            if not skip:
                if numerical:
                    filtered_data.append(dtype(data_point))
                else:
                    filtered_data.append(data_point)
        if numerical and normalize:
            filtered_data = filtered_data / np.sum(filtered_data)
        return np.array(filtered_data)

    # Combine data from multiple sources
    source_dict = dict()
    # Read in data and sort alphanumeric features (e.g. POS tags) alphabetically
    df_features_a = pd.read_table(source_files[0], header=None, names=['Tag', 'Count'], skip_blank_lines=True)
    df_features_a = df_features_a.sort_values('Tag', ascending=True)
    df_features_b = pd.read_table(source_files[1], header=None, names=['Tag', 'Count'], skip_blank_lines=True)
    df_features_b = df_features_b.sort_values('Tag', ascending=True)
    # Isolate columns to be plotted
    entries = _filter_data(df_features_a.iloc[:, column_ids[0]].values, False)
    counts_a = _filter_data(df_features_a.iloc[:, column_ids[1]].values, True)
    counts_b = _filter_data(df_features_b.iloc[:, column_ids[1]].values, True)
    # Construct dataframe to be visualized
    source_dict[column_names[0]] = entries
    source_dict[column_names[1]] = counts_a
    source_dict['temp'] = counts_b
    # Generate frequency mask to exclude low-frequency features from the plot
    frequency_mask = np.array(
        [int(counts_a[i] >= freq_bound or counts_b[i] >= freq_bound) for i in range(counts_a.shape[0])])
    source_dict['frequency_mask'] = frequency_mask
    features = pd.DataFrame.from_dict(source_dict)
    # Sort by count value and apply frequency mask
    features = features.sort_values(column_names[0], ascending=True)
    if freq_bound > 0:
        features = features.drop(features[features.frequency_mask == 0].index)
    # Restructure dataframe to be compatible with a factor-plot
    features_a = features
    features_b = features_a.copy(deep=True)
    features_a = features_a.drop(['temp', 'frequency_mask'], axis=1)
    corpus_col_a = np.array([col_ids[0]] * len(features_a))
    features_a[column_names[2]] = corpus_col_a
    features_b = features_b.drop([column_names[1], 'frequency_mask'], axis=1)
    features_b.columns = [column_names[0], column_names[1]]
    corpus_col_b = np.array([col_ids[1]] * len(features_b))
    features_b[column_names[2]] = corpus_col_b
    features = features_a.append(features_b)

    # Set plot parameters
    sns.set_style('whitegrid')
    sns.set_context('paper')

    # Make factor-plot
    if grouped:
        fgrid = sns.factorplot(x=column_names[1], y=column_names[0], hue=column_names[2], data=features, kind='bar',
                               palette='RdBu_r', size=4.5, aspect=2)
    else:
        fgrid = sns.FacetGrid(features, col=column_names[2], size=4.5)
        fgrid.map(sns.barplot, column_names[1], column_names[0], palette='Set2')
    if title is not None:
        plt.subplots_adjust(top=0.85)
        fgrid.fig.suptitle(title)
    sns.despine()
    plt.show()


def significance_tests(data_sources, col_ids, equal_var, test_id='all', dtype=float, count_values=False):
    """ Performs the significance (i.e. hypothesis) tests between specified data sets. """

    def _filter_data(raw_data):
        """ Filters plotable data. """
        # Same functionality as within the visualization functions
        legal_inventory = digits + '.'
        filtered_data = list()
        for data_point in raw_data:
            skip = False
            for symbol in list(str(data_point)):
                if symbol not in legal_inventory:
                    skip = True
            if not skip:
                filtered_data.append(dtype(data_point))
        return filtered_data

    # Read in source files
    data_arrays = list()
    for i in range(len(data_sources)):
        df_features = pd.read_table(data_sources[i], header=None, skip_blank_lines=True)
        # Designate data to be analyzed
        if count_values:
            # Optionally transform read-in data into a format better suited for hypothesis testing
            entries = _filter_data(df_features.iloc[:, col_ids[i]].values)
            counts = _filter_data(df_features.iloc[:, col_ids[i] + 1].values)
            data_array = list()
            for j in range(len(entries)):
                data_array += [entries[j]] * counts[j]
        else:
            data_array = _filter_data(df_features.iloc[:, col_ids[i]].values)
        data_arrays.append(np.array(data_array))

    sample_a = data_arrays[0]
    sample_b = data_arrays[1]

    # Define significance test functions for normally distributed and skewed distributions
    def _t_test(_sample_a, _sample_b):
        """ Computes the independent t-test statistic. """
        res = stats.ttest_ind(_sample_a, _sample_b, axis=0, equal_var=equal_var, nan_policy='propagate')
        print('Independent t-test\nt-statistic: {}\np-value: {}'.format(res[0], res[1]))
        print('-' * 10)

    def _mann_whitney(_sample_a, _sample_b):
        """ Computes the Mann-Whitney rank test statistic. """
        res = stats.mannwhitneyu(_sample_a, _sample_b, use_continuity=True)
        print('Mann-Whitney rank test\nU-statistic: {}\np-value: {}'.format(res[0], res[1]))
        print('-' * 10)

    def _wilcoxon(_sample_a, _sample_b):
        """ Computes the Wilcoxon rank-sum statistic. """
        res = stats.ranksums(_sample_a, _sample_b)
        print('Wilcoxon rank-sum\nstatistic: {}\np-value: {}'.format(res[0], res[1]))
        print('-' * 10)

    def _effect_size(_sample_a, _sample_b):
        """ Computes Cohen's d effect size for the difference between the means of two distributions. """
        mean_a = np.mean(_sample_a)
        mean_b = np.mean(_sample_b)
        std_a = np.std(_sample_a)
        std_b = np.std(_sample_b)
        std_pooled = np.sqrt((np.square(std_a) + np.square(std_b)) / 2)
        print('Effect size: {}'.format((mean_a - mean_b) / std_pooled))

    # Run specified test(s) on the supplied data
    print('=' * 10)
    if test_id == 't_test' or test_id == 'all':
        _t_test(sample_a, sample_b)
    if test_id == 'whitney' or test_id == 'all':
        _mann_whitney(sample_a, sample_b)
    if test_id == 'wilcoxon' or test_id == 'all':
        _wilcoxon(sample_a, sample_b)
    _effect_size(sample_a, sample_b)

    print('=' * 10)
    print('Evaluation completed')


# ================== #
# Perform evaluation #
# ================== #

# Define evaluation options
class EvalOpt(object):
    """ Mini-options for corpus evaluation."""

    def __init__(self):
        self.root_dir = os.getcwd()
        self.freq_bound = 0


# Declare input variables and locations
eval_opt = EvalOpt()
low_id_path = os.path.join(eval_opt.root_dir, '../data/europarl/europarl_v7_low.txt')
high_id_path = os.path.join(eval_opt.root_dir, '../data/europarl/europarl_v7_high.txt')
id_annotated_path = os.path.join(eval_opt.root_dir, '../data/europarl/europarl_annotated.txt')
low_id_name = 'europarl_low'
high_id_name = 'europarl_high'
full_name = 'europarl_full'
eval_dir = os.path.join(eval_opt.root_dir, '../data/corpus_eval/')
if not os.path.exists(eval_dir):
    os.mkdir(eval_dir)

# Perform evaluation (uncommented sections are executed upon running the script)

"""
# DATA COLLECTION
# Vocab
low_id_sents, low_id_vocab = read_file(low_id_path, low_id_name, eval_dir, eval_opt)
high_id_sents, high_id_vocab = read_file(high_id_path, high_id_name, eval_dir, eval_opt)

# Sentence lengths
get_length_stats(low_id_sents, low_id_name, eval_dir)
get_length_stats(high_id_sents, high_id_name, eval_dir)

# N-grams
get_ngrams(low_id_sents, 2, low_id_name, eval_dir)
get_ngrams(low_id_sents, 3, low_id_name, eval_dir)
get_ngrams(high_id_sents, 2, high_id_name, eval_dir)
get_ngrams(high_id_sents, 3, high_id_name, eval_dir)

# POS tags and dependency parses
get_parses(low_id_sents, low_id_name, eval_dir)
get_parses(high_id_sents, high_id_name, eval_dir)

# Vocab overlap
get_vocab_overlap(low_id_vocab, high_id_vocab, 3, low_id_name, high_id_name, eval_dir)

# N-gram overlap
low_bigrams = os.path.join(eval_dir, 'europarl_low_2-gram_counts.txt')
low_trigrams = os.path.join(eval_dir, 'europarl_low_3-gram_counts.txt')
high_bigrams = os.path.join(eval_dir, 'europarl_high_2-gram_counts.txt')
high_trigrams = os.path.join(eval_dir, 'europarl_high_3-gram_counts.txt')
get_ngram_overlap(low_bigrams, high_bigrams, 3, low_id_name, high_id_name, eval_dir)
get_ngram_overlap(low_trigrams, high_trigrams, 3, low_id_name, high_id_name, eval_dir)

# ID feature extraction
construct_annotated_corpora(id_annotated_path, low_id_path, low_id_name, eval_dir)
construct_annotated_corpora(id_annotated_path, high_id_path, high_id_name, eval_dir)
construct_annotated_corpora(id_annotated_path, None, full_name, eval_dir)

"""
"""
# VISUALIZATION
# Sources
# Vocab
low_id_vocab = os.path.join(eval_dir, 'europarl_low_raw_vocab.txt')
high_id_vocab = os.path.join(eval_dir, 'europarl_high_raw_vocab.txt')

# N-grams
low_id_bigrams = os.path.join(eval_dir, 'europarl_low_2-gram_counts.txt')
high_id_bigrams = os.path.join(eval_dir, 'europarl_high_2-gram_counts.txt')
low_id_trigrams = os.path.join(eval_dir, 'europarl_low_3-gram_counts.txt')
high_id_trigrams = os.path.join(eval_dir, 'europarl_high_3-gram_counts.txt')

# POS tags
low_id_pos_tags = os.path.join(eval_dir, 'europarl_low_tag_counts.txt')
high_id_pos_tags = os.path.join(eval_dir, 'europarl_high_tag_counts.txt')

# Sentence lengths
low_id_sent_lens = os.path.join(eval_dir, 'europarl_low_sentence_lengths.txt')
high_id_sent_lens = os.path.join(eval_dir, 'europarl_high_sentence_lengths.txt')

# Parse stats
low_id_parse_stats = os.path.join(eval_dir, 'europarl_low_parse_stats.txt')
high_id_parse_stats = os.path.join(eval_dir, 'europarl_high_parse_stats.txt')

# Surprisal
low_id_features = os.path.join(eval_dir, 'europarl_low_ID_features.txt')
high_id_features = os.path.join(eval_dir, 'europarl_high_ID_features.txt')
full_id_features = os.path.join(eval_dir, 'europarl_full_ID_features.txt')
"""
"""
# Plotting
# Facet-/ Grid-plots
# N-grams
plot_grid_linear([low_id_vocab, high_id_vocab, low_id_bigrams, high_id_bigrams, low_id_trigrams, high_id_trigrams],
                 [0, 0, 0, 0, 0, 0], [2, 2, 2, 2, 2, 2],
                 ['1-grams', '1-grams', '2-grams', '2-grams', '3-grams', '3-grams'],
                 ['low-ID', 'high-ID', 'low-ID', 'high-ID', 'low-ID', 'high-ID'],
                 ['N-gram rank', 'N-gram frequency', 'Arity', 'Corpus'],
                 'N-gram distributions in the ID-variant corpora')

# POS tags
plot_grid_bar([low_id_pos_tags, high_id_pos_tags], ['low-ID', 'high-ID'], [0, 1], 
              ['PTB tag', 'Corpus fraction', 'Corpus'], True, 0.01, False, 'POS tags in the induced ID-variant corpora')
plot_grid_bar([low_id_pos_tags, high_id_pos_tags], ['low-ID', 'high-ID'], [0, 1],
              ['PTB tag', 'Corpus fraction', 'Corpus'], True, 0.01, True, 'POS tags in the induced ID-variant corpora')

# Sentence lengths
plot_grid_dist([low_id_sent_lens, high_id_sent_lens],
               [0, 0], ['low-ID', 'high-ID'], ['Sentence length', 'Corpus'],
               'Sentence lengths in the induced ID-variant corpora', count_values=True, dtype=int)

# Dependency parses and DLT costs
plot_grid_dist([low_id_parse_stats, high_id_parse_stats],
               [0, 0], ['low-ID', 'high-ID'], ['Mean dependency arc length per sentence', 'Corpus'],
               'Dependency arc lengths in the induced ID-variant corpora')
plot_grid_dist([low_id_parse_stats, high_id_parse_stats],
               [1, 1], ['low-ID', 'high-ID'], ['Mean DLT integration cost per sentence', 'Corpus'],
               'DLT integration costs in the induced ID-variant corpora')

# ID features
plot_grid_dist([low_id_features, high_id_features],
               [0, 0], ['low-ID', 'high-ID'], ['Mean word surprisal per sentence', 'Corpus'],
               'Normalized surprisal scores in the induced ID-variant corpora')
plot_grid_dist([low_id_features, high_id_features],
               [1, 1], ['low-ID', 'high-ID'], ['Mean word UID divergence per sentence', 'Corpus'],
               'Normalized UID divergence scores in the induced ID-variant corpora')
"""
"""
# Plotting
# Individual plots
# N-grams
plot_linear(low_id_vocab, [0, 2], True, '1-gram rank', '1-gram frequency',
            '1-grams in the induced low-ID corpus')
plot_linear(high_id_vocab, [0, 2], True, '1-gram rank', '1-gram frequency',
            '1-grams in the induced high-ID corpus')
plot_linear(low_id_bigrams, [0, 2], True, '2-gram rank', '2-gram frequency',
            '2-grams in the induced low-ID corpus')
plot_linear(high_id_bigrams, [0, 2], True, '2-gram rank', '2-gram frequency',
            '2-grams in the induced high-ID corpus')
plot_linear(low_id_trigrams, [0, 2], True, '3-gram rank', '3-gram frequency',
            '3-grams in the induced low-ID corpus')
plot_linear(high_id_trigrams, [0, 2], True, '3-gram rank', '3-gram frequency',
            '3-grams in the induced high-ID corpus')

# POS tags
plot_bar([low_id_pos_tags, high_id_pos_tags], [0, 1], ['PTB tag', 'Corpus fraction'], True, True, False, 0.01,
         'POS tags in the induced low-ID corpus')
plot_bar([high_id_pos_tags, low_id_pos_tags], [0, 1], ['PTB tag', 'Corpus fraction'], True, True, False, 0.01,
         'POS tags in the induced high-ID corpus')
plot_bar([low_id_pos_tags, high_id_pos_tags], [0, 1], ['PTB tag', 'Corpus fraction'], True, True, True, 0.01,
         'POS tag fraction divergence in low-ID corpus relative to high-ID corpus')

# Sentence lengths
plot_dist(low_id_sent_lens, [0, 1], 'Sentence length', 'Corpus count',
          'Sentence lengths in the induced low-ID corpus', int)
plot_dist(high_id_sent_lens, [0, 1], 'Sentence length', 'Corpus count',
          'Sentence lengths in the induced high-ID corpus', int)

# Dependency parses and DLT costs
plot_dist(low_id_parse_stats, [0], 'Mean dependency arc length per sentence', 'Corpus count',
          'Dependency arc lengths in the induced low-ID corpus')
plot_dist(high_id_parse_stats, [0], 'Mean dependency arc length per sentence', 'Corpus count',
          'Dependency arc lengths in the induced high-ID corpus')
plot_dist(low_id_parse_stats, [1], 'Mean DLT integration cost per sentence', 'Corpus count',
          'DLT integration costs in the induced low-ID corpus')
plot_dist(high_id_parse_stats, [1], 'Mean DLT integration cost per sentence', 'Corpus count',
          'DLT integration costs in the induced high-ID corpus')

# ID features

plot_dist(low_id_features, [0], 'Mean word surprisal per sentence', 'Corpus count',
          'Normalized surprisal scores in the induced low-ID corpus')
plot_dist(high_id_features, [0], 'Mean word surprisal per sentence', 'Corpus count',
          'Normalized surprisal scores in the induced high-ID corpus')
plot_dist(full_id_features, [0], 'Mean word surprisal per sentence', 'Corpus count',
          'Normalized surprisal scores in the 90k-Europarl corpus')

plot_dist(low_id_features, [1], 'Mean word UID divergence per sentence', 'Corpus count',
          'UID divergence scores in the induced low-ID corpus')
plot_dist(high_id_features, [1], 'Mean word UID divergence per sentence', 'Corpus count',
          'UID divergence scores in the induced high-ID corpus')
plot_dist(full_id_features, [1], 'Mean word UID divergence per sentence', 'Corpus count',
          'UID divergence scores in the 90k-Europarl corpus')
"""
"""
# SIGNIFICANCE TESTING
# Sentence lengths
print('Running significance tests for sentence lengths ...')
significance_tests([low_id_sent_lens, high_id_sent_lens], [0, 0], True,
                   test_id='all', dtype=int, count_values=True)

# Dependency arc lengths
print('Running significance tests for dependency arc lengths ...')
significance_tests([low_id_parse_stats, high_id_parse_stats], [0, 0], True, test_id='all')

# DLT costs
print('Running significance tests for DLT integration costs ...')
significance_tests([low_id_parse_stats, high_id_parse_stats], [1, 1], True, test_id='all')

# Surprisal
print('Running significance tests for average sentence surprisal ...')
significance_tests([low_id_features, high_id_features], [0, 0], False, test_id='all')

# UID divergence
print('Running significance tests for average sentence UID divergence ...')
significance_tests([low_id_features, high_id_features], [1, 1], True, test_id='all')
"""
