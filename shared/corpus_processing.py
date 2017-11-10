""" Applies mild pre-processing to the Europarl v7 monolingual English corpus prior to commencing the
data construction phase of IDGAN training. """

import os
from shared.util import clean_europarl

data_dir = os.path.join(os.getcwd(), '../data/europarl')

# Clean Europarl
source_path = os.path.join(data_dir, 'europarl_v7_en.txt')
clean_path = os.path.join(data_dir, 'europarl_clean.txt')
if not os.path.exists(clean_path):
    clean_europarl(source_path, clean_path, keep_above=2)
