""" A set of functions for evaluating the training performance of IDGAN by visualizing the trajectories of various
metrics tracked throughout the training process. """

import os
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def pickle_to_db(pickle_file, features_string, features='adv', max_epoch=None):
    """ Translates a training pickle into a plot-able padas dataframe. """
    with open(pickle_file, 'rb') as out_file:
        valid_dict = pickle.load(out_file)

    # Optionally restrict the visualized training/ validation epochs to obtain more comparable sub-plots
    if max_epoch is not None:
        for key in valid_dict.keys():
            if len(valid_dict[key]) > max_epoch:
                valid_dict[key] = valid_dict[key][:max_epoch]

    # Isolate only the adversarial losses of the generator and discriminator networks
    if features == 'adv':
        plot_dict = dict()
        plot_dict['Loss'] = valid_dict['gen_valid_losses'] + valid_dict['disc_valid_losses']
        plot_dict['Network'] = \
            ['Generator'] * len(valid_dict['gen_valid_losses']) + \
            ['Discriminator'] * len(valid_dict['disc_valid_losses'])
        plot_dict['Epochs'] = [j + 1 for j in range(len(valid_dict['gen_valid_losses']))] * 2
        plot_dict['Condition'] = [features_string] * len(plot_dict['Loss'])
        df = pd.DataFrame.from_dict(plot_dict)
        return df

    # Isolate only the achieved information density reduction
    elif features == 'id_only':
        plot_dict = dict()
        plot_dict['ID reduction'] = valid_dict['id_reduction_scores']
        plot_dict['Epochs'] = [j + 1 for j in range(len(plot_dict['ID reduction']))]
        plot_dict['Condition'] = [features_string] * len(plot_dict['ID reduction'])
        df = pd.DataFrame.from_dict(plot_dict)
        return df

    # Isolate the full generator and discriminator losses as well as the ID reduction
    elif features == 'full':
        plot_dict = dict()
        plot_dict['Loss'] = \
            valid_dict['gen_valid_losses'] + valid_dict['disc_valid_losses'] + valid_dict['id_reduction_scores']
        plot_dict['Value'] = \
            ['Generator loss'] * len(valid_dict['gen_adv_losses']) + \
            ['Discriminator loss'] * len(valid_dict['gen_valid_losses']) + \
            ['ID reduction'] * len(valid_dict['id_reduction_scores'])
        plot_dict['Epochs'] = [j + 1 for j in range(len(valid_dict['gen_adv_losses']))] * 3
        plot_dict['Condition'] = [features_string] * len(plot_dict['Loss'])
        df = pd.DataFrame.from_dict(plot_dict)
        return df


def plot_curves(df, plotted_features='adv', title=None):
    """ Produces a factor-plot on the basis of the provided dataframe. """

    # Set plot parameters
    sns.set_style('whitegrid')
    sns.set_context('paper')

    # Sanity check: Examine the visualized dataframe
    print(df)

    # Produce plots
    if plotted_features == 'adv':
        g = sns.FacetGrid(df, col='Condition', hue='Network', palette='Set2', col_wrap=2, size=1.8, aspect=2.0)
        g.map(plt.plot, 'Epochs', 'Loss', marker='').add_legend()
    elif plotted_features == 'id_only':
        g = sns.FacetGrid(df, col='Condition', palette='Set3', col_wrap=3, size=2.0, aspect=2.0)
        g.map(plt.plot, 'Epochs', 'ID reduction', marker='').add_legend()
    elif plotted_features == 'full':
        g = sns.FacetGrid(df, col='Condition', hue='Value', palette='Set1', col_wrap=2, size=2.0, aspect=2.0)
        g.map(plt.plot, 'Epochs', 'Loss', marker='').add_legend()

    if title is not None:
        plt.subplots_adjust(top=0.85)
        g.fig.suptitle(title)

    sns.despine()
    plt.show()


# ========== #
# Make plots #
# ========== #

# Specify paths to pickle files containing the training/ validation logs
local_dir = os.getcwd()
nll_path = os.path.join(local_dir, '../data/nll_pickles')
wgangp_path = os.path.join(local_dir, '../data/wgan_pickles')
# Collect visualized dataframes within a sheared list
dfs = list()

# Declare files and plot features for the NLLGAN experiments
nll_files = ['G1_D1_AL1.0_RL0.0_TDFalse_CDFalse/valid_pickle_archive.pkl',
             'G2_D1_AL1.0_RL0.0_TDFalse_CDFalse/valid_pickle_archive.pkl',
             'G2_D1_AL1.0_RL0.0_TDFalse_CDTrue/valid_pickle_archive.pkl',
             'G2_D1_AL0.9_RL0.1_TDFalse_CDTrue/valid_pickle_archive.pkl',
             'G2_D1_AL0.9_RL0.1_TDTrue_CDTrue/valid_pickle_archive.pkl',
             'G2_D1_AL0.8_RL0.2_TDFalse_CDFalse/valid_pickle_archive.pkl',
             'G2_D1_AL0.8_RL0.2_TDFalse_CDTrue/valid_pickle_archive.pkl',
             'G2_D1_AL0.7_RL0.3_TDFalse_CDTrue/valid_pickle_archive.pkl',
             ]
nll_features = ['G:1|D:1|A:1.0|R:0.0',
                'G:2|D:1|A:1.0|R:0.0',
                'G:2|D:1|A:1.0|R:0.0|CrossD',
                'G:2|D:1|A:0.9|R:0.1|CrossD',
                'G:2|D:1|A:0.9|R:0.1|CrossD|TrainD',
                'G:2|D:1|A:0.8|R:0.2',
                'G:2|D:1|A:0.8|R:0.2|CrossD',
                'G:2|D:1|A:0.7|R:0.3|CrossD'
                ]

# Declare files and plot features for the WGAN-GP experiments
wgangp_files = ['G1_D4_AL1.0_RL0.0_TDFalse_CDFalse/valid_pickle_archive.pkl',
                'G1_D4_AL0.9_RL0.1_TDFalse_CDFalse/valid_pickle_archive.pkl',
                'G1_D4_AL0.8_RL0.2_TDFalse_CDFalse/valid_pickle_archive.pkl',
                'G1_D4_AL0.7_RL0.3_TDFalse_CDFalse/valid_pickle_archive.pkl',
                'G1_D5_AL0.9_RL0.1_TDFalse_CDFalse/valid_pickle_archive.pkl',
                'G1_D5_AL0.8_RL0.2_TDFalse_CDFalse/valid_pickle_archive.pkl',
                'G1_D5_AL0.7_RL0.3_TDFalse_CDFalse/valid_pickle_archive.pkl',
                'G1_D5_AL0.5_RL0.5_TDFalse_CDFalse/valid_pickle_archive.pkl'
                ]

wgangp_features = ['G:1|D:4|A:1.0|R:0.0',
                   'G:1|D:4|A:0.9|R:0.1',
                   'G:1|D:4|A:0.8|R:0.2',
                   'G:1|D:4|A:0.7|R:0.3',
                   'G:1|D:5|A:0.9|R:0.1',
                   'G:1|D:5|A:0.8|R:0.2',
                   'G:1|D:5|A:0.7|R:0.3',
                   'G:1|D:5|A:0.5|R:0.5'
                   ]

# Generate individual dataframes
pf = 'full'
for i in range(len(nll_files)):
    dfs.append(pickle_to_db(os.path.join(nll_path, nll_files[i]), nll_features[i], features=pf))

# Merge and plot the dataframe combining all relevant data for the specified set of experiments (i.e. NLL/ WGAN-GP)
joint_df = pd.concat(dfs)
plot_curves(joint_df, pf, None)
