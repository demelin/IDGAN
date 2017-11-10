import os

from autoencoder.src.options.base_options import BaseOptions


class TrainOptionsSmall(BaseOptions):
    """ Reduced training parameters for quick model evaluation on local machines; for comprehensive comments
     refer to corresponding train_options. """

    def __init__(self):
        super(TrainOptionsSmall, self).__init__()
        # Data pre-processing
        self.max_sent_len = None
        self.sent_select = 'truncate'
        self.mark_borders = True
        self.freq_bound = 3
        self.shuffle = True
        self.lower = True
        self.pad = True
        self.num_buckets = 6

        # Encoder hyper-parameters
        self.enc_static_keep_prob = 0.6
        self.enc_rnn_keep_prob = 0.6
        self.attentive_encoding = False
        self.enc_attention_dims = 256

        # Decoder hyper-parameters
        self.dec_static_keep_prob = 0.6
        self.dec_rnn_keep_prob = 0.6
        self.attentive_decoding = True
        self.dec_attention_dims = 256
        self.length_slack = 0

        # Loss and optimization hyper-parameters
        self.samples = 25
        self.sampled_values = None
        self.remove_accidental_hits = True
        self.l2_beta = 1e-5
        self.grad_clip_norm = 24.0
        self.learning_rate = 0.001

        # Session hyper-parameters
        self.train_id = 'toy'
        self.is_train = True
        self.use_reconstruction_objective = True
        self.allow_dropout = True
        self.use_candidate_sampling = True
        self.is_local = True
        self.batch_size = 32
        self.num_epochs = 100
        self.scheduling_constant = 17.48
        self.report_freq = 10
        self.summary_freq = 5
        self.save_freq = None
        self.enable_early_stopping = True
        self.start_early_stopping = 2
        self.annealing_step = 2
        self.annealing_factor = 0.9
        self.patience = 20

        # Target directories
        self.save_dir = os.path.join(self.save_dir, self.train_id)
        self.log_dir = os.path.join(self.log_dir, self.train_id)
        self.out_dir = os.path.join(self.out_dir, self.train_id)
        dir_list = [self.save_dir, self.log_dir, self.out_dir]
        for dir_id in dir_list:
            if not os.path.exists(dir_id):
                os.makedirs(dir_id)
