import os

from autoencoder.src.options.base_options import BaseOptions


class TrainOptions(BaseOptions):
    """ Default training options for the sequence auto-encoder model. """

    def __init__(self):
        super(TrainOptions, self).__init__()
        # Data pre-processing
        self.max_sent_len = None
        self.sent_select = 'truncate'
        self.mark_borders = True
        self.freq_bound = 3
        self.shuffle = True
        self.lower = True
        self.pad = True
        self.num_buckets = 8

        # Encoder hyper-parameters
        self.enc_static_keep_prob = 0.5
        self.enc_rnn_keep_prob = 0.5
        self.attentive_encoding = False
        self.enc_attention_dims = 512

        # Decoder hyper-parameters
        self.attentive_decoding = True
        self.dec_attention_dims = 512
        self.dec_static_keep_prob = 0.5
        self.dec_rnn_keep_prob = 0.5
        # Denotes the number of words the decoder is allowed to generate beyond the length of its inputs
        self.length_slack = 0

        # Loss and optimization hyper-parameters
        self.samples = 25
        self.sampled_values = None
        self.remove_accidental_hits = True
        self.l2_beta = 1e-5
        # Disabled within the model
        self.grad_clip_norm = None
        # Higher starting rate encourages rapid initial exploration (reduced to 0.0001 for the high-ID corpus)
        self.learning_rate = 0.001

        # Session hyper-parameters
        # Identifies current training session, choose from ['source', 'target', 'toy', other]
        self.train_id = 'source'
        self.is_train = True
        self.use_reconstruction_objective = True
        # Dropout is enabled at inference time for IDGAN, as it provides an additional source of noise
        self.allow_dropout = True
        self.use_candidate_sampling = True
        self.is_local = False
        self.batch_size = 64
        self.num_epochs = 200
        # Scheduling constant corresponds to expected training convergence after ~100 epochs
        self.scheduling_constant = 17.48
        self.report_freq = 100  # in steps
        self.summary_freq = 50  # in steps
        self.save_freq = None
        self.enable_early_stopping = False
        # 'Warm-up phase'
        self.start_early_stopping = 10  # in epochs
        self.annealing_step = 5  # in epochs
        self.annealing_factor = 0.9
        self.patience = 30  # in epochs

        # Create target directories for training materials and outputs
        self.save_dir = os.path.join(self.save_dir, self.train_id)
        self.log_dir = os.path.join(self.log_dir, self.train_id)
        self.out_dir = os.path.join(self.out_dir, self.train_id)
        dir_list = [self.save_dir, self.log_dir, self.out_dir]
        for dir_id in dir_list:
            if not os.path.exists(dir_id):
                os.makedirs(dir_id)
