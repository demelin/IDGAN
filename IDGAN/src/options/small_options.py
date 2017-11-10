import os
from IDGAN.src.options.base_options import BaseOptions


class TrainOptionsSmall(BaseOptions):
    """ Reduced training parameters for quick model evaluation on local machines; for comprehensive comments
     refer to corresponding train_options."""

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
        self.num_buckets = 8

        # Encoder/ Generator hyper-parameters
        self.enc_static_keep_prob = 0.5
        self.enc_rnn_keep_prob = 0.5
        self.enc_attention_dims = 256

        # Decoder hyper-parameters
        self.attentive_decoding = True
        self.dec_attention_dims = 256
        self.dec_static_keep_prob = 0.5
        self.dec_rnn_keep_prob = 0.5
        self.length_slack = 0

        # Discriminator hyper-parameters
        self.disc_static_keep_prob = 0.5
        self.enable_shortcuts = False

        # GAN hyper-parameters
        self.gan_type = 'NLLGAN'
        self.adv_lambda = 1.0
        self.rec_lambda = 1.0 - self.adv_lambda
        self.id_lambda = 0.0
        self.gen_steps = 1
        self.disc_steps = 1
        self.train_dec = False
        self.cross_dec = False
        self.smooth_labels = True
        self.flip_labels = True

        # Loss and optimization hyper-parameters
        self.schedule_sampling = True
        self.samples = 25
        self.sampled_values = None
        self.remove_accidental_hits = True
        self.l2_beta = 1e-5
        self.grad_clip_norm = None
        self.gen_lr = 0.001
        self.disc_lr = 0.001

        # Session hyper-parameters
        self.train_id = '{:s}_G{:d}_D{:d}_AL{:.4f}_RL{:.4f}_TD{}_CD{}'\
            .format(self.gan_type, self.gen_steps, self.disc_steps, self.adv_lambda, self.rec_lambda, self.train_dec,
                    self.cross_dec)
        self.use_toy = True
        self.is_train = True
        self.use_reconstruction_objective = True
        self.allow_dropout = True
        self.batch_size = 16
        self.num_epochs = 50
        self.scheduling_constant = 10.6
        self.report_freq = 100
        self.summary_freq = 50
        self.save_freq = 20
        self.enable_early_stopping = True
        self.start_early_stopping = 1
        self.annealing_step = 2
        self.gen_annealing_factor = 0.9
        self.disc_annealing_factor = 0.9
        self.patience = 5

        # Target directories
        self.save_dir = os.path.join(self.save_dir, self.train_id)
        self.log_dir = os.path.join(self.log_dir, self.train_id)
        self.out_dir = os.path.join(self.out_dir, self.train_id)
        dir_list = [self.save_dir, self.log_dir, self.out_dir]
        for dir_id in dir_list:
            if not os.path.exists(dir_id):
                os.makedirs(dir_id)
