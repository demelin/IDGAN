import os
from IDGAN.src.options.base_options import BaseOptions


class TrainOptions(BaseOptions):
    """ Default training options for the IDGAN system. """

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

        # Encoder/ Generator hyper-parameters
        self.enc_static_keep_prob = 0.5
        self.enc_rnn_keep_prob = 0.5
        self.enc_attention_dims = 512

        # Decoder hyper-parameters
        self.attentive_decoding = True
        self.dec_attention_dims = 512
        self.dec_static_keep_prob = 0.5
        self.dec_rnn_keep_prob = 0.5
        self.length_slack = 0

        # Discriminator hyper-parameters
        self.disc_static_keep_prob = 0.5
        self.enable_shortcuts = False

        # GAN hyper-parameters
        # Selects the GAN objective type, choose from ['NLLGAN', 'WGAN', 'WGANGP' 'BCEGAN', 'LSGAN']
        # Only 'NLLGAN' and 'WGANGP' could be thoroughly examined for thesis submission
        self.gan_type = 'NLLGAN'
        # Weights for the partial objectives used in training the generator
        self.adv_lambda = 1.0
        self.rec_lambda = 1.0 - self.adv_lambda
        self.id_lambda = 0.0
        # GAN component network updates for a single GAN training step
        # 1:1 corresponds to synchronous updates (recommended by Goodfellow's NIPS workshop)
        self.gen_steps = 1
        self.disc_steps = 1
        # Decoder training configurations
        self.train_dec = False
        self.cross_dec = False
        # Toggles multi-task learning, choose from ['static', 'dynamic', None]
        self.multi_task = 'static'
        # 'Tricks' for stabilizing the GAN training
        self.smooth_labels = True
        self.flip_labels = True

        # Loss and optimization hyper-parameters
        self.schedule_sampling = True
        # Candidate sampling loss used in conjunction with the reconstruction objective
        self.samples = 25
        self.sampled_values = None
        self.remove_accidental_hits = True
        self.l2_beta = 1e-5
        # Disabled within the model
        self.grad_clip_norm = None
        # Generator and discriminator may be assigned different learning rates
        self.gen_lr = 0.0001
        self.disc_lr = 0.0001

        # Session hyper-parameters
        # train_id identifies the current training configuration;
        # useful when running multiple experiments that differ in the choice of hyper-parameters
        self.train_id = '{:s}_G{:d}_D{:d}_AL{:.4f}_RL{:.4f}_TD{}_CD{}_MT{}'\
            .format(self.gan_type, self.gen_steps, self.disc_steps, self.adv_lambda, self.rec_lambda, self.train_dec,
                    self.cross_dec, self.multi_task)
        self.use_toy = False
        self.is_train = True
        self.use_reconstruction_objective = True
        # Forces dropout to remain active during inference time, see github.com/soumith/ganhacks
        self.allow_dropout = True
        self.batch_size = 32
        self.num_epochs = 100
        # Scheduling constant corresponds to expected training convergence after ~50 epochs
        self.scheduling_constant = 10.6
        self.report_freq = 100  # in steps
        self.summary_freq = 50  # in steps
        self.save_freq = 20  # in epochs
        # 'Warm-up' period
        self.enable_early_stopping = True
        self.start_early_stopping = 5  # in epochs
        self.annealing_step = 5  # in epochs
        # Generator's and discriminator's lr may be reduced by unequal factors
        self.gen_annealing_factor = 0.9
        self.disc_annealing_factor = 0.9
        self.patience = 20  # in epochs

        # Create target directories for training materials and outputs
        self.save_dir = os.path.join(self.save_dir, self.train_id)
        self.log_dir = os.path.join(self.log_dir, self.train_id)
        self.out_dir = os.path.join(self.out_dir, self.train_id)
        dir_list = [self.save_dir, self.log_dir, self.out_dir]
        for dir_id in dir_list:
            if not os.path.exists(dir_id):
                os.makedirs(dir_id)
