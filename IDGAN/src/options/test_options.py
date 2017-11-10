from IDGAN.src.options.train_options import TrainOptions
# from IDGAN.src.options.small_options import TrainOptionsSmall


class TestOptions(TrainOptions):
    """ Default testing options for the IDGAN system. """

    def __init__(self):
        super(TestOptions, self).__init__()
        # Data pre-processing
        self.shuffle = False

        # Encoder hyper-parameters
        self.enc_static_keep_prob = 0.5
        self.enc_rnn_keep_prob = 0.5

        # Decoder hyper-parameters
        self.dec_static_keep_prob = 0.5
        self.dec_rnn_keep_prob = 0.5
        self.length_slack = 8

        # Discriminator hyper-parameters
        self.disc_static_keep_prob = 1.0

        # Session hyper-parameters
        self.is_train = False
        self.allow_dropout = True
        self.beam_width = 3
        self.num_samples = 3
        self.batch_size = 1
