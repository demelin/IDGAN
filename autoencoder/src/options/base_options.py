import os


class BaseOptions(object):
    """ Base auto-encoder options common to all parameter configurations. """

    def __init__(self):
        # Declare model directory structure
        cwd = os.getcwd()
        # Applies when model is accessed in PyCharm/ on a local machine
        if 'autoencoder' in cwd:
            self.root_dir = os.path.join(cwd, '../..')
            self.local_dir = os.path.join(cwd, '..')
        # Applies when model is accessed from a script/ on a remote cluster
        else:
            self.root_dir = cwd
            self.local_dir = os.path.join(cwd, 'autoencoder/src')
        self.data_dir = os.path.join(self.root_dir, 'data')
        self.save_dir = os.path.join(self.local_dir, 'checkpoints')
        self.log_dir = os.path.join(self.local_dir, 'logs')
        self.out_dir = os.path.join(self.local_dir, 'out')

        # Create specified directories if they do not already exist
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Model size
        self.embedding_dims = 256
        self.enc_hidden_dims = 256
        self.dec_hidden_dims = 256
        self.enc_num_layers = 2
        self.dec_num_layers = 2
