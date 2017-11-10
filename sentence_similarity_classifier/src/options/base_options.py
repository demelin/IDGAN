import os


class BaseOptions(object):
    """ Base sentence similarity classifier options common to all parameter configurations. """

    def __init__(self):
        # Declare model directory structure
        cwd = os.getcwd()
        # Applies when model is accessed in PyCharm/ on a local machine
        if 'sentence_similarity_classifier' in cwd:
            self.root_dir = os.path.join(cwd, '../..')
            self.local_dir = os.path.join(cwd, '..')
        # Applies when model is accessed from a script/ on a remote cluster
        else:
            self.root_dir = cwd
            self.local_dir = os.path.join(cwd, 'sentence_similarity_classifier/src')
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
        self.hidden_dims = 50
	self.attention_dims = 50
        self.num_layers = 2
