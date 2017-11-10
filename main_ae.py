""" Used to call the training / inference session functions defined for the SAEs.
The uncommented line denotes the session functionality executed upon call. """

from autoencoder.src.codebase import session

if __name__ == '__main__':
    # session.train_session()
    # session.test_session()
    session.test_to_file()
