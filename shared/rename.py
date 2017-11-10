""" Script used to inspect TensorFlow checkpoint files and rename the trained parameters contained therein.
Allows for improved compatibility when initializing models with pre-trained parameters, as naming / scoping conflicts
may prevent parameters from loading properly. Script adopted from batzner@github and extended with the inspection
functionality. """

import sys
import getopt
import tensorflow as tf

usage_str = 'python tensorflow_rename_variables.py --checkpoint_dir=path/to/dir/ ' \
            '--replace_from=substr --replace_to=substr --add_prefix=abc --dry_run --list_all'


def rename(checkpoint_dir, replace_from, replace_to, add_prefix, dry_run, list_all):
    """ Lists and optionally renames variables within the specified TensorFlow checkpoint file. """
    # Locate latest checkpoint within the specified directory
    checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
    with tf.Session() as sess:
        # Iterate over variables contained within the checkpoint file
        for var_name, _ in tf.contrib.framework.list_variables(checkpoint_dir):
            var = tf.contrib.framework.load_variable(checkpoint_dir, var_name)
            # Assign a new name to the variable, if replace_from matches a substring of current variable name
            new_name = var_name
            if None not in [replace_from, replace_to]:
                # Replace a substring
                new_name = new_name.replace(replace_from, replace_to)
            if add_prefix:
                # Or just add a prefix
                new_name = add_prefix + new_name

            if dry_run:
                # Preview renaming results
                if var_name != new_name:
                    print('%s would be renamed to %s.' % (var_name, new_name))
            else:
                # Rename currently inspected variable
                if var_name != new_name:
                    print('Renaming %s to %s.' % (var_name, new_name))
                var = tf.Variable(var, name=new_name)

            if list_all:
                # Inspect the contents of the checkpoint file
                print(var_name)

        if not dry_run:
            # Save any changes done to variable names
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            saver.save(sess, checkpoint.model_checkpoint_path)


def main(argv):
    # Declare command line args and calls
    checkpoint_dir = None
    replace_from = None
    replace_to = None
    add_prefix = None
    dry_run = False
    list_all = False

    try:
        opts, args = getopt.getopt(argv, 'h', ['help=', 'checkpoint_dir=', 'replace_from=',
                                               'replace_to=', 'add_prefix=', 'dry_run', 'list_all'])
    except getopt.GetoptError:
        print(usage_str)
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print(usage_str)
            sys.exit()
        elif opt == '--checkpoint_dir':
            checkpoint_dir = arg
        elif opt == '--replace_from':
            replace_from = arg
        elif opt == '--replace_to':
            replace_to = arg
        elif opt == '--add_prefix':
            add_prefix = arg
        elif opt == '--dry_run':
            dry_run = True
        elif opt == '--list_all':
            list_all = True

    if not checkpoint_dir:
        print('Please specify a checkpoint_dir. Usage:')
        print(usage_str)
        sys.exit(2)

    rename(checkpoint_dir, replace_from, replace_to, add_prefix, dry_run, list_all)


if __name__ == '__main__':
    main(sys.argv[1:])
