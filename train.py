import tensorflow as tf
from tqdm import tqdm
import numpy as np


class Train:
    """Trainer class for the CNN.
    It's also responsible for loading/saving the model checkpoints from/to experiments/experiment_name/checkpoint_dir"""

    def __init__(self, sess, model, data, summarizer):
        self.sess = sess
        self.model = model
        self.args = self.model.args
        self.saver = tf.train.Saver(max_to_keep=self.args.max_to_keep,
                                    keep_checkpoint_every_n_hours=10,
                                    save_relative_paths=True)
        # Summarizer references
        self.data = data
        self.summarizer = summarizer

        # Initializing the model
        self.init = None
        self.__init_model()

        # Loading the model checkpoint if exists
        self.__load_model()

    ############################################################################################################
    # Model related methods
    def __init_model(self):
        print("Initializing the model...")
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init)
        print("Model initialized\n\n")

    def save_model(self):
        """
        Save Model Checkpoint
        :return:
        """
        print("Saving a checkpoint")
        self.saver.save(self.sess, self.args.checkpoint_dir, self.model.global_step_tensor)
        print("Checkpoint Saved\n\n")

    def __load_model(self):
        latest_checkpoint = tf.train.latest_checkpoint(self.args.checkpoint_dir)
        if latest_checkpoint:
            print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver.restore(self.sess, latest_checkpoint)
            print("Checkpoint loaded\n\n")
        else:
            print("First time to train!\n\n")

    ############################################################################################################
    # Train and Test methods
    def train(self):
        pass

    def test(self, test_type='val'):
        pass
