from utils import parse_args, create_experiment_dirs, calculate_flops
from train import Train
from summarizer import Summarizer
import tensorflow as tf


def main():
    # Parse the JSON arguments
    try:
        config_args = parse_args()
    except:
        print("Add a config file using \'--config file_name.json\'")
        exit(1)

    # Create the experiment directories
    _, config_args.summary_dir, config_args.checkpoint_dir = create_experiment_dirs(config_args.experiment_dir)

    # Reset the default Tensorflow graph
    tf.reset_default_graph()

    # Tensorflow specific configuration
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # Data loading
    # The batch size is equal to 1 when testing to simulate the real experiment.

    # Model creation
    print("Building the model...")
    print("Model is built successfully\n\n")

    # Summarizer creation
    summarizer = Summarizer(sess, config_args.summary_dir)
    # Train class
    # trainer = Train(sess, model, data, summarizer)

    if config_args.train_or_test == 'train':
        try:
            print("FLOPs for batch size = " + str(config_args.batch_size) + "\n")
            calculate_flops()
            print("Training...")
            # trainer.train()
            print("Training Finished\n\n")
        except KeyboardInterrupt:
            # trainer.save_model()
            pass

    elif config_args.train_or_test == 'test':
        print("FLOPs for single inference \n")
        calculate_flops()
        # This can be 'val' or 'test' or even 'train' according to the needs.
        print("Testing...")
        # trainer.test('val')
        print("Testing Finished\n\n")

    else:
        raise ValueError("Train or Test options only are allowed")


if __name__ == '__main__':
    main()
