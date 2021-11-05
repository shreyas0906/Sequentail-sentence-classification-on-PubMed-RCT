import os, glob, time
import datetime
import tensorflow as tf
import src
from argparse import ArgumentParser
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def gen_model_path():
    """
    This function generates a new directory to save the models
    :return: path to save the model.
    """
    now = datetime.datetime.now()

    if not os.path.exists(os.getcwd() + '/models'):
        os.makedirs(os.getcwd() + '/models')
    else:
        name = 'model_{}-{}-{}:{}'.format(now.day, now.month, now.hour, now.minute)
        if not os.path.exists('models/' + name):
            os.makedirs('models/' + name)

        return 'models/' + name #+ '/model_ssc.h5'


def get_latest_model_dir():
    """
    Function to return the latest model.
    This is based on the time at which the model was created.
    :return: directory of the latest model
    """
    return max([os.path.join('src/models', d) for d in os.listdir('src/models')], key=os.path.getmtime)


def check_gpu_status():
    """
    Function to check whether GPU is available to train the model
    :return:
    """
    physical_devices = tf.config.list_physical_devices('GPU')

    if physical_devices:
        for device in physical_devices:
            print(f"\nGPU is available: {device}\n")
    else:
        print("\nGPU not available\n")


def train(args):
    """
    To train the model.
    :param args: contains the training configurations
    :return:
    """
    check_gpu_status()
    save_model_path = gen_model_path()

    tensorboard_callback = src.callbacks.tensorboard_callbacks('src/logs')
    reducde_lr_on_plateau = src.callbacks.reduce_lr_on_plateau()
    model_checkpoints = src.callbacks.model_checkpoint(save_model_path)

    start = time.time()

    model = src.models.tribrid_model()
    train_dataset, validation_dataset = src.models.get_data_for_training('tribrid')
    print(model.summary())
    plot_model(model, to_file='tribrid.png', show_layer_names=True, show_shapes=True, dpi=96)
    history = model.fit(train_dataset,
                        epochs=args.epochs,
                        validation_data=validation_dataset,
                        callbacks=[tensorboard_callback, reducde_lr_on_plateau])# , model_checkpoints

    print(f"Time taken to train: {(time.time() - start)/60:.2f} mins")
    model.save(save_model_path, save_format='tf')


if __name__ == '__main__':

    p = ArgumentParser()
    p.add_argument('--epochs', required=False, type=int, default=50, help='Number of epochs to train on')
    p.add_argument('--train', required=False, type=str, default='True', help='flag to train the data')
    p.format_usage()
    args = p.parse_args()

    if args.train:
        train(args)

    # loaded_model = load_model(get_latest_model_dir())
    # print(loaded_model.summary())