from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.callbacks import ReduceLROnPlateau


def model_checkpoint(save_model_path):
    """
    Callbacks to save the best version of the model
    :return: ModelCheckpoint object
    """
    checkpoint = ModelCheckpoint(filepath=save_model_path,
                                 monitor='accuracy',
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=True,
                                 mode='max')
    return checkpoint


def tensorboard_callbacks(log_dir):
    """
    Tensorboard callback to monitor the training progression.
    :param log_dir: name of the directory to save the logs of training.
    :return: tensorboard object
    """
    tensorboard = TensorBoard(log_dir='logs/'+ log_dir)
    return tensorboard


def reduce_lr_on_plateau():
    """
    Training strategy to modify the learning_rate if the model struggles to reach ideal performance
    :return: tf.keras.callbacks.ReduceLROnPlateau object
    """
    reduce_lr = ReduceLROnPlateau(monitor='loss',
                                  factor=0.1,
                                  patience=3,
                                  verbose=0,
                                  mode='auto',
                                  min_delta=0.0001,
                                  cooldown=0,
                                  min_lr=0.00001)

    return reduce_lr

