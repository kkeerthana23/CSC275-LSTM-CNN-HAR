from attention_layer import AttenLayer
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sn

lstm_units = 200
attention_units = 400
downsample = 2
window_length = 1000
threshold = 0.6
step = 200
labels = ["bed", "fall", "pickup", "run", "sitdown", "standup", "walk"]


def build_att_bilstm_cnn_model():
    if downsample > 1:
        length = len(np.ones((window_length,))[::downsample])
        x_in = tf.keras.Input(shape=(length, 90))
    else:
        x_in = tf.keras.Input(shape=(window_length, 90))
    x_tensor = tf.keras.layers.Conv1D(64, 5, activation="relu")(x_in)
    x_tensor = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=lstm_units, return_sequences=True))(x_tensor)
    x_tensor = AttenLayer(attention_units)(x_tensor)
    pred = tf.keras.layers.Dense(len(labels), activation='softmax')(x_tensor)
    model = tf.keras.Model(inputs=x_in, outputs=pred)
    return model


def train_valid_split(numpy_tuple, train_portion=0.9, seed=379):
    """
    Returns Train and Valid Datset with the format of (x_train, y_train, x_valid, y_valid),
    where x_train and y_train are shuffled randomly.

    Args:
        numpy_tuple  : tuple of numpy array: (x_bed, x_fall, x_pickup, x_run, x_sitdown, x_standup, x_walk)
        train_portion: float, range (0,1)
        seed         : random seed
    """
    np.random.seed(seed=seed)
    x_train = []
    x_valid = []
    y_valid = []
    y_train = []

    for i, x_arr in enumerate(numpy_tuple):
        index = np.random.permutation([i for i in range(x_arr.shape[0])])
        split_len = int(train_portion * x_arr.shape[0])
        x_train.append(x_arr[index[:split_len], ...])
        tmpy = np.zeros((split_len, 7))
        tmpy[:, i] = 1
        y_train.append(tmpy)
        x_valid.append(x_arr[index[split_len:], ...])
        tmpy = np.zeros((x_arr.shape[0] - split_len, 7))
        tmpy[:, i] = 1
        y_valid.append(tmpy)

    x_train = np.concatenate(x_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    x_valid = np.concatenate(x_valid, axis=0)
    y_valid = np.concatenate(y_valid, axis=0)

    index = np.random.permutation([i for i in range(x_train.shape[0])])
    x_train = x_train[index, ...]
    y_train = y_train[index, ...]
    return x_train, y_train, x_valid, y_valid

def load_model(hdf5path):
    """
    Returns the Tensorflow Model for AttenLayer
    Args:
        hdf5path: str, the model file path
    """
    model = tf.keras.models.load_model(hdf5path, custom_objects={'AttenLayer':AttenLayer})
    return model