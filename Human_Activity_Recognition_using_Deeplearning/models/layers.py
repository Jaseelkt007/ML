import tensorflow as tf
from tensorflow.keras.regularizers import l2


def lstm_block(inputs, lstm_units, dropout_rate):

    # LSTM Layer
    out = tf.keras.layers.LSTM(lstm_units, return_sequences = True, kernel_regularizer = l2(1e-4))(inputs)
    out = tf.keras.layers.Dropout(dropout_rate)(out)

    return out

def gru_block(inputs, gru_units, dropout_rate):

    # gru Layer
    out = tf.keras.layers.GRU(gru_units, return_sequences = True, kernel_regularizer = l2(1e-4))(inputs)
    out = tf.keras.layers.Dropout(dropout_rate)(out)

    return out
