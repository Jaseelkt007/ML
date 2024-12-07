import gin
import tensorflow as tf
from tensorflow.keras.regularizers import l2


@gin.configurable
def vgg_block(inputs, filters, kernel_size):
    """A single VGG block consisting of two convolutional layers, followed by a max-pooling layer.
    Parameters:
        inputs (Tensor): input of the VGG block
        filters (int): number of filters used for the convolutional layers
        kernel_size (tuple: 2): kernel size used for the convolutional layers, e.g. (3, 3)
    Returns:
        (Tensor): output of the VGG block
    """

    out = tf.keras.layers.Conv2D(filters, kernel_size, padding='same', activation = 'relu', kernel_regularizer=l2(1e-4))(inputs)
    out = tf.keras.layers.MaxPool2D((2, 2))(out)

    out = tf.keras.layers.Conv2D(filters, kernel_size, padding='same', activation = 'relu', kernel_regularizer=l2(1e-4))(out)
    out = tf.keras.layers.MaxPool2D((2, 2))(out)

    return out

@gin.configurable
def mobilenet_block(inputs, filters, strides = 1):

    out = tf.keras.layers.DepthwiseConv2D(
        kernel_size = (3, 3),
        strides = strides,
        padding = 'same',
        use_bias = False
     )(inputs)

    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.ReLU()(out)

    out = tf.keras.layers.Conv2D(
        filters = filters,
        kernel_size=  (1, 1),
        strides = 1,
        padding = 'same',
        use_bias=False
    )(out)

    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.ReLU()(out)

    return out

@gin.configurable
def InceptionResNetV2_block(inputs, filters):
    """
    Inception V2 module with multiple branches (Factorized convolutions, pooling, etc.).
    """
    # Branch 1: 1x1 Convolution
    branch_1 = tf.keras.layers.Conv2D(filters, (1, 1), padding="same", use_bias=False)(inputs)
    branch_1 = tf.keras.layers.BatchNormalization()(branch_1)
    branch_1 = tf.keras.layers.ReLU()(branch_1)

    # Branch 2: 1x1 Convolution -> 3x3 Convolution
    branch_2 = tf.keras.layers.Conv2D(filters, (1, 1), padding="same", use_bias=False)(inputs)
    branch_2 = tf.keras.layers.BatchNormalization()(branch_2)
    branch_2 = tf.keras.layers.ReLU()(branch_2)
    branch_2 = tf.keras.layers.Conv2D(filters, (3, 3), padding="same", use_bias=False)(branch_2)
    branch_2 = tf.keras.layers.BatchNormalization()(branch_2)
    branch_2 = tf.keras.layers.ReLU()(branch_2)

    # Branch 3: 1x1 Convolution -> Factorized 3x3 Convolutions (3x3 split into 1x3 and 3x1)
    branch_3 = tf.keras.layers.Conv2D(filters, (1, 1), padding="same", use_bias=False)(inputs)
    branch_3 = tf.keras.layers.BatchNormalization()(branch_3)
    branch_3 = tf.keras.layers.ReLU()(branch_3)
    branch_3 = tf.keras.layers.Conv2D(filters, (1, 3), padding="same", use_bias=False)(branch_3)
    branch_3 = tf.keras.layers.BatchNormalization()(branch_3)
    branch_3 = tf.keras.layers.ReLU()(branch_3)
    branch_3 = tf.keras.layers.Conv2D(filters, (3, 1), padding="same", use_bias=False)(branch_3)
    branch_3 = tf.keras.layers.BatchNormalization()(branch_3)
    branch_3 = tf.keras.layers.ReLU()(branch_3)

    # Branch 4: Pooling + 1x1 Convolution
    branch_4 = tf.keras.layers.MaxPooling2D((3, 3), strides=1, padding="same")(inputs)
    branch_4 = tf.keras.layers.Conv2D(filters, (1, 1), padding="same", use_bias=False)(branch_4)
    branch_4 = tf.keras.layers.BatchNormalization()(branch_4)
    branch_4 = tf.keras.layers.ReLU()(branch_4)

    # Concatenate all branches
    outputs = tf.keras.layers.Concatenate()([branch_1, branch_2, branch_3, branch_4])

    return outputs