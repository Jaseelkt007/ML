import gin
import tensorflow as tf
from tensorflow.keras.applications import MobileNet, VGG16, InceptionResNetV2
from models.layers import vgg_block, mobilenet_block, InceptionResNetV2_block
from tensorflow.keras.regularizers import l2

@gin.configurable
def vgg_like(n_classes, base_filters, n_blocks, dense_units, dropout_rate, input_shape = (256, 256, 3) ):
    """Defines a VGG-like architecture.

    Parameters:
        input_shape (tuple: 3): input shape of the neural network
        n_classes (int): number of classes, corresponding to the number of output neurons
        base_filters (int): number of base filters, which are doubled for every VGG block
        n_blocks (int): number of VGG blocks
        dense_units (int): number of dense units
        dropout_rate (float): dropout rate

    Returns:
        (keras.Model): keras model object
    """
    #Load the pretrained VGG16 model excluding the top classification layer
    base_model = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
    base_model.trainable = False

    assert n_blocks > 0, 'Number of blocks has to be at least 1.'

    inputs = tf.keras.Input(shape = input_shape)
    print(inputs.shape) # -> (None, 256, 256, 3)
    out = base_model(inputs)
    print(f"vgg output shape:{out.shape}") # -> (None, 8, 8, 512)
    for i in range(n_blocks):
        out = vgg_block(out, base_filters, kernel_size=(3, 3))
    out = tf.keras.layers.GlobalAveragePooling2D()(out)
    out = tf.keras.layers.Dense(dense_units, kernel_regularizer=l2(1e-4))(out)
    out = tf.keras.layers.LeakyReLU(alpha = 0.01)(out)
    out = tf.keras.layers.Dropout(dropout_rate)(out)
    outputs = tf.keras.layers.Dense(n_classes - 1, activation = 'sigmoid', kernel_regularizer=l2(1e-4))(out)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name='vgg_like') , base_model

@gin.configurable
def mobilenet_like(n_classes, base_filters, n_blocks, dense_units, input_shape = (256, 256, 3), dropout_rate = 0.5):

    base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
    base_model.trainable = False

    assert n_blocks > 0, 'Number of blocks has to be at least 1.'
    inputs = tf.keras.Input(shape = input_shape)
    out = base_model(inputs)
    print(f"MobileNet output shape:{out.shape}")
    for i in range(n_blocks):
        out = mobilenet_block(out, filters =  base_filters * 2 ** (i+1), strides = 1)
    out = tf.keras.layers.GlobalAveragePooling2D()(out)
    out = tf.keras.layers.Dense(dense_units, activation=tf.nn.relu)(out)
    out = tf.keras.layers.Dropout(dropout_rate)(out)
    outputs = tf.keras.layers.Dense(n_classes-1, activation = tf.sigmoid)(out)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name='mobilenet_like'), base_model


@gin.configurable
def inception_v2_like(n_classes, base_filters, n_blocks, dense_units, input_shape = (256, 256, 3), dropout_rate = 0.5):


    base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
    base_model.trainable = False

    assert n_blocks > 0, 'Number of blocks has to be at least 1.'
    inputs = tf.keras.Input(shape = input_shape)
    out = base_model(inputs)
    print(f"InceptionResNetV2 output shape:{out.shape}")
    for i in range(n_blocks):
        out = InceptionResNetV2_block(out, filters =  base_filters * 2 ** (i+1))
    out = tf.keras.layers.GlobalAveragePooling2D()(out)
    out = tf.keras.layers.Dense(dense_units, activation=tf.nn.relu)(out)
    out = tf.keras.layers.Dropout(dropout_rate)(out)
    outputs = tf.keras.layers.Dense(n_classes-1, activation = tf.sigmoid)(out)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name='inception_v2_like'), base_model
