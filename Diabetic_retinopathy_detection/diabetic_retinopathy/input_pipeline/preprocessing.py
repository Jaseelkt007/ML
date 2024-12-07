import gin
import tensorflow as tf



@gin.configurable
def preprocess(image, label, img_height = 256, img_width = 256):
    """Dataset preprocessing: Normalizing and resizing"""
  #  print(f"Before augment - Image Min: {tf.reduce_min(image)}, Max: {tf.reduce_max(image)}")
    image = tf.image.resize(image, (img_height, img_width))
    # Normalize image to [0, 1] and resize
    image = tf.cast(image, tf.float32) / 255.0

    image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    #print(f"After augment - Image Min: {tf.reduce_min(image)}, Max: {tf.reduce_max(image)}")
    return image, label

augmentation_layer = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(factor=0.2),  # Approximately ±10 degrees,
        tf.keras.layers.RandomZoom(height_factor=0.2, width_factor=0.2), # small zoom
        tf.keras.layers.RandomBrightness(0.1),
        tf.keras.layers.RandomContrast(0.1),
        tf.keras.layers.RandomFlip("horizontal_and_vertical")
    ])

def augment(image, label):
    """Data augmentation"""
   # tf.print("Before augment - Image Min:", tf.reduce_min(image), ", Max:", tf.reduce_max(image))
    image = augmentation_layer(image)
   # tf.print("After augment - Image Min:", tf.reduce_min(image), ", Max:", tf.reduce_max(image))
    return image, label









