import gin
from tensorflow.data.experimental import AUTOTUNE
import tensorflow as tf
import logging
from input_pipeline.preprocessing import preprocess, augment
import tensorflow_datasets as tfds
from tensorflow.keras.utils import image_dataset_from_directory

@gin.configurable
def load(batch_size, name, data_dir, test_data_dir, caching=True):
    if name == "idrid":
        logging.info(f"Preparing dataset {name}...")

        # Load dataset from directory structure, where each subdirectory represents a class,return an object, which is an iterable tuples (image, label)
        full_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            batch_size=batch_size,
            label_mode='int') # use 'int' for integer label , for classification

        # Calculate the number of examples for shuffle buffer size
        num_examples = sum(1 for _ in full_ds.unbatch())
        for images, _ in full_ds.take(1):
            image_shape = images.shape[1:]
            break

        class_names = full_ds.class_names
        num_classes = len(class_names)

        # Define df_info
        ds_info = {
            "num_examples": num_examples,
            "features": {
                "image": {"shape": image_shape, "dtype": tf.float32},
                "label": {"num_classes": num_classes, "dtype": tf.int64}
            }
        }
        num_batches = int(len(full_ds) * 0.8)
        print(f"no of batches is {num_batches}")

        # Split into training and validation sets
        total_samples = sum(1 for _ in full_ds.unbatch())
        val_size = int(0.2 * total_samples)
        train_size = int(total_samples) - val_size

        # flatten the dataset
        ds_train = full_ds.unbatch().take(train_size)
        ds_val = full_ds.unbatch().skip(train_size)

        # allow the data to be processed in chunks during training and validation
        ds_train = ds_train.batch(batch_size = batch_size)
        ds_val = ds_val.batch(batch_size = batch_size)

        ds_test= None
        if test_data_dir:
            ds_test = tf.keras.preprocessing.image_dataset_from_directory(
                test_data_dir, batch_size = 1, label_mode = 'int'
            )

        # Prepare and return the training and validation datasets
        return prepare(ds_train, ds_val, num_batches, ds_test=ds_test, ds_info=ds_info, caching=caching)


    elif name == "eyepacs":
        logging.info(f"Preparing dataset {name}...")
        (ds_train, ds_val, ds_test), ds_info = tfds.load(
            'diabetic_retinopathy_detection/btgraham-300',
            split=['train', 'validation', 'test'],
            shuffle_files=True,
            with_info=True,
            data_dir=data_dir
        )

        def _preprocess(img_label_dict):

            return img_label_dict['image'], img_label_dict['label']

        ds_train = ds_train.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_val = ds_val.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_test = ds_test.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        return prepare(ds_train, ds_val, ds_test, ds_info)

    elif name == "mnist":
        logging.info(f"Preparing dataset {name}...")
        (ds_train, ds_val, ds_test), ds_info = tfds.load(
            'mnist',
            split=['train[:90%]', 'train[90%:]', 'test'],
            shuffle_files=True,
            with_info=True,
            data_dir=data_dir
        )

        return prepare(ds_train, ds_val, ds_test, ds_info)

    else:
        raise ValueError


@gin.configurable
def prepare(ds_train, ds_val, num_batches, ds_test = None, ds_info=None, caching=True):

    """Prepare datasets with preprocessing, augmentation, batching, caching, and prefetching"""
    # Prepare training dataset
    ds_train = ds_train.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

    ds_train = ds_train.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if caching:
        ds_train = ds_train.cache()

    if ds_info:
       shuffle_buffer_size = ds_info.get("num_examples", 1000) // 10  # Default to 1000 if ds_info not provided
       ds_train = ds_train.shuffle(shuffle_buffer_size)
    else:
        ds_train = ds_train.shuffle(1000)  # Fallback shuffle size

    ds_train = ds_train.repeat().prefetch(tf.data.AUTOTUNE)

    # Prepare validation dataset (no augmentation)
    ds_val = ds_val.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if caching:
        ds_val = ds_val.cache()
    ds_val = ds_val.prefetch(tf.data.AUTOTUNE)

    # Prepare test dataset if available (no augmentation)
    if ds_test is not None:
        ds_test = ds_test.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        if caching:
            ds_test = ds_test.cache()
        ds_test = ds_test.prefetch(tf.data.AUTOTUNE)


    return ds_train, ds_val, ds_test, ds_info , num_batches


