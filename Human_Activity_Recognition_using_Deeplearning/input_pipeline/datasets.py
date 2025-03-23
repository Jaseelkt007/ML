import gin
from tensorflow.data.experimental import AUTOTUNE
import tensorflow as tf
import logging
from input_pipeline.preprocessing import preprocess, oversample
from input_pipeline.sliding_window import sliding_window
from input_pipeline.parse_labels import cal_exp_lengths, parse_labels, create_label_tensor, remap_labels
import numpy as np
import os
from visualization.visualization import visualize_data

@gin.configurable
def load(batch_size, name, data_dir, labels_file):
    if name == "HAPT":
        logging.info(f"Preparing dataset {name}...")

        acc_files = sorted([f for f in os.listdir(data_dir) if f.startswith("acc")])
        gyro_files = sorted([f for f in os.listdir(data_dir) if f.startswith("gyro")])

        ds_train = []
        ds_val = []
        ds_test = []

        for acc_file, gyro_file in zip(acc_files, gyro_files):

            acc_file_path = os.path.join(data_dir, acc_file)
            gyro_file_path = os.path.join(data_dir, gyro_file)
            acc_data = tf.convert_to_tensor(np.loadtxt(acc_file_path), dtype=tf.float32)
            gyro_data = tf.convert_to_tensor(np.loadtxt(gyro_file_path), dtype=tf.float32)
            combined = tf.concat([acc_data, gyro_data], axis=1)

            # Extract user ID from filename
            user_id = int(acc_file.split('_user')[1].split('.txt')[0])

            if 1 <= user_id <= 21:
                ds_train.append(combined)
            elif 28 <= user_id <= 30:
                ds_val.append(combined)
            elif 22 <= user_id <= 27:
                ds_test.append(combined)

        ds_train = tf.concat(ds_train, axis=0)  # aixs = 0 means vertical concatenation
        ds_val = tf.concat(ds_val, axis=0)
        ds_test = tf.concat(ds_test, axis=0)


    else:
        raise ValueError

    experiment_length = cal_exp_lengths(data_dir)
    total_time_steps = sum(experiment_length)
    segments = parse_labels(labels_file, exp_lengths=experiment_length)
    label_tensor = create_label_tensor(segments, total_time_steps)

    # Calculate cumulative lengths for splitting
    train_length = ds_train.shape[0]

    test_length = ds_test.shape[0]

    train_labels = label_tensor[:train_length]
    test_labels = label_tensor[train_length:train_length+test_length]
    val_labels = label_tensor[train_length + test_length:]

    ds_train, ds_val, ds_test = prepare(ds_train, ds_val, ds_test, train_labels, val_labels, test_labels, batch_size)

    return ds_train, ds_val, ds_test, batch_size


@gin.configurable
def prepare(ds_train, ds_val, ds_test, train_labels, val_labels, test_labels , batch_size, window_size = 250):
    """Prepare datasets with preprocessing, batching, caching, and prefetching"""

    def prepare_dataset(data, labels, batch_size, window_size, overlap = 0.5, shuffle_buffer = 64, cache = True , is_training=True, debug=False):
        # Step 1 : Normalize the data
        data = preprocess(data)

        if is_training:
            datasets = oversample(data, labels,debug=debug)
        else:
            datasets = tf.data.Dataset.from_tensor_slices((data, labels))
        print(f"datasets:{datasets.take(1)}")

        # visualize_data(datasets, oversample = True)
        # Step 2 : Create sliding window
        dataset = sliding_window(datasets, window_size=window_size, overlap=overlap)

        # Filter out window with label = 0:
        dataset = dataset.filter(lambda _, label: label > 0)

        # Step 3 : since previous steps are deterministic, caching is done before preprocessing
        if cache:
            dataset = dataset.cache()

        if is_training:
            dataset = dataset.shuffle(shuffle_buffer).repeat()
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        return dataset

    # Prepare datasets
    ds_train = prepare_dataset(ds_train, train_labels, batch_size, window_size, cache = True, debug=True)
    ds_val = prepare_dataset(ds_val, val_labels, batch_size, window_size, cache = True, is_training=False)
    ds_test = prepare_dataset(ds_test, test_labels, batch_size, window_size, cache = True, is_training=False)

    ds_train = remap_labels(ds_train)
    ds_val = remap_labels(ds_val)
    ds_test = remap_labels(ds_test)

    return ds_train, ds_val, ds_test
