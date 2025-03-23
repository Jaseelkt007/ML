import gin
import tensorflow as tf

@gin.configurable
def preprocess(data):
    """Dataset preprocessing: Normalizing"""
    mean = tf.reduce_mean(data, axis = 0)
    std = tf.math.reduce_std(data, axis = 0)

    return (data - mean)/ std

def oversample(data, labels, debug=True):
    """
    Efficiently oversample the dataset by iterating through labels and checking sizes before oversampling.

    Args:
        data: Tensor of input data.
        labels: Tensor of corresponding labels.
        debug: Boolean to enable debug information.

    Returns:
        tf.data.Dataset: Oversampled dataset with consistent shapes and original order.
    """
    # Ensure labels are 1D
    labels = tf.squeeze(labels)

    # Get unique activities and their counts
    activities,_, activity_counts = tf.unique_with_counts(labels)

    # Determine max counts for majority and minority classes
    primary_mask = tf.logical_and(activities <7, activities !=0) # Exclude label 0
    transition_mask = tf.logical_and(activities >=7 , activities!=0) # Exclude label 0

    max_activity = tf.reduce_max(tf.boolean_mask(activity_counts, primary_mask))  # Max count of majority labels
    max_transition_activity = tf.reduce_max(tf.boolean_mask(activity_counts, transition_mask)) + 80000  # Max count of minority labels

    if debug:
        print(f"Max activity for primary labels (majority): {max_activity.numpy()}")
        print(f"Max activity for transition labels (minority): {max_transition_activity.numpy()}")

    # Initialize lists for oversampled data and labels
    oversampled_data = []
    oversampled_labels = []

    # Dictionary to track oversampled counts for each activity
    activity_counts_dict = {activity: 0 for activity in activities.numpy()}

    # Iterate through all labels
    for idx, label in enumerate(labels.numpy()):
        # Skip if already oversampled
        if activity_counts_dict[label] >= (max_activity if label < 7 else max_transition_activity):
            continue

        # Find indices for current label
        activity_indices = tf.where(labels == label)[:, 0]

        # Determine oversampling size
        if label in [7]: # Special handling for label 7
            oversample_size = max_transition_activity # + 40000 - activity_counts_dict[label]
        elif label < 7:
            oversample_size = max_activity - activity_counts_dict[label]
        else:
            oversample_size = max_transition_activity - activity_counts_dict[label]

        oversample_size = min(oversample_size, tf.shape(activity_indices)[0])  # Avoid oversampling beyond data

        # Perform random sampling with replacement
        sampled_indices = tf.random.uniform(
            shape=[oversample_size], minval=0, maxval=tf.shape(activity_indices)[0], dtype=tf.int32
        )
        sampled_indices = tf.gather(activity_indices, sampled_indices)

        # Gather oversampled data and labels
        gathered_data = tf.gather(data, sampled_indices)
        gathered_labels = tf.gather(labels, sampled_indices)

        # Append to results
        oversampled_data.append(gathered_data)
        oversampled_labels.append(gathered_labels)

        # Update counts
        activity_counts_dict[label] += oversample_size

        # Debug info
        if debug and idx < 10:  # Limit debug to first 10 iterations
            print(f"Label: {label}, Oversample Size: {oversample_size}")
            print(f"Current Count: {activity_counts_dict[label]}")
            print(f"Gathered Data Shape: {gathered_data.shape}")
            print(f"Gathered Labels Shape: {gathered_labels.shape}")

        # Break if all labels are oversampled to desired size
        if all(
            count >= (max_activity if activity < 7 else max_transition_activity)
            for activity, count in activity_counts_dict.items()
        ):
            break

    # Concatenate results
    oversampled_data = tf.concat(oversampled_data, axis=0)
    oversampled_labels = tf.concat(oversampled_labels, axis=0)

    # Debug final shapes
    if debug:
        print(f"Final Oversampled Data Shape: {oversampled_data.shape}")
        print(f"Final Oversampled Labels Shape: {oversampled_labels.shape}")

    # Create tf.data.Dataset
    return tf.data.Dataset.from_tensor_slices((oversampled_data, oversampled_labels))






