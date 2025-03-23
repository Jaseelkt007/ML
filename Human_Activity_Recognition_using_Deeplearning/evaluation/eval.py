import tensorflow as tf
from evaluation.metrics import ConfusionMatrix
import wandb
import gin

@gin.configurable
def evaluate(model, ds_test):

    metrics = ConfusionMatrix()
    accuracy_list = []

    for idx, (dataset, labels) in enumerate(ds_test):

        # Model
        final_predictions = model(dataset, training=False)

        # Convert predictions to class labels
        predicted_labels = tf.argmax(final_predictions, axis=-1)
        true_labels = tf.cast(labels, tf.int64)

        matches = tf.cast(predicted_labels == true_labels, tf.float32)
        if tf.size(matches) > 0:
            batch_accuracy = tf.reduce_mean(matches) # tf.reduce_mean([1 2 3 4 5]) => (1+2+3+4+5)/5 = 3
            accuracy_list.append(batch_accuracy.numpy())
        else:
            print("No non-zero labels in this batch. Skipping accuracy calculation.")

        # Update confusion matrix metrics
        metrics.update_state(predicted_labels, true_labels)

    # Calculate accuracy
    accuracy = 100 * sum(accuracy_list) / len(accuracy_list)

    # Log the test accuracy to WandB
    wandb.log({'Evaluation_accuracy': accuracy})

    # Plot confusion matrix
    metrics.plot_confusion_matrix(normalize=True)

    # Results
    matrix = metrics.result().numpy()
    print(f"Evaluation_accuracy: {accuracy:6f}%")
    print("Confusion Matrix:")
    print(matrix)

    return {
        "Evaluation_accuracy": accuracy,
        "confusion_matrix": matrix
    }