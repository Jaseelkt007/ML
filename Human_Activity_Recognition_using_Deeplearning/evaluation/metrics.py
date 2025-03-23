import tensorflow as tf
import gin
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

@gin.configurable
class ConfusionMatrix(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name, labels_name, save_path, **kwargs):
        super(ConfusionMatrix, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.labels_name = labels_name
        self.save_path = save_path
        self.matrix = tf.Variable(initial_value=tf.zeros((self.num_classes, self.num_classes), dtype=tf.float32), dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):

        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.int32)

        new_matrix = tf.math.confusion_matrix(y_true, y_pred, num_classes = self.num_classes, dtype = tf.float32)
        self.matrix.assign_add(new_matrix)

    def result(self):
        return self.matrix

    def plot_confusion_matrix(self, normalize, cmap="Reds"):
        matrix = self.matrix.numpy()

        if normalize:
            matrix = matrix.astype('float') / matrix.sum(axis = 1, keepdims = True) # Normalize the matrix in each row
            matrix = np.nan_to_num(matrix) # Replace NaN with 0

        plt.figure(figsize=(10, 10))
        sns.heatmap(matrix, annot = True, fmt=".2f", cmap = cmap, xticklabels = self.labels_name, yticklabels = self.labels_name)
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        print("DEBUG: self.save_path =", self.save_path)
        if self.save_path:
            plt.savefig(self.save_path)
            print(f"Confusion matrix saved to {self.save_path}")
