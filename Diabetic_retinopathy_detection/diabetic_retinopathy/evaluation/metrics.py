import tensorflow as tf

class ConfusionMatrix(tf.keras.metrics.Metric):
    def __init__(self, num_classes=2, name="confusion_matrix", **kwargs):
        super(ConfusionMatrix, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.tp = self.add_weight(name="true_positives", initializer="zeros", dtype=tf.int32)
        self.fp = self.add_weight(name="false_positives", initializer="zeros", dtype=tf.int32)
        self.tn = self.add_weight(name="true_negatives", initializer="zeros", dtype=tf.int32)
        self.fn = self.add_weight(name="false_negatives", initializer="zeros", dtype=tf.int32)

    def update_state(self, y_true, y_pred, sample_weight=None):

        # Calculate confusion matrix components
        self.tp.assign_add(tf.reduce_sum(tf.cast((y_pred == 1) & (y_true == 1), tf.int32)))
        self.fp.assign_add(tf.reduce_sum(tf.cast((y_pred == 1) & (y_true == 0), tf.int32)))
        self.tn.assign_add(tf.reduce_sum(tf.cast((y_pred == 0) & (y_true == 0), tf.int32)))
        self.fn.assign_add(tf.reduce_sum(tf.cast((y_pred == 0) & (y_true == 1), tf.int32)))

    def result(self):
        # Return confusion matrix components as a dictionary
        return {
            "tp": self.tp.numpy(),
            "fp": self.fp.numpy(),
            "tn": self.tn.numpy(),
            "fn": self.fn.numpy()
        }


