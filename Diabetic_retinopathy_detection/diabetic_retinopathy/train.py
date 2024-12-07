import gin
import tensorflow as tf
import logging
import wandb


@gin.configurable
class Trainer(object):
    def __init__(self, model, ds_train, ds_val, ds_info, run_paths, num_batches, total_epochs, learning_rate):
        # Summary Writer
        # ....

        # Checkpoint Manager
        # ...
        self.checkpoint = tf.train.Checkpoint(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                                              model=model)
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint,
                                                             directory=run_paths["path_ckpts_train"],
                                                             max_to_keep = 1)
        # Loss objective
        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=False) # from_logits=False: output has already been processed through the sigmoid activation function.
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # Metrics
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')

        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')

        self.model = model
        self.ds_train = ds_train
        self.ds_val = ds_val
        self.ds_info = ds_info
        self.run_paths = run_paths
        self.total_epochs = total_epochs
        self.log_interval = num_batches
        #self.ckpt_interval = ckpt_interval


        print(f"Number of batches in validation dataset: {len(list(self.ds_val))}")
        for image, label in ds_val.take(1):
            print(f"Validation batch shape: {image.shape}, Label shape: {label.shape}")

    @tf.function
    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = self.model(images, training=True)
            loss = self.loss_object(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(labels, predictions)

    @tf.function
    def test_step(self, images, labels):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = self.model(images, training=False)
        t_loss = self.loss_object(labels, predictions)

        self.test_loss(t_loss)
        self.test_accuracy(labels, predictions)


    def train(self):
        #print(f"no of batches is {self.log_interval}")
        for idx, (images, labels) in enumerate(self.ds_train):

            step = idx + 1
            self.train_step(images, labels)

            if step % (1 * self.log_interval) == 0:

                # Reset test metrics
                self.test_loss.reset_states()
                self.test_accuracy.reset_states()

                for test_images, test_labels in self.ds_val:
                    self.test_step(test_images, test_labels)

                template = 'epochs: {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
                logging.info(template.format(step/self.log_interval,
                                             self.train_loss.result(),
                                             self.train_accuracy.result() * 100,
                                             self.test_loss.result(),
                                             self.test_accuracy.result() * 100))

                # wandb logging
                wandb.log({'train_acc': self.train_accuracy.result() * 100, 'train_loss': self.train_loss.result(),
                           'val_acc': self.test_accuracy.result() * 100, 'val_loss': self.test_loss.result(),
                           'step': step})

                # Write summary to tensorboard
                # ...

                # Reset train metrics
                self.train_loss.reset_states()
                self.train_accuracy.reset_states()

                yield self.test_accuracy.result().numpy()

            # if step % self.ckpt_interval == 0:
            #     logging.info(f'Saving checkpoint to {self.run_paths["path_ckpts_train"]}.')
            #     # Save checkpoint
            #     # ...
            #     self.checkpoint_manager.save()

            if step % (self.total_epochs * self.log_interval) == 0:
                logging.info(f'Finished training after {step/self.log_interval} epochs.')
                # Save final checkpoint
                # ...
                self.checkpoint_manager.save()
                return self.test_accuracy.result().numpy()