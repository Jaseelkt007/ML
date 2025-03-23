import gin
import logging
import wandb
from absl import app, flags
from train import Trainer
from evaluation.eval import evaluate
from input_pipeline.datasets import load
from utils import utils_params, utils_misc
from models.architectures import lstm_like, gru_like
import tensorflow as tf
import random

import numpy as np
import os

def random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    #os.environ['TF_DETERMINISTIC_OPS'] = '1'
random_seed(41)

FLAGS = flags.FLAGS
flags.DEFINE_boolean('train',True,'Specify whether to train or evaluate a model.')

def train_model(model, ds_train, ds_val, batch_size, run_paths, path_model_id):
    print('-' * 88)
    print(f'Starting training {path_model_id}')
    model.summary()
    trainer = Trainer(model, ds_train, ds_val, run_paths, batch_size)
    for layer in model.layers:
        print(layer.name, layer.trainable)
    for _ in trainer.train():
        continue
    print(f"Training checkpoint path for {path_model_id}: {run_paths['path_ckpts_train']}")
    print(f'Training completed for {path_model_id}')
    print('-' * 88)

def main(argv):

    # generate folder structures
    run_paths_1 = utils_params.gen_run_folder(path_model_id = 'lstm_like')
    run_paths_2 = utils_params.gen_run_folder(path_model_id = 'gru_like')


    # set loggers
    utils_misc.set_loggers(run_paths_1['path_logs_train'], logging.INFO)
    utils_misc.set_loggers(run_paths_2['path_logs_train'], logging.INFO)


    # gin-config
    gin.parse_config_files_and_bindings(['DL_LAB_HAPT/HAR/Human_Activity_Recognition/configs/config.gin'], [])
    utils_params.save_config(run_paths_1['path_gin'], gin.config_str())
    utils_params.save_config(run_paths_2['path_gin'], gin.config_str())

    # setup pipeline
    ds_train, ds_val, ds_test, batch_size = load()


    # model
    model_1 = lstm_like(input_shape=(250, 6), n_classes=12)

    model_2 = gru_like(input_shape=(250, 6), n_classes=12)

    if FLAGS.train:

        def train_selected_model(model_type, ds_train, ds_val, batch_size, run_paths):
            """
            Train the selected model (LSTM or GRU).

            Args:
                model_type (str): Type of model to train ('lstm' or 'gru').
                ds_train: Training dataset.
                ds_val: Validation dataset.
                batch_size (int): Batch size for training.
                run_paths (dict): Run paths dictionary.
            """
            if model_type == 'lstm':
                model = model_1
                path_model_id = 'lstm_like'
            elif model_type == 'gru':
                model = model_2
                path_model_id = 'gru_like'
            else:
                raise ValueError("Invalid model_type. Choose either 'lstm' or 'gru'.")

            wandb.init(project='Human_Activity_Recognition', name=run_paths['model_id'],
                       config=utils_params.gin_config_to_readable_dictionary(gin.config._CONFIG))

            train_model(model=model,
                        ds_train=ds_train,
                        ds_val=ds_val,
                        batch_size=batch_size,
                        run_paths=run_paths,
                        path_model_id=path_model_id)

            wandb.finish()

        # Example usage
        train_selected_model(model_type='gru', ds_train=ds_train, ds_val=ds_val, batch_size=batch_size,
                     run_paths=run_paths_2)
        #train_selected_model(model_type='lstm', ds_train=ds_train, ds_val=ds_val, batch_size=batch_size,
        #                     run_paths=run_paths_1)


    else:
        def restore_checkpoint(model, checkpoint_path):
            """
            Restore the latest checkpoint for the given model.

            Args:
                model: The model to restore the checkpoint for.
                checkpoint_path (str): Path to the checkpoint directory.
            """
            checkpoint = tf.train.Checkpoint(model=model)
            latest_checkpoint = tf.train.latest_checkpoint(checkpoint_path)
            if latest_checkpoint:
                print(f"Restoring from checkpoint: {latest_checkpoint}")
                status = checkpoint.restore(latest_checkpoint)
                status.expect_partial()
            else:
                print("No checkpoint found. Starting from scratch.")

        # Restore LSTM checkpoint
        # restore_checkpoint(model=model_1, checkpoint_path='/home/RUS_CIP/st186731/DL_LAB_HAPT/HAR/experiments/gru_like/ckpts')

        # Restore GRU checkpoint
        restore_checkpoint(model=model_2, checkpoint_path='/home/RUS_CIP/st186731/DL_LAB_HAPT/HAR/experiments/lstm_like/ckpts')

        wandb.init(project='Human_Activity_Recognition', name='evaluation_phase',
                   config=utils_params.gin_config_to_readable_dictionary(gin.config._CONFIG))

        evaluate(model=model_2, ds_test=ds_test)
        wandb.finish()



if __name__ == "__main__":
    wandb.login(key="")
    app.run(main)
