import gin
import logging
import wandb
from absl import app, flags
from train import Trainer
from evaluation.eval import evaluate
from input_pipeline import datasets
from utils import utils_params, utils_misc
from models.architectures import mobilenet_like, vgg_like, inception_v2_like
import tensorflow as tf
from deep_visualization.GRAD_CAM_visualization import grad_cam_visualization
import random
import numpy as np
import os

def random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    #os.environ['TF_DETERMINISTIC_OPS'] = '1'
random_seed(47)

FLAGS = flags.FLAGS
flags.DEFINE_boolean('train', True,'Specify whether to train or evaluate a model.')

@gin.configurable
def train_model(model, base_model, ds_train, ds_val, num_batches, unfrz_layer, ds_info, run_paths, path_model_id):
    print('-' * 88)
    print(f'Starting training {path_model_id}')
    model.summary()
    trainer = Trainer(model, ds_train, ds_val, ds_info, run_paths, num_batches)
    for layer in model.layers:
        print(layer.name, layer.trainable)
    for _ in trainer.train():
        continue
    for layer in base_model.layers[-unfrz_layer:]:
        layer.trainable = True
    for _ in trainer.train():
        continue
    print(f"Training checkpoint path for {path_model_id}: {run_paths['path_ckpts_train']}")
    print(f'Training completed for {path_model_id}')
    print('-' * 88)

def main(argv):

    # generate folder structures
    run_paths_1 = utils_params.gen_run_folder(path_model_id = 'mobilenet_like')
    run_paths_2 = utils_params.gen_run_folder(path_model_id = 'vgg_like')
    run_paths_3 = utils_params.gen_run_folder(path_model_id = 'inception_v2_like')

    # set loggers
    utils_misc.set_loggers(run_paths_1['path_logs_train'], logging.INFO)
    utils_misc.set_loggers(run_paths_2['path_logs_train'], logging.INFO)
    utils_misc.set_loggers(run_paths_3['path_logs_train'], logging.INFO)

    # gin-config
    gin.parse_config_files_and_bindings(['/home/RUS_CIP/st186731/dl-lab-24w-team04/diabetic_retinopathy/configs/config.gin'], [])
    utils_params.save_config(run_paths_1['path_gin'], gin.config_str())
    utils_params.save_config(run_paths_2['path_gin'], gin.config_str())
    utils_params.save_config(run_paths_3['path_gin'], gin.config_str())

    # setup pipeline
    ds_train, ds_val, ds_test, ds_info, num_batches = datasets.load(name = 'idrid')

     # model
    model_1, base_model_1 = mobilenet_like(input_shape=ds_info["features"]["image"]["shape"],
                                           n_classes=ds_info["features"]["label"]["num_classes"])
    model_2, base_model_2 = vgg_like(input_shape=ds_info["features"]["image"]["shape"],
                                       n_classes=ds_info["features"]["label"]["num_classes"])
    model_3, base_model_3 = inception_v2_like(input_shape=ds_info["features"]["image"]["shape"],
                                       n_classes=ds_info["features"]["label"]["num_classes"])

    if FLAGS.train:

        # Model_1
        wandb.init(project='diabetic-retinopathy-detection', name=run_paths_1['model_id'],
                    config=utils_params.gin_config_to_readable_dictionary(gin.config._CONFIG))# setup wandb

        train_model(model = model_1,
                    base_model = base_model_1,
                    ds_train = ds_train,
                    ds_val = ds_val,
                    num_batches = num_batches,
                    ds_info = ds_info,
                    run_paths = run_paths_1,
                    path_model_id = 'mobilenet_like')
        wandb.finish()

        # Model_2
        wandb.init(project='diabetic-retinopathy-detection', name=run_paths_2['model_id'],
                   config=utils_params.gin_config_to_readable_dictionary(gin.config._CONFIG))
        train_model(model = model_2,
                    base_model = base_model_2,
                    ds_train = ds_train,
                    ds_val = ds_val,
                    num_batches = num_batches,
                    ds_info = ds_info,
                    run_paths = run_paths_2,
                    path_model_id = 'vgg_like')
        wandb.finish()


        # Model_3
        wandb.init(project='diabetic-retinopathy-detection', name=run_paths_3['model_id'],
                   config=utils_params.gin_config_to_readable_dictionary(gin.config._CONFIG))
        train_model(model = model_3,
                    base_model = base_model_3,
                    ds_train = ds_train,
                    ds_val = ds_val,
                    num_batches = num_batches,
                    ds_info = ds_info,
                    run_paths = run_paths_3,
                    path_model_id = 'inception_v2_like')
        wandb.finish()

    else:
        checkpoint_path_1 = '/home/RUS_CIP/st186731/dl-lab-24w-team04/experiments/run_2024-12-07T14-51-45-371592_mobilenet_like/ckpts'
        checkpoint_path_2 = '/home/RUS_CIP/st186731/dl-lab-24w-team04/experiments/run_2024-12-07T14-51-45-371988_vgg_like/ckpts'
        checkpoint_path_3 = '/home/RUS_CIP/st186731/dl-lab-24w-team04/experiments/run_2024-12-07T14-51-45-372289_inception_v2_like/ckpts'

        checkpoint_1 = tf.train.Checkpoint(model = model_1)
        latest_checkpoint_1 = tf.train.latest_checkpoint(checkpoint_path_1)
        if latest_checkpoint_1:
            print(f"Restoring from checkpoint_1: {latest_checkpoint_1}")
            checkpoint_1.restore(latest_checkpoint_1)
        else:
            print("No checkpoint found. Starting from scratch.")

        # Model_2
        checkpoint_2 = tf.train.Checkpoint(model = model_2)
        latest_checkpoint_2 = tf.train.latest_checkpoint(checkpoint_path_2)
        if latest_checkpoint_2:
            print(f"Restoring from checkpoint_2: {latest_checkpoint_2}")
            checkpoint_2.restore(latest_checkpoint_2)
        else:
            print("No checkpoint found. Starting from scratch.")

        # Model_3
        checkpoint_3 = tf.train.Checkpoint(model = model_3)
        latest_checkpoint_3 = tf.train.latest_checkpoint(checkpoint_path_3)
        if latest_checkpoint_3:
            print(f"Restoring from checkpoint_3: {latest_checkpoint_3}")
            checkpoint_3.restore(latest_checkpoint_3)
        else:
            print("No checkpoint found. Starting from scratch.")

        wandb.init(project='diabetic-retinopathy-detection', name='evaluation_phase',
                   config=utils_params.gin_config_to_readable_dictionary(gin.config._CONFIG))

        evaluate(model_1 = model_1, model_2 = model_2, model_3 = model_3, ds_test = ds_test , ensemble=False)

        grad_cam_visualization(model = model_2)

if __name__ == "__main__":
    wandb.login(key="40c93726af78ad0b90c6fe3174c18599ecf9f619")
    app.run(main)