import logging
import wandb
import gin
import math

import tensorflow as tf
from input_pipeline.datasets import load
from models.architectures import lstm_like, gru_like
from train import Trainer
from utils import utils_params, utils_misc

@gin.configurable
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

def evaluate(model, ds_test):

    accuracy_list = []

    for idx, (dataset, labels) in enumerate(ds_test):
        final_predictions = model(dataset, training=False)
        # print(f"final_predictions:{final_predictions}")

        # Convert predictions to class labels
        predicted_labels = tf.argmax(final_predictions, axis=-1)
        true_labels = tf.cast(labels, tf.int64)

        matches = tf.cast(predicted_labels == true_labels, tf.float32)
        if tf.size(matches) > 0:
            batch_accuracy = tf.reduce_mean(matches)  # tf.reduce_mean([1 2 3 4 5]) => (1+2+3+4+5)/5 = 3
            accuracy_list.append(batch_accuracy.numpy())
        else:
            print("No non-zero labels in this batch. Skipping accuracy calculation.")

    # Calculate accuracy
    accuracy = 100*sum(accuracy_list) / len(accuracy_list)

    # Log the test accuracy to WandB
    wandb.log({'Evaluation_accuracy':accuracy})

    return accuracy

def train_func():

    with wandb.init() as run:
        gin.clear_config()
        # Hyperparameters
        bindings = []
        for key, value in run.config.items():
            if isinstance(value, str): # This checks whether the variable value is of type str.
                bindings.append(f"{key}='{value}'")
            else:
                bindings.append(f"{key}={value}")

        # generate folder structures
        model_type = run.config['model_type']
        run_paths = utils_params.gen_run_folder(path_model_id = model_type)

        # set loggers
        utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

        # gin-config
        gin.parse_config_files_and_bindings(['DL_LAB_HAPT/HAR/Human_Activity_Recognition/configs/config.gin'], bindings)
        utils_params.save_config(run_paths['path_gin'], gin.config_str())

        # setup pipeline
        ds_train, ds_val, ds_test, batch_size = load(name='HAPT')

        # Model
        if model_type == 'lstm_like':
            model = lstm_like(input_shape=(128, 6), n_classes=12)
            train_model(model=model,
                        ds_train=ds_train,
                        ds_val=ds_val,
                        batch_size=batch_size,
                        run_paths=run_paths,
                        path_model_id='lstm_like')
        elif model_type == 'gru_like':
            model = gru_like(input_shape=(128, 6), n_classes=12)
            train_model(model=model,
                        ds_train=ds_train,
                        ds_val=ds_val,
                        batch_size=batch_size,
                        run_paths=run_paths,
                        path_model_id='gru_like')

        else:
            raise ValueError

        # Evaluate the model after training
        print(f"Evaluating {model_type} on the test dataset...")

        accuracy = evaluate(model, ds_test)
        print(f"Evaluation accuracy for {model_type}: {accuracy}")

        # Log the test accuracy to WandB
        wandb.log({'evaluation_accuracy': accuracy})

        wandb.finish()

model_types = ['lstm_like', 'gru_like']

for model in model_types:
    if model == 'lstm_like':
        sweep_config = {
            'name': f"{model}-sweep",
            'method': 'bayes',
            'metric': {
                'name': 'val_acc',
                'goal': 'maximize'
            },
            'parameters': {
                'Trainer.total_epochs': {
                    'values': [55]
                },
                'Trainer.poly_loss_alpha': {
                    'distribution': 'uniform',
                    'min': 0.1,
                    'max': 1
                },
                'Trainer.rdrop_alpha': {
                    'distribution': 'uniform',
                    'min': 0.1,
                    'max': 1
                },
                'model_type':{
                    'values': [model]
                },
                'lstm_like.lstm_units': {
                    'distribution': 'q_uniform',
                    'min': 32,
                    'max': 128,
                    'q': 1
                },
                'lstm_like.n_blocks': {
                    'distribution': 'q_uniform',
                    'min': 1,
                    'max': 3,
                    'q': 1
                },
                'lstm_like.dense_units': {
                    'distribution': 'q_uniform',  # Use q_uniform for quantized values
                    'min': 32,
                    'max': 256,
                    'q': 1  # Step size, ensures only integers are selected
                },
                'lstm_like.dropout_rate_lstm_block': {
                    'distribution': 'uniform',
                    'min': 0.2,
                    'max': 0.6
                },
                'lstm_like.dropout_rate_dense_layer': {
                    'distribution': 'uniform',
                    'min': 0.2,
                    'max': 0.6
                }
            }
        }
        sweep_id = wandb.sweep(sweep_config)

        wandb.agent(sweep_id, function=train_func, count=100)

    elif model == 'gru_like':
        sweep_config = {
            'name': f"{model}-sweep",
            'method': 'bayes',
            'metric': {
                'name': 'val_acc',
                'goal': 'maximize'
            },
            'parameters': {
                'Trainer.total_epochs': {
                    'values': [50]
                },
                'Trainer.poly_loss_alpha': {
                    'distribution': 'uniform',
                    'min': 0.1,
                    'max': 1
                },
                'Trainer.rdrop_alpha': {
                    'distribution': 'uniform',
                    'min': 0.1,
                    'max': 1
                },
                'model_type':{
                    'values': [model]
                },
                'gru_like.gru_units': {
                    'distribution': 'q_uniform',
                    'min': 32,
                    'max': 128,
                    'q': 1
                },
                'gru_like.n_blocks': {
                    'distribution': 'q_uniform',
                    'min': 1,
                    'max': 3,
                    'q': 1
                },
                'gru_like.dense_units': {
                    'distribution': 'q_log_uniform',
                    'q': 1,
                    'min': math.log(32),
                    'max': math.log(256)
                },
                'gru_like.dropout_rate_gru_block': {
                    'distribution': 'uniform',
                    'min': 0.2,
                    'max': 0.6
                },
                'gru_like.dropout_rate_dense_layer': {
                    'distribution': 'uniform',
                    'min': 0.2,
                    'max': 0.6
                }
            }
        }
        sweep_id = wandb.sweep(sweep_config)

        wandb.agent(sweep_id, function=train_func, count=100)

