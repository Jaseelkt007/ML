# DL Project : Human Activity Recognition

# How to run the code
1. Run the main.py with FLAGS.train = True for gru model by running the sbatch.sh. You can get the training accuracy, validation accuracy and checkpoints for 2 models (gru model and lstm model). After that, you can click the link of wandb to check the charts.
2. Run the main.py with FLAGS.train = True for lstm model by running the sbatch.sh(please add # to train_selected_model(model_type='gru', ds_train=ds_train, ds_val=ds_val, batch_size=batch_size, run_paths=run_paths_2) and cancel the # of train_selected_model(model_type='lstm', ds_train=ds_train, ds_val=ds_val, batch_size=batch_size, run_paths=run_paths_1) in main.py and add # to the codes of # Training for gru_like and cancel the # to the codes of # Training for lstm_like in config.gin). 
3. Run the main.py with  FLAGS.train = False for gru_like by running the sbatch.sh. You can get the confusion_matrix and evaluation accuracy of both models.
4. Run the main.py with FLAGS.train = False for lstm_like by running the sbatch.sh (please add # to restore_checkpoint(model=model_2, checkpoint_path=/home/RUS_CIP/st186731/DL_LAB_HAPT/HAR/experiments/lstm_like/ckpts) and cancel # to restore_checkpoint(model=model_1, checkpoint_path=/home/RUS_CIP/st186731/DL_LAB_HAPT/HAR/experiments/gru_like/ckpts) in main.py)
5. Run the visualization.py by running the sbatch.sh
6. Run the wandb_sweep.py(for Hyper parameter optimization) by running the sbatch.sh(Please add # to all codes of # Metrics in config.gin). You can get the training accuracy, validation accuracy and checkpoints for 2 models(gru and lstm). After that, you can click the link of wandb in the result to get the hyperparameters you need and check the charts.

# Results

After training, we get the results for both models: GRU and LSTM. See table below
# Table : LSTM and GRU Model Performance

| Model | Trainable Parameters | Validation Accuracy |    Test Accuracy    |
|-------|----------------------|---------------------|---------------------|
| LSTM  | 84,578               | 69.94%              |        82.35%       |
| GRU   | 86,424               | 80.48%              |        83,5%        |