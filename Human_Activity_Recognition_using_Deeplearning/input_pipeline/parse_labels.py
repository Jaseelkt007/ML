import os
import numpy as np

def cal_exp_lengths(data_path, file_prefix ='acc_exp', file_suffix=".txt"):
    """
    Calculate the number of rows for each experiment file and return the cumulative lengths.
    """
    exp_length =[0]
    filenames = sorted(os.listdir(data_path))
    for file in filenames:
        if file.startswith(file_prefix) and file.endswith(file_suffix):
            file_path = os.path.join(data_path, file)
            num_rows = np.loadtxt(file_path).shape[0]
            exp_length.append(num_rows)
    return exp_length

def parse_labels(label_file_path, exp_lengths):
    """
    Parse the label file and adjust the activity start and end indices based on cumulative experiment lengths.
    """
    segments =[]
    cumulative_offset = 0
    current_experiment_id = -1
    with open(label_file_path,'r') as f:
        for line in f:
            parts = line.strip().split()
            exp_id = int(parts[0])
            act_id = int(parts[2])
            start_idx = int(parts[3])
            end_idx = int(parts[4])

            # Update the cummulative offset for next experiment
            if exp_id != current_experiment_id:
                cumulative_offset += exp_lengths[exp_id -1] # Add previous exp length , intial element in exp_len = 0
                current_experiment_id = exp_id

            adjusted_start_idx = start_idx + cumulative_offset
            adjusted_end_idx = end_idx + cumulative_offset

            segments.append({
                "activity_id" : act_id,
                "start_idx" : adjusted_start_idx,
                "end_idx" : adjusted_end_idx
            })
    return segments

def create_label_tensor(segments, total_time_steps):
    """
    Generate a label tensor for all time steps based on parsed activity segments.
    """
    label_tensor = np.zeros(total_time_steps, dtype=int)
    for segment in segments:
      activity_id = segment["activity_id"]
      start_idx = segment["start_idx"]
      end_idx = segment["end_idx"]

      # Update label_array
      label_tensor[start_idx:end_idx] = activity_id
    return label_tensor

def remap_labels(dataset):
    """
    Remap labels from (1,2,3,...,12) to (0,1,....11)
    """
    return dataset.map(lambda data, label: (data, label - 1))
