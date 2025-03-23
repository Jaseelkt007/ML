import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


np.random.seed(42)  # Make sure that we can get the same color every time

# Function to convert TensorFlow dataset to NumPy arrays
def dataset_to_numpy_array(dataset, num_samples=100):
    """
    Convert the first `num_samples` from a TensorFlow dataset to NumPy arrays.
    """
    data_list = []
    labels_list = []

    for data, label in dataset.take(num_samples):  # Take only the first `num_samples`
        data_list.append(data.numpy())
        labels_list.append(label.numpy())

    # Convert to NumPy arrays
    data_array = np.array(data_list)
    labels_array = np.array(labels_list)

    return data_array, labels_array

def visualize_data(dataset, oversample, data_path=None, all_files=None, folder_path_raw_data = '/home/RUS_CIP/st186731/DL_LAB_HAPT/visualization', acc_data=None, gyro_data=None):

    if oversample:
        # Step 1: Convert TensorFlow dataset to NumPy arrays
        num_samples_to_visualize = 10
        #num_samples_to_visualize=2000000
        data_array, labels_array = dataset_to_numpy_array(dataset, num_samples=num_samples_to_visualize)

        # Step 2: Split accelerometer and gyrometer data
        acc_data = data_array[:, :3]  # First 3 columns are accelerometer data
        gyro_data = data_array[:, 3:]  # Last 3 columns are gyrometer data

        # Step 3: Get unique activities and assign colors
        unique_acts = np.unique(labels_array)
        cmap = plt.get_cmap("tab20", len(unique_acts))
        color_map = {act_id: cmap(i) if i != 6 else "#8B0000" for i, act_id in enumerate(unique_acts)}

        # Step 4: Create the figure
        fig = plt.figure(figsize=(18, 8))

        # Top plot: Accelerometer data
        acc_plot = fig.add_subplot(2, 1, 1)
        time_steps = np.arange(acc_data.shape[0])  # Number of samples as time steps

        for i in range(3):  # Plot X, Y, Z data
            color = ['red', 'green', 'blue']
            column_data = acc_data[:, i]  # Get data for X, Y, Z
            acc_plot.plot(time_steps, column_data, color=color[i], label=f"Acc {['X', 'Y', 'Z'][i]}")

        # Add patches for activities
        for idx, act_id in enumerate(labels_array):  # Highlight segments by activity
            c = color_map[act_id]
            acc_plot.axvspan(idx - 0.5, idx + 0.5, color=c, alpha=0.3)

        acc_plot.set_title("Accelerometer Data (Oversampled)")
        acc_plot.legend()

        # Bottom plot: Gyrometer data
        gyro_plot = fig.add_subplot(2, 1, 2)
        for i in range(3):  # Plot X, Y, Z data
            color = ['red', 'green', 'blue']
            column_data = gyro_data[:, i]  # Get data for X, Y, Z
            gyro_plot.plot(time_steps, column_data, color=color[i], label=f"Gyro {['X', 'Y', 'Z'][i]}")

        # Add patches for activities
        for idx, act_id in enumerate(labels_array):  # Highlight segments by activity
            c = color_map[act_id]
            gyro_plot.axvspan(idx - 0.5, idx + 0.5, color=c, alpha=0.3)

        gyro_plot.set_title("Gyrometer Data (Oversampled)")
        gyro_plot.legend()

        # Add activity legend
        legend_patches = [
            patches.Patch(color=color_map[act_id], label=f"Activity {act_id}") for act_id in unique_acts
        ]
        plt.legend(handles=legend_patches, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=5)

        # Adjust subplot spacing and show the plot
        plt.subplots_adjust(hspace=0.5)
        # Save the plot to the path (folder_path_raw_data)

        os.makedirs(folder_path_raw_data, exist_ok=True)

        file_path = os.path.join(folder_path_raw_data, "visualization_oversampled_data.png")
        plt.savefig(file_path)
        print(f"plot saved to:{file_path}")



    else:

        label_files = os.path.join(data_path, "labels.txt")
        labels = np.loadtxt(label_files, dtype=int)

        exp_id = 1
        labels_exp1 = labels[labels[:, 0] == exp_id] # labels[:, 0] extracts the 0th column

        # Create the full path for activity_labels.txt
        activity_labels_path = os.path.join(all_files, "activity_labels.txt")
        activity_labels = []
        with open(activity_labels_path, "r") as f: # Open the activity in read mode
            # " 1 WALKING DOWNSTAIRS " -> ["1", "WALKING", "DOWNSTAIRS"]
            for line in f:
                parts = line.strip().split() # strip() removes leading and trailing whitespace (spaces, tabs, newlines); split() then splits the stripped string by whitespace into a list of substrings
                activity_labels.append((int(parts[0]), " ".join(parts[1:])))

        # Convert list of tuples into a dictionary
        activity_dict = dict(activity_labels)

        # Extracts unique act_id from the third column of labels_exp1 (index 2)
        unique_acts = np.unique(labels_exp1[:, 2])
        cmap = plt.get_cmap("tab20", len(unique_acts))
        color_map = {}
        for i, act_id in enumerate(unique_acts):
            if i != 6:
                color_map[act_id] = cmap(i)
            else:
                color_map[act_id] = "#8B0000"

        fig = plt.figure(figsize=(18, 6))

        # Top plot is for accelerometer data
        acc_plot = fig.add_subplot(3,1,1)

        # Get the accelerometer data for x, y and z axis
        for i in range(3):
            color = ['red', 'green', 'blue']
            column_data = acc_data[:, i]
            plt.plot(np.arange(len(column_data)), column_data, color=color[i])

        # Iterate over all rows (segments), plotting one small segment at a time.
        for row in labels_exp1:
            _, _, act_id, start_idx, end_idx = row

            # Use different colors to distinguish activities
            c = color_map[act_id]
            acc_plot.axvspan(start_idx, end_idx, color=c, alpha=0.8)

        acc_plot.set_title("Accelerometer for exp01 (all activities)")

        # Middle plot is for gyrometer data
        gyro_plot = fig.add_subplot(3,1,2)

        # Get the gyrometer data for x, y and z axis
        for i in range(3):
            color = ['red', 'green', 'blue']
            column_data = gyro_data[:, i]
            plt.plot(np.arange(len(column_data)), column_data, color=color[i])

        # Iterate over all rows (segments), plotting one small segment at a time.
        for row in labels_exp1:
            _, _, act_id, start_idx, end_idx = row

            # Use different colors to distinguish activities
            c = color_map[act_id]
            gyro_plot.axvspan(start_idx, end_idx, color=c, alpha=0.8)

        gyro_plot.set_title("Gyrometer for exp01 (all activities)")

        # Bottom plot is for the bar
        bar_plot = fig.add_subplot(3, 1, 3)

        num_acts = len(unique_acts)
        fraction_block_width = 1.0 / num_acts  # fraction of the subplot width per activity

        for i, act_id in enumerate(unique_acts):
            act_name = activity_dict.get(i+1)
            print(f"act_name:{act_name}")
            c = color_map[act_id]

            # Create a rectangle patch for each activity block
            rect = patches.Rectangle(
                (i * fraction_block_width, 0),  # Along the x-axis, the blocks are placed sequentially,
                                                    # with each block shifted by i * fraction_block_width from x = 0.
                                                    # The 0 indicates the rectangle starts at the very bottom of the subplot’s y-axis.
                fraction_block_width,  # width
                1.0,  # height (entire subplot)
                facecolor=c,
                edgecolor="black",
                alpha=0.8 # transparency
            )
            bar_plot.add_patch(rect)

            # Place the text label in the middle of the block
            bar_plot.text(
                (i + 0.5) * fraction_block_width, 0.5, act_name,
                ha="center", va="center", rotation=45, fontsize=8
            )# ha: horizotal alignment; va: vertical alignment

        bar_plot.set_xticks([])
        bar_plot.set_yticks([])

        # Adjust subplot spacing so the text at bottom is visible
        plt.subplots_adjust(hspace=0.8, bottom=0.1)

        # Save the plot to the path (folder_path_raw_data)
        if folder_path_raw_data:
            os.makedirs(folder_path_raw_data, exist_ok=True)
            file_path = os.path.join(folder_path_raw_data, "visualization_raw_data.png")
            plt.savefig(file_path)
            print(f"plot saved to:{file_path}")
        else:
            raise ValueError

        plt.show()

if __name__ == "__main__":
    data_path = '/home/RUS_CIP/st186731/DL_LAB_HAPT/DL_LAB_HAPT_DATASET/HAPT_Data_Set/RawData'
    all_files = '/home/RUS_CIP/st186731/DL_LAB_HAPT/DL_LAB_HAPT_DATASET/HAPT_Data_Set'
    acc_file = os.path.join(data_path, "acc_exp01_user01.txt")
    acc_data = np.loadtxt(acc_file)

    gyro_file = os.path.join(data_path, "gyro_exp01_user01.txt")
    gyro_data = np.loadtxt(gyro_file)
    folder_path_raw_data = '/home/RUS_CIP/st186731/DL_LAB_HAPT/visualization'

    visualize_data(oversample = False, dataset = None, data_path = data_path, all_files = all_files,
                   folder_path_raw_data = folder_path_raw_data, acc_data = acc_data,
                   gyro_data = gyro_data)

