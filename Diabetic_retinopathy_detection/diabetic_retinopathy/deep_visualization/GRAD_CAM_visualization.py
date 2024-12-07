import gin
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt


@gin.configurable
def grad_cam_visualization(model, img_path):

    def find_target_layer(model):
        for layer in reversed(model.layers):
            # check to see if the layer has a 4D output
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer.name

    last_conv_layer_name = find_target_layer(model)

    if last_conv_layer_name:
        print(f"The last conv layer name is {last_conv_layer_name}")
    else:
        print(f"No Convolutional Layers found in the model")

    # Function to generate Grad-CAM
    def grad_cam(model, img_path, last_conv_layer_name):

        # Load and preprocess the image
        image = cv2.imread(img_path)  # Load image in BGR format

        if image is None:
            print(f"Error: File not found or cannot be opened at {img_path}")
        else:
            print("Image loaded successfully!")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB format
        image = np.array(image) / 255.0  # Normalize the pixel values to [0, 1]
        img_array = np.expand_dims(image, axis=0)  # Add batch dimension
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

        # Predict using the model
        predictions = model(img_array, training=False)
        print("Predicted class:", 1 if predictions[0] > 0.5 else 0)

        # Initialize a model for Grad-CAM
        grad_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=[model.get_layer(last_conv_layer_name).output, model.output])

        with tf.GradientTape() as tape:
            # Ensure that the input tensor is watched
            tape.watch(img_tensor)

            # Get outputs for the last conv layer and the predictions
            conv_outputs, predictions = grad_model(img_array)

            # Extract the target class score
            class_output = predictions[:, 0]

        # Calculate gradients of the target class score with respect to conv layer output
        grads = tape.gradient(class_output, conv_outputs)
        print(f"grads shape is {grads.shape}")

        # Compute the mean intensity of the gradients for each feature map
        weights = tf.reduce_mean(grads, axis=(0, 1, 2))  # along, B, H, W
        print(f"weight shape is {weights.shape}")

        # Compute the weighted sum of the feature maps
        cam = tf.reduce_sum(tf.multiply(weights, conv_outputs[0]), axis=-1)

        # Apply ReLU to retain only positive values
        cam = tf.maximum(cam, 0)

        # Normalize the heatmap
        cam /= tf.reduce_max(cam) if tf.reduce_max(cam) != 0 else 1

        return cam

    # Function to display Grad-CAM
    def display_grad_cam(heatmap, img, alpha=0.5):

        # Load the image
        image = cv2.imread(img_path)  # Load image in BGR format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

        # Normalize for Matplot lib
        img = np.array(image) / 255.0

        if isinstance(heatmap, tf.Tensor):
            heatmap = heatmap.numpy()

        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # resize to. (4,4) to (256,256)

        if heatmap.shape[:2] != img.shape[:2]:
            raise ValueError("The heatmap and image must have the same spatial dimensions.")

        # Overlay heatmap on original image
        colormap = plt.cm.jet(heatmap)[:, :,
                   :3]  # convert the heat map to 3 channel, extract the first 3 channel , ignore the transparancy channel, (256,256,3)

        blended_image = alpha * colormap + (1 - alpha) * img

        # Display images
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(img)
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("Blended Image")
        plt.imshow(blended_image)
        plt.axis("off")

        plt.tight_layout()
        plt.show()


    # Generate Grad-CAM heatmap
    heatmap = grad_cam(model, img_path, last_conv_layer_name)

    # Visualize the result
    display_grad_cam(heatmap, img_path)



