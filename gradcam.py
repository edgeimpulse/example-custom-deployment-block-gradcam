### Dataset Bias and Grad-CAM Visualization with Convolutional Neural Networks

# **1. Setup and Introduction**

# Install required libraries
# !pip install tensorflow==2.11 opencv-python requests

# Import necessary modules
import os
import numpy as np
import time
import json
import sys
import argparse
import requests
import zipfile
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model, load_model
import cv2


# **2. Download Dataset and Model from Edge Impulse**
parser = argparse.ArgumentParser(description="Grad-CAM deployment block")
parser.add_argument('--api-key', type=str, required=False, help="Edge Impulse Studio API Key")
parser.add_argument('--metadata', type=str, required=False, help="Deployment metadata json file")
parser.add_argument('--alpha', type=float, default=0.4, required=False, help="Alpha value")
parser.add_argument('--pooling-gradients', type=str, default="sum_abs", choices=["mean", "sum_abs"], help="Method to pool gradients for Grad-CAM ('mean' or 'sum_abs')")
parser.add_argument('--heatmap-normalization', type=str, default="percentile", choices=["percentile", "simple"], help="Method to normalize the Grad-CAM heatmap ('percentile' or 'simple')")
parser.add_argument('--skip-dataset-download', action='store_true', help="Skip dataset download")
args = parser.parse_args()

# Define Edge Impulse API credentials
ei_api_key = None
alpha=args.alpha
pooling_gradients = args.pooling_gradients
heatmap_normalization = args.heatmap_normalization

# Get EI API Key
if args.metadata:
    with open(args.metadata) as f:
        metadata = json.load(f)
        ei_api_key = metadata['project']['apiKey']
        ei_project_name = metadata['project']['name']
        output_folder = metadata['folders']['output']
        print(f"Retrieved {ei_project_name} project's metadata...")
else:
    print('No metadata.json file found, will use --api-key')
    if args.api_key is None:
        print("Error: --api-key not set")
        sys.exit(1)
    ei_api_key = args.api_key
    output_folder = None


# Define the base URL and headers
base_url = "https://studio.edgeimpulse.com/v1/api"
headers = {
    "x-api-key": ei_api_key,
    "Content-Type": "application/json"
}

# Retrieve project information to extract class names
def get_project_id():
    url = f"{base_url}/projects"
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()

project_id = get_project_id()
EI_PROJECT_ID = str(project_id["projects"][0]["id"])
print(f"Project ID: {EI_PROJECT_ID}")

# Set the output folder if it was not already defined
if not output_folder:
    output_folder = f"output_{EI_PROJECT_ID}"


# Retrieve the Impulse information to extract learn block ID
def get_impulse_info(project_id):
    url = f"{base_url}/{project_id}/impulse"
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()

impulse_info = get_impulse_info(EI_PROJECT_ID)
learn_block_id = impulse_info["impulse"]["learnBlocks"][0]["id"]
print(f"Retrieved learn block ID: {learn_block_id}")

# Retrieve project information to extract class names
def get_project_info(project_id):
    url = f"{base_url}/{project_id}"
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()

project_info = get_project_info(EI_PROJECT_ID)
classes = project_info["dataSummaryPerCategory"]["training"]["labels"]
print(f"Retrieved class names: {classes}")

# Create export job for dataset
def create_export_job(project_id):
    url = f"{base_url}/{project_id}/jobs/export/original"
    response = requests.post(url, headers=headers, json={"uploaderFriendlyFilenames":True,"retainCrops":True})
    response.raise_for_status()
    return response.json()

# Get export URL for dataset with a retry mechanism
def get_export_url(project_id, max_retries=20, delay=10):
    print(f"Retrieving dataset export URL...")
    url = f"{base_url}/{project_id}/export/get-url"
    for attempt in range(max_retries):
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            json_response = response.json()
            if "url" in json_response:
                return json_response["url"]
        print(f"Attempt {attempt + 1} failed. URL not ready... Retrying in {delay} seconds...")
        # Start export job and retrieve dataset URL
        if(attempt==0):
            print(f"Creating export job for to retrieve the dataset")
            create_export_job(EI_PROJECT_ID)
        time.sleep(delay)
    raise RuntimeError("Failed to retrieve export URL after multiple attempts.")


# Download the dataset
def download_dataset(url, project_name):
    print(f"Downloading dataset...")
    response = requests.get(url)
    response.raise_for_status()
    
    file_path = os.path.join(output_folder, f"{project_name}.zip")
    os.makedirs(output_folder, exist_ok=True)
    with open(file_path, 'wb') as file:
        file.write(response.content)
    print(f"Downloaded dataset for project '{project_name}' to '{file_path}'")
    return file_path


if not args.skip_dataset_download:
    dataset_url = get_export_url(EI_PROJECT_ID)
    dataset_zip_path = download_dataset(dataset_url, "edge_impulse_dataset")

    # Extract the dataset
    dataset_dir = os.path.join(output_folder, "edge_impulse_dataset")
    print(dataset_dir)
    os.makedirs(dataset_dir, exist_ok=True)
    with zipfile.ZipFile(dataset_zip_path, 'r') as zip_ref:
        zip_ref.extractall(dataset_dir)
else:
    print("Skipping dataset download as per --skip-dataset-download flag.")
    dataset_dir = os.path.join(output_folder, "edge_impulse_dataset")
    print(dataset_dir)

# Download the trained model
def get_model_url(project_id, learn_block_id):
    url = f"{base_url}/{project_id}/learn-data/{learn_block_id}/model/tflite-h5"
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    file_path = output_folder + "/model-h5.zip"
    with open(file_path, "wb") as f:
        f.write(response.content)
    print(f"Model downloaded and saved to {file_path}")
    return file_path

# Unzip the model
model_zip_path = get_model_url(EI_PROJECT_ID, learn_block_id)
model_dir = output_folder + "/edge-impulse-model"
os.makedirs(model_dir, exist_ok=True)
with zipfile.ZipFile(model_zip_path, 'r') as zip_ref:
    zip_ref.extractall(model_dir)

# Locate the .h5 model file
model_file_path = os.path.join(model_dir, "model.h5")
if not os.path.exists(model_file_path):
    raise FileNotFoundError("The extracted model does not contain the expected .h5 file.")

print("Model and dataset prepared successfully!")

# **3. Preprocessing and Model Setup**

# Load your pre-trained model
model = load_model(model_file_path, compile=False)

# Get the input size from the model
input_shape = model.input_shape
input_size = (input_shape[1], input_shape[2])  # Assumes input shape is (batch_size, height, width, channels)

# Get the number of output categories from the model
output_shape = model.output_shape
num_categories = output_shape[-1]

# Assuming class names are stored in the project info
class_names = classes if num_categories > 1 else None  # Handle regression models

# Function to find the last convolutional layer dynamically
def find_last_conv_layer(model):
    if isinstance(model, tf.keras.Model) or isinstance(model, tf.keras.Sequential):
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer.name
            if hasattr(layer, 'layers'):  # Handle nested models
                nested_layer = find_last_conv_layer(layer)
                if nested_layer:
                    return nested_layer
    raise ValueError("No convolutional layer found in the model.")

# Find the last convolutional layer dynamically
last_conv_layer_name = find_last_conv_layer(model)

# Grad-CAM model
def create_grad_model(model, last_conv_layer_name):
    if last_conv_layer_name not in [layer.name for layer in model.layers]:
        # Check if the model contains nested models
        for layer in model.layers:
            if hasattr(layer, 'layers'):
                nested_model = layer
                break
        else:
            raise ValueError("Nested model not found, and last conv layer is invalid.")
        base_model = nested_model
    else:
        base_model = model

    return Model(
        [base_model.input],
        [base_model.get_layer(last_conv_layer_name).output, base_model.output]
    )

grad_model = create_grad_model(model, last_conv_layer_name)

# Function to preprocess an image
def preprocess_image(img_path):
    try:
        img = image.load_img(img_path, target_size=input_size)  # Ensure target size matches model input size
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img / 255.0  # Normalize
        return img
    except Exception as e:
        print(f"Error processing image {img_path}: {e}")
        return None

# **4. Grad-CAM Implementation**

# Grad-CAM implementation
@tf.function
def make_gradcam_heatmap(img_array, grad_model, pooling_gradients, heatmap_normalization):
    with tf.GradientTape() as tape:
        # Compute predictions and activations
        last_conv_layer_output, preds = grad_model(img_array)
        
        if class_names:
            # Classification model
            pred_index = tf.argmax(preds[0])  # Extract scalar index of the predicted class
            pred_index = tf.cast(pred_index, tf.int32)  # Ensure it's an integer
            class_channel = tf.gather(preds[0], pred_index)  # Gather the value for the predicted class
        else:
            # Regression model (single output)
            class_channel = preds[:, 0]  # Use the single regression output

    # Calculate gradients with respect to the last convolutional layer
    grads = tape.gradient(class_channel, last_conv_layer_output)  # <-- This line is critical

    # Pool gradients based on the selected method
    if pooling_gradients == "mean":
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # Mean pooling
    elif pooling_gradients == "sum_abs":
        pooled_grads = tf.reduce_sum(tf.abs(grads), axis=(0, 1, 2))  # Sum of absolute gradients
    else:
        raise ValueError("Invalid pooling-gradients method. Choose 'mean' or 'sum_abs'.")

    # Create heatmap
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize the heatmap based on the selected method
    if heatmap_normalization == "percentile":
        heatmap = tf.maximum(heatmap, 0)  # ReLU to remove negatives
        max_value = tf.reduce_max(heatmap)
        heatmap = heatmap / max_value if max_value != 0 else heatmap  # Avoid division by zero
    elif heatmap_normalization == "simple":
        heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)  # Simple normalization
    else:
        raise ValueError("Invalid heatmap-normalization method. Choose 'percentile' or 'simple'.")

    return heatmap


# Function to display and save Grad-CAM heatmap
def display_and_save_gradcam(img_path, heatmap, output_dir, alpha=alpha):
    img = cv2.imread(img_path)
    img = cv2.resize(img, input_size)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * alpha + img

    output_path = os.path.join(output_dir, os.path.basename(img_path))
    cv2.imwrite(output_path, superimposed_img)

# **5. Grad-CAM Visualization on Dataset**

test_set_dir = os.path.join(dataset_dir, "testing") 
correct_dir = output_folder + "/gradcam/correct"
incorrect_dir = output_folder + "/gradcam/incorrect"
os.makedirs(correct_dir, exist_ok=True)
os.makedirs(incorrect_dir, exist_ok=True)

for img_name in [f for f in os.listdir(test_set_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]:  # Filter only image files
    img_path = os.path.join(test_set_dir, img_name)

    # Dynamically determine the true class from the file name (label before the first dot)
    true_class = img_name.split('.')[0]

    img_array = preprocess_image(img_path)
    if img_array is None:
        print(f"Skipping image {img_name} due to preprocessing error.")
        continue  # Skip this image
    preds = model.predict(img_array)

    if class_names:
        # Classification
        predicted_class = class_names[np.argmax(preds)]
        grad_model = create_grad_model(model, last_conv_layer_name)
        heatmap = make_gradcam_heatmap(
            img_array, 
            grad_model, 
            pooling_gradients=pooling_gradients, 
            heatmap_normalization=heatmap_normalization
        ).numpy()

        # Determine the output directory based on prediction correctness
        if predicted_class == true_class:
            output_dir = correct_dir
        else:
            output_dir = incorrect_dir

    else:
        # Regression
        predicted_value = preds[0][0]
        error = abs(predicted_value - float(true_class))  # Assuming filenames contain true regression values
        heatmap = make_gradcam_heatmap(
            img_array, 
            grad_model, 
            pooling_gradients=pooling_gradients, 
            heatmap_normalization=heatmap_normalization
        ).numpy()

        # Determine the output directory based on prediction correctness
        threshold = 0.1  # Define a threshold for acceptable error
        if error <= threshold:
            output_dir = correct_dir
        else:
            output_dir = incorrect_dir

    # Save the Grad-CAM image
    display_and_save_gradcam(img_path, heatmap, output_dir)

print("Grad-CAM visualizations completed.")

# Final step: Create a zip archive of the output folder
shutil.make_archive(base_name=os.path.join(output_folder, 'deploy'), format='zip', root_dir=os.path.join(output_folder, 'gradcam'))
