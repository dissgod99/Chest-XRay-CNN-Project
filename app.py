import streamlit as st
import base64
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from random import shuffle
from typing import Callable
import cv2
from PIL import Image, ImageDraw, ImageFont
from torchsummary import summary
from torchviz import make_dot
import graphviz
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from io import BytesIO
from CustomCNN import CNNModel
import numpy as np
import matplotlib.pyplot as plt
import re

ALLOWED_FILE_TYPES = ["png", "jpg", "jpeg"]
IMAGES_PER_ROW = 4
IMAGE_SCREEN_PERCENTAGE = 1/IMAGES_PER_ROW*100
IMAGE_SIZE=180
UPLOAD_FOLDER = "uploaded_images"
SALIENCY_FOLDER = "saliency_images"
PREDICTION_FOLDER = "prediction_images"
RESULTS_FOLDER = "results"

def empty_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            # Recursively remove subdirectories and their contents
            empty_folder(file_path)
            os.rmdir(file_path)


def predict_image(model, image_rgb):
    #Part1: Preprocessing
    image_tensor=transforms.Compose([
        transforms.RandomRotation(30),  # Randomly rotate images in the range (degrees, 0 to 180)
        transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
        transforms.RandomVerticalFlip(),  # Randomly flip images vertically
        transforms.Resize(size=(IMAGE_SIZE, IMAGE_SIZE)), #resize image
        #transforms.RandomResizedCrop(size=(new_size, new_size)),  # Randomly crop and resize images
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),  # Randomly adjust brightness, contrast, and saturation
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
    ])
    #image is the result of Image.open(file)
    transformed_image = image_tensor(image_rgb)
    #Add one dimension to use the image in the model with batch size 1
    transformed_image = transformed_image[np.newaxis, :]

    # Make prediction
    with torch.no_grad():
        outputs = model(transformed_image)
        softmax = nn.Softmax(dim=1)
        softmax_outputs = softmax(outputs)
        _, prediction = torch.max(softmax_outputs, 1)
        if prediction.item() == 0:
            label = "NORMAL"
        else:
            label = "PNEUMONIA"
        probability_majority_class = softmax_outputs.squeeze(0)[torch.argmax(softmax_outputs)].item()
    
    return prediction.item(), label, probability_majority_class

#Code based on the following article: 
#https://medium.datadriveninvestor.com/visualizing-neural-networks-using-saliency-maps-in-pytorch-289d8e244ab4
def calculate_saliency(model, img_path: str, filename:str):

    image_tensor=transforms.Compose([
        transforms.RandomRotation(30),  # Randomly rotate images in the range (degrees, 0 to 180)
        transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
        transforms.RandomVerticalFlip(),  # Randomly flip images vertically
        transforms.Resize(size=(IMAGE_SIZE, IMAGE_SIZE)), #resize image
        #transforms.RandomResizedCrop(size=(new_size, new_size)),  # Randomly crop and resize images
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),  # Randomly adjust brightness, contrast, and saturation
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
    ])


    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img_rgb = Image.fromarray(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
    transformed_image = image_tensor(img_rgb)
    image = transformed_image.unsqueeze(0) # Add batch dimension and move to appropriate device
    image.requires_grad_()

    scores = model(image)
    # Get the index corresponding to the maximum score and the maximum score itself.
    score_max_index = scores.argmax()
    score_max = scores[0,score_max_index]
    score_max.backward()

    saliency, _ = torch.max(image.grad.data.abs(),dim=1)
    # code to plot the saliency map as a heatmap
    plt.imshow(saliency[0], cmap=plt.cm.hot)
    plt.axis('off')
    #plt.colorbar()  # Add a colorbar for reference
    # Save the plot as a JPEG image
    heat_name = filename
    saliency_path = os.path.join('saliency_images', f'{filename}_saliency_plot.jpeg')
    plt.savefig(saliency_path, format='jpeg', bbox_inches='tight', pad_inches=0.05)
    
    return saliency[0]


def download_images(output_folder_name:str,
                    images_path="uploaded_images/", 
                    saliency_path="saliency_images/",
                    predictions_images="prediction_images/"):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder_name):
        os.makedirs(output_folder_name)
    
    # Iterate through the images in the directory and copy them to the output folder
    for idx, filename in os.listdir(images_path):
        pass
    pass


def merge_xray_saliency_prediction(x_ray_filename:str, 
                                   saliency_filename:str, 
                                   prediction_filename:str, 
                                   download_directory="results/",
                                   x_ray_directory=UPLOAD_FOLDER+"/",
                                   saliency_directory=SALIENCY_FOLDER+"/",
                                   prediction_directory=PREDICTION_FOLDER+"/",
                                   size=350):
    # Open the two JPEG images
    image_xray = Image.open(x_ray_directory + x_ray_filename).resize((size, size))
    image_saliency = Image.open(saliency_directory + saliency_filename).resize((size, size))
    title = prediction_filename
    # Get the dimensions of the images
    width1, height1 = image_xray.size
    width2, height2 = image_saliency.size

    # Calculate the width and height of the merged image
    merged_width = width1 + width2
    merged_height = max(height1, height2) + 50  # Increase height to accommodate title

    # Create a new blank image with the calculated dimensions
    merged_image = Image.new("RGB", (merged_width, merged_height))

    # Paste the first image on the left side of the merged image
    merged_image.paste(image_xray, (0, 0))

    # Paste the second image on the right side of the merged image
    merged_image.paste(image_saliency, (width1, 0))

    # Draw the title text at the bottom middle of the merged image
    #title = prediction

    pattern = r'(\d+\.\d+)%_(\w+)'

    matches = re.search(pattern, prediction_directory + prediction_filename)
    if matches:
        percentage = matches.group(1) + "%"
        label = matches.group(2)
    title = percentage + " " + label
    draw = ImageDraw.Draw(merged_image)
    font = ImageFont.truetype("arial.ttf", 25)  # Load a suitable font
    text_width, text_height = draw.textsize(title, font=font)
    text_x = (merged_width - text_width) // 2
    text_y = merged_height - text_height - 10  # Position above the bottom edge
    draw.text((text_x, text_y), title, font=font, fill=(255, 255, 255))  # Fill with white color

    # Save the merged image with the title as a new JPEG file
    merged_image.save(f"{download_directory}{x_ray_filename}_Results.jpg")

def trigger_download():
    #output_folder_name = st.text_input("Enter output folder name:", "downloaded_images")
    for idx, filename in enumerate(os.listdir(UPLOAD_FOLDER)):
        merge_xray_saliency_prediction(x_ray_filename=filename,
                                    saliency_filename=os.listdir(SALIENCY_FOLDER)[idx],
                                    prediction_filename=os.listdir(PREDICTION_FOLDER)[idx])


def main():
    #model = torch.load("trained_model_32_resize_moreTransformations_acc82.pth")
    model = CNNModel(num_classes=2)

    # Load the trained parameters into the model
    model.load_state_dict(torch.load('trained_model_32_resize_moreTransformations_acc82.pth'))
    model.eval()  # Set the model to evaluation mode
    
    print("Model Successfully Loaded!")
    st.set_page_config(page_title="X-Ray Anomalert")
    st.header("X-Ray Anomalert")
    #st.text_input("Let's analyze your Chest X-Ray")
    
    # Create Image Folder 
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    
    # Create Saliency Folder
    if not os.path.exists(SALIENCY_FOLDER):
        os.makedirs(SALIENCY_FOLDER)
    
    if not os.path.exists(PREDICTION_FOLDER):
        os.makedirs(PREDICTION_FOLDER)

    with st.sidebar:
        st.subheader("Your X-Rays")
        uploaded_files = st.file_uploader("Upload your Chest X-Rays and click on 'Process'",
                         accept_multiple_files=True,
                         type=ALLOWED_FILE_TYPES)
        process_button = st.button("Process")
        download_button = st.button("Download All Files", disabled=False if process_button else True,
                                    on_click=trigger_download)

    # Check if the "Process" button is clicked
    if process_button:
        if uploaded_files:
            st.write("Number of X-Rays uploaded:", len(uploaded_files))
            print(uploaded_files)
            # Iterate through the uploaded files and display images in rows
            counter_4_predictions = 0
            for i in range(0, len(uploaded_files), IMAGES_PER_ROW):
                row_files = uploaded_files[i:i+IMAGES_PER_ROW]
                
                # Create a row using HTML and CSS styling
                row_html = '<div style="display: flex; ">'
                for idx, file in enumerate(row_files):
                    # Save the uploaded file to the folder
                    file_path = os.path.join(UPLOAD_FOLDER, file.name)
                    #print(file.name)
                    #print(file.name[:-5])
                    with open(file_path, "wb") as f:
                        f.write(file.read())
                    # You can perform further processing on each uploaded file here
                    img = cv2.imread(file_path)
                    image_resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
                    image_rgb_to_pred = Image.fromarray(cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB))
                    

                    saliency_value = calculate_saliency(model=model, img_path=file_path, filename=file.name[:-5])
                    file_path_saliency = os.path.join(SALIENCY_FOLDER, file.name[:-5]+"_saliency_plot.jpeg")

                    #print(image_to_pred.size())
                    prediction, label, probability_majority_class = predict_image(model=model, image_rgb=image_rgb_to_pred)
                    probability_rounded = round(probability_majority_class*100, 2)
                    file_path_predictions = os.path.join(PREDICTION_FOLDER, f"{counter_4_predictions}_{probability_rounded}%_{label}")
                    with open(file_path_predictions, "wb") as f_predictions:
                        #f_predictions.write()
                        counter_4_predictions += 1

                    # Calculate the saliency map for the current image
                    #saliency_map = calculate_saliency_map(image_rgb_to_pred, model)
                    print(f"({i}) Label: {label}, Probability: {probability_majority_class*100:.2f}%")
                    
                    # Convert the image to base64
                    with open(file_path, "rb") as f:
                        image_data = f.read()
                    image_base64 = base64.b64encode(image_data).decode("utf-8")

                    # Convert the image to base64
                    with open(file_path_saliency, "rb") as f_saliency:
                        image_data_saliency = f_saliency.read()
                    image_base64_saliency = base64.b64encode(image_data_saliency).decode("utf-8")
                    
                    # Create a container div for each image and its prediction
                    row_html += f'<div style="display: inline-block; width: {IMAGE_SCREEN_PERCENTAGE}%; text-align: center; margin: 5px;">'

                    # Add the image with the corresponding prediction and label
                    row_html += f'<img src="data:image/png;base64,{image_base64}" alt="Prediction: {label}: {probability_majority_class*100:.2f}%" style="width: 100%; margin: 5px">'
                    row_html += f'<img src="data:image/png;base64,{image_base64_saliency}" alt="Saliency Map" style="width: 100%;">'
                    row_html += f'<p style="font-size: 12px; margin-top: 5px;">{probability_majority_class*100:.2f}% {label}</p>'

                    # Close the container div
                    row_html += '</div>'
                row_html += '</div>'
                
                # Display the row
                st.markdown(row_html, unsafe_allow_html=True)
                #plt.colorbar()
                
        else:
            st.write("No X-Rays uploaded.")
    

if __name__ == "__main__":
    empty_folder(UPLOAD_FOLDER)
    empty_folder(SALIENCY_FOLDER)
    empty_folder(PREDICTION_FOLDER)
    #empty_folder(RESULTS_FOLDER)
    main()
    #if download_images:
    #    trigger_download()