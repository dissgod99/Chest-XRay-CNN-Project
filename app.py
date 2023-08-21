import streamlit as st
import base64
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from random import shuffle
from typing import Callable
import cv2
from PIL import Image
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

ALLOWED_FILE_TYPES = ["png", "jpg", "jpeg"]
IMAGES_PER_ROW = 3
IMAGE_SCREEN_PERCENTAGE = 1/IMAGES_PER_ROW*100
IMAGE_SIZE=180
UPLOAD_FOLDER = "uploaded_images"

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
        #print(softmax_outputs)
        #_, prediction = torch.max(softmax_outputs, 1)
        #softmax = nn.Softmax(dim=1)
        #softmax_outputs = softmax(prob)
        #print(softmax_outputs)
        _, prediction = torch.max(softmax_outputs, 1)
        if prediction.item() == 0:
            label = "NORMAL"
        else:
            label = "PNEUMONIA"
        probability_majority_class = softmax_outputs.squeeze(0)[torch.argmax(softmax_outputs)].item()
    
    return prediction.item(), label, probability_majority_class



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
    

    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    with st.sidebar:
        st.subheader("Your X-Rays")
        uploaded_files = st.file_uploader("Upload your Chest X-Rays and click on 'Process'",
                         accept_multiple_files=True,
                         type=ALLOWED_FILE_TYPES)
        process_button = st.button("Process")

    # Check if the "Process" button is clicked
    if process_button:
        if uploaded_files:
            st.write("Number of X-Rays uploaded:", len(uploaded_files))
            print(uploaded_files)
            # Iterate through the uploaded files and display images in rows
            for i in range(0, len(uploaded_files), IMAGES_PER_ROW):
                row_files = uploaded_files[i:i+IMAGES_PER_ROW]
                
                # Create a row using HTML and CSS styling
                row_html = '<div style="display: flex; ">'
                for idx, file in enumerate(row_files):
                    # Save the uploaded file to the folder
                    file_path = os.path.join(UPLOAD_FOLDER, file.name)
                    with open(file_path, "wb") as f:
                        f.write(file.read())
                    # You can perform further processing on each uploaded file here
                    img = cv2.imread(file_path)
                    image_resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
                    image_rgb_to_pred = Image.fromarray(cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB))
                    
                    #print(image_to_pred.size())
                    prediction, label, probability_majority_class = predict_image(model=model, image_rgb=image_rgb_to_pred)
                    print(f"({i}) Label: {label}, Probability: {probability_majority_class:.2f}%")
                    #print(predicted_prob)
                    # Convert the image to base64
                    with open(file_path, "rb") as f:
                        image_data = f.read()
                    image_base64 = base64.b64encode(image_data).decode("utf-8")
                    row_html += f'<div>Prediction: {label}</div>'
                    row_html += f'<div>Probability: {probability_majority_class:.2f}%</div>'
                    # Use the base64 image data in the img tag
                    row_html += f'<img src="data:image/png;base64,{image_base64}" alt="Prediction: {label}: {probability_majority_class:.2f}%" width="{IMAGE_SCREEN_PERCENTAGE}%" style="margin: 5px;">'
                    # Add text under the image
                    #row_html += f'<p>X-Ray [{i+1}]</p>'  # Change the description accordingly
                row_html += '</div>'
                
                # Display the row
                st.markdown(row_html, unsafe_allow_html=True)
        else:
            st.write("No X-Rays uploaded.")

if __name__ == "__main__":
    empty_folder(UPLOAD_FOLDER)
    main()
    