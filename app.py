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


ALLOWED_FILE_TYPES = ["png", "jpg", "jpeg"]
IMAGES_PER_ROW = 3
IMAGE_SCREEN_PERCENTAGE = 1/IMAGES_PER_ROW*100
IMAGE_SIZE=180
UPLOAD_FOLDER = "uploaded_images"


def predict_image(model, image):
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
    transformed_image = image_tensor(image)

    # Make prediction
    with torch.no_grad():
        outputs = model(transformed_image)
        softmax = nn.Softmax(dim=1)
        softmax_outputs = softmax(outputs)
        print(softmax_outputs)
        _, prediction = torch.max(softmax_outputs, 1)
        #predicted_prob = torch.sigmoid(outputs).item()
    
    return prediction



def main():
    model = torch.load("trained_model_32_resize_moreTransformations_acc82.pth")
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
                for file in row_files:
                    # Save the uploaded file to the folder
                    file_path = os.path.join(UPLOAD_FOLDER+"/", file.name)
                    with open(file_path, "wb") as f:
                        f.write(file.read())
                    # You can perform further processing on each uploaded file here
                    img = cv2.imread(UPLOAD_FOLDER+"/"+file.name)
                    image_to_pred = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
                    image_to_pred = Image.fromarray(cv2.cvtColor(image_to_pred, cv2.COLOR_BGR2RGB))
                    """image_to_pred = Image.open(file)
                    if image_to_pred.mode != 'RGB':
                        image_to_pred = image_to_pred.convert('RGB')"""
                    #print(image_to_pred.size())
                    predicted_prob = predict_image(model=model, image=image_to_pred)
                    image_data = file.read()
                    image_base64 = base64.b64encode(image_data).decode("utf-8")
                
                    # Use the base64 image data in the img tag
                    row_html += f'<img src="data:image/png;base64,{image_base64}" alt="Prediction: {55.58964:.2f}" width="{IMAGE_SCREEN_PERCENTAGE}%" style="margin: 5px;">'
                    # Add text under the image
                    #row_html += f'<p>X-Ray [{i+1}]</p>'  # Change the description accordingly
                row_html += '</div>'
                
                # Display the row
                st.markdown(row_html, unsafe_allow_html=True)
        else:
            st.write("No X-Rays uploaded.")

if __name__ == "__main__":
    main()