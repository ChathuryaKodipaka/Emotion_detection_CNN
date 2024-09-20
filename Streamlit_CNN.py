import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import gdown

# Define the Google Drive URL for your model
file_id = '1-Xoepf6oNhh3QwGi0EnrRXiqlZxblcg8'
download_url = f"https://drive.google.com/uc?id={file_id}"

# Download the model file from Google Drive
model_path = 'cnn_model.keras'
gdown.download(download_url, model_path, quiet=False)

# Load the trained model
model = tf.keras.models.load_model(model_path)

# Define class labels
class_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Streamlit app title
st.title('Image Classification with CNN and OpenCV')

# Instructions
st.write("Upload an image to classify it using the trained CNN model with OpenCV preprocessing.")

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Preprocessing function using OpenCV
def preprocess_image_opencv(image):
    # Convert the image to an OpenCV format (numpy array)
    image = np.array(image)
    
    # Convert to RGB (Streamlit handles PIL images in RGB format)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize to 48x48 (for your model)
    image_resized = cv2.resize(image, (48, 48))
    
    # Normalize the image
    image_resized = image_resized / 255.0
    
    # Add batch dimension (1, 48, 48, 3)
    image_expanded = np.expand_dims(image_resized, axis=0)
    
    return image_expanded

# Make prediction if an image is uploaded
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image using OpenCV
    img = preprocess_image_opencv(image)
    
    # Make prediction
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)[0]

    # Display the result
    st.write(f"Predicted Class: {class_labels[predicted_class]}")
