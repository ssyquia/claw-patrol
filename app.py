import streamlit as st
import tensorflow as tf
import numpy as np
import keras
import requests
import os

# Google Drive file ID for the model
FILE_ID = "1_6w8NsbOKBRzEmikrbh-UN7PcU7EpM8c"
DESTINATION = "model/dino_classifier.keras"

# Function to download model from Google Drive
def download_file_from_google_drive(file_id, destination):
    URL = f"https://drive.google.com/uc?id={file_id}&export=download"
    response = requests.get(URL, stream=True)

    if response.status_code == 200:
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(32768):
                if chunk:
                    f.write(chunk)
        print(f"File downloaded to {destination}")
    else:
        print(f"Failed to download file from Google Drive, status code {response.status_code}")

# Define columns
c1, c2, c3 = st.columns(3)

# Display GIF
c2.image(
    "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExd3F0Mmd1YnY4dGg3a2VwbjYzcnZ4dDR4dHh2OWNqMXlkbTMxZjZ6MCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9cw/i4xCE6mf6XggfAVRpq/giphy.gif",
    width=250)

# Title and header
st.markdown(
    "<h1 style='text-align: center; color: black;'>Dinosaur Species Detector</h1>",
    unsafe_allow_html=True)

dino_names = [
    "Ankylosaurus", "Brachiosaurus", "Compsognathus", "Corythosaurus",
    "Dilophosaurus", "Dimorphodon", "Gallimimus", "Microceratus",
    "Pachycephalosaurus", "Parasaurolophus", "Spinosaurus", 
    "Stegosaurus", "Triceratops", "Tyrannosaurus_Rex", "Velociraptor"
]

# File uploader for images
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, use_column_width=True)

    # Classify uploaded image
    with st.spinner("Classifying..."):
        # Load the model
        model = keras.models.load_model(DESTINATION)

        # Preprocess the image
        img = tf.image.decode_jpeg(uploaded_file.read(), channels=3)
        img = tf.cast(img, tf.float32)
        img /= 255.0
        img = tf.image.resize(img, (224, 224))
        img = tf.expand_dims(img, axis=0)

        # Make prediction
        model_pred = model.predict(img)
        
        if len(model_pred) > 0:
            pred = dino_names[np.argmax(model_pred)]
            st.write("The dinosaur in the image is: ", pred)
        else:
            st.error("Could not classify the image. Try again with another image.")
