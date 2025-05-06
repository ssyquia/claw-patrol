import streamlit as st
import tensorflow as tf
import numpy as np
import keras
# import requests # No longer needed for download
import os
import gdown # Import gdown

# Google Drive file ID for the model
FILE_ID = "16R2ffLu12dBeqhlq1Em8DAH1DuHy-27X"
DESTINATION = "model/dino_classifier.keras"

# Function to download model from Google Drive using gdown
def download_file_from_google_drive(file_id, destination):
    # Ensure the destination directory exists
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    # Construct the full Google Drive URL
    url = f'https://drive.google.com/uc?id={file_id}'
    try:
        print(f"Attempting to download model from {url} to {destination}")
        # Use gdown.download - quiet=False shows progress, fuzzy=True helps with URLs
        gdown.download(url, destination, quiet=False, fuzzy=True)
        
        # Verify download by checking file existence
        if os.path.exists(destination):
            print(f"File downloaded successfully to {destination}")
            return True
        else:
            print(f"Download command finished, but file not found at {destination}")
            return False
            
    except Exception as e:
        print(f"Error during gdown download: {e}")
        # Optionally, try to remove partially downloaded file
        if os.path.exists(destination):
            try:
                os.remove(destination)
                print(f"Removed potentially incomplete file: {destination}")
            except OSError as rm_err:
                print(f"Error removing file {destination}: {rm_err}")
        return False

# --- Download Model at Startup (if needed) ---
if not os.path.exists(DESTINATION):
    st.info(f"Model not found locally. Downloading from Google Drive (ID: {FILE_ID})...")
    with st.spinner("Downloading model... This may take a moment (approx. 350MB)."):
        download_successful = download_file_from_google_drive(FILE_ID, DESTINATION)
        if not download_successful:
            st.error("Failed to download the model. Please check the file ID, network connection, and Google Drive link permissions.")
            st.stop() # Stop execution if download fails
        else:
            st.success("Model downloaded successfully!")
else:
    st.info("Model found locally.")

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
    st.image(uploaded_file, use_container_width=True)

    # Classify uploaded image
    with st.spinner("Classifying..."):
        # Load the model
        # Check again if the file exists before loading, just in case
        if os.path.exists(DESTINATION):
            try:
                model = keras.models.load_model(DESTINATION)

                # Preprocess the image
                img = tf.image.decode_image(uploaded_file.read(), channels=3) # Use decode_image for flexibility
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
                    st.error("Prediction array was empty. Could not classify the image.")
            except Exception as e:
                st.error(f"Error loading model or classifying image: {e}")
        else:
            st.error("Model file not found. Cannot classify image. Please ensure the download completed successfully.")
