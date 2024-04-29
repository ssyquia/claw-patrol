import streamlit as st
import tensorflow as tf
import numpy as np
import keras

# Define columns
c1, c2, c3, c4, c5 = st.columns(5)

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

        model = keras.models.load_model("model/dino_classifier.keras")

        img = tf.image.decode_jpeg(uploaded_file.read(), channels=3)
        img = tf.cast(img, tf.float32)
        img /= 255.0
        img = tf.image.resize(img, (224, 224))
        img = tf.expand_dims(img, axis=0)
        model_pred = model.predict(img)
        
        if len(model_pred) > 0:
            pred = dino_names[np.argmax(model_pred)]
            st.write("The dinosaur in the image is: ", pred)
        else:
            st.error(
                "Could not classify the image. Try again with another image.")