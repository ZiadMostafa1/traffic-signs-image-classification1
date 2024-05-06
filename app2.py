import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

import os

if os.path.isfile('traffic_classifier.h5'):
    print("File exists")
else:
    print("File does not exist")

# Load the saved model
model = load_model('traffic_classifier.h5')

st.title('Traffic Sign Classifier')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file)

    # Convert the image to RGB
    image = img.convert('RGB')
    
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    image = image.resize((30,30))  # Resize image
    image = np.expand_dims(image, axis=0)  # Expand dimensions
    image = np.array(image)  # Convert image to numpy array

    # Make a prediction
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions)
    st.write("Class: ", predicted_class)