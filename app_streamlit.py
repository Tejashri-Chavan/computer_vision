import streamlit as st
import requests
import numpy as np
import cv2
from PIL import Image

st.title("Brain MRI Metastasis Segmentation")

# File uploader for MRI images
uploaded_file = st.file_uploader("Upload an MRI Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Read the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert image to bytes
    file_bytes = np.array(image).tobytes()
    
    # Call the FastAPI backend for prediction
    response = requests.post("http://127.0.0.1:8000/predict/", files={"file": (uploaded_file.name, file_bytes)})

    if response.status_code == 200:
        mask = np.frombuffer(response.json()["mask"], dtype=np.uint8)
        mask_image = cv2.imdecode(mask, cv2.IMREAD_GRAYSCALE)
        st.image(mask_image, caption="Segmentation Mask", use_column_width=True)
    else:
        st.error("Error: Could not get prediction from the server.")
