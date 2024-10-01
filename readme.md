# Brain MRI Metastasis Segmentation

## Overview
This project demonstrates the use of computer vision techniques for segmenting brain MRI metastases using deep learning architectures: Nested U-Net and Attention U-Net. A web application is developed to showcase the model's capabilities, allowing users to upload MRI images and visualize segmentation results.

## Dataset
- **Source**: The dataset consists of brain MRI images and their corresponding metastasis segmentation masks.
- **Link**: [Dataset Download](https://dicom5c.blob.core.windows.net/public/Data.zip)
- **Structure**: The dataset is split into 80% training and 20% testing sets. Images without corresponding masks are ignored.

## Technologies Used
- Python
- FastAPI
- Streamlit
- TensorFlow/Keras
- OpenCV
- NumPy
- Pillow

## Project Structure
