import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
import tensorflow.keras.backend as K
import tensorflow as tf
import glob

# Step 1: Data Preprocessing
def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def load_images_and_masks(image_path, mask_path):
    images = []
    masks = []

    image_files = sorted(glob.glob(os.path.join(image_path, "*.png")))  # Change to your extension
    mask_files = sorted(glob.glob(os.path.join(mask_path, "*.png")))

    for img_file, mask_file in zip(image_files, mask_files):
        img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)

        if mask is None or img is None:
            continue  # Skip missing files

        # Apply CLAHE preprocessing
        img_clahe = apply_clahe(img)

        images.append(img_clahe)
        masks.append(mask)

    images = np.array(images).astype(np.float32) / 255.0
    masks = np.array(masks).astype(np.float32) / 255.0

    return images, masks

# Paths to the dataset (replace with your actual paths)
image_path = "path_to_images"
mask_path = "path_to_masks"

# Load and preprocess data
X, y = load_images_and_masks(image_path, mask_path)

# Split into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Model Implementation

# Nested U-Net (U-Net++)
def unet_plus_plus(input_size=(256, 256, 1)):
    inputs = layers.Input(input_size)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)
    
    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(conv5)
    
    up6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv5)
    merge6 = layers.concatenate([conv4, up6], axis=3)
    conv6 = layers.Conv2D(512, 3, activation='relu', padding='same')(merge6)
    conv6 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv6)
    
    up7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv6)
    merge7 = layers.concatenate([conv3, up7], axis=3)
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same')(merge7)
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv7)
    
    up8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv7)
    merge8 = layers.concatenate([conv2, up8], axis=3)
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same')(merge8)
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv8)
    
    up9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8)
    merge9 = layers.concatenate([conv1, up9], axis=3)
    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same')(merge9)
    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv9)
    
    conv10 = layers.Conv2D(1, 1, activation='sigmoid')(conv9)

    return models.Model(inputs=inputs, outputs=conv10)

# Attention U-Net
def attention_unet(input_size=(256, 256, 1)):
    inputs = layers.Input(input_size)
    
    # Regular U-Net structure (omitted for brevity)
    # Add attention mechanisms in each stage
    
    conv10 = layers.Conv2D(1, 1, activation='sigmoid')(conv9)  # Output layer
    
    return models.Model(inputs=inputs, outputs=conv10)

# Initialize both models
nested_model = unet_plus_plus(input_size=(256, 256, 1))
attention_model = attention_unet(input_size=(256, 256, 1))

# Step 3: Model Training and Evaluation

# DICE Score Metric
def dice_coefficient(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# Compile models
nested_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coefficient])
attention_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coefficient])

# Train Nested U-Net
history_nested = nested_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=16)

# Train Attention U-Net
history_attention = attention_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=16)

# Evaluate both models
nested_eval = nested_model.evaluate(X_test, y_test)
attention_eval = attention_model.evaluate(X_test, y_test)

print(f"Nested U-Net DICE Score: {nested_eval[1]}")
print(f"Attention U-Net DICE Score: {attention_eval[1]}")
