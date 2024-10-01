from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from io import BytesIO

app = FastAPI()

# Load your trained model
model = load_model("path_to_your_trained_model.h5", custom_objects={"dice_coefficient": dice_coefficient})

def apply_clahe(image):
    """Apply CLAHE to enhance the image contrast."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read image file
    image = await file.read()
    image = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_GRAYSCALE)

    # Preprocess the image (apply CLAHE and normalization)
    image = apply_clahe(image)
    image = image.astype(np.float32) / 255.0
    image = cv2.resize(image, (256, 256))  # Resize to match model input
    image = np.expand_dims(image, axis=[0, -1])  # Add batch and channel dimension

    # Make prediction
    prediction = model.predict(image)
    predicted_mask = (prediction[0, ..., 0] > 0.5).astype(np.uint8) * 255  # Binarize the mask

    # Convert the mask to a format that can be returned
    _, buffer = cv2.imencode('.png', predicted_mask)
    return JSONResponse(content={"mask": buffer.tobytes()})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
