import io
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import tensorflow as tf

app = FastAPI()

# Allow Make.com, WordPress, Render, localhost etc.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load MoveNet
MODEL_PATH = "models/movenet.tflite"

try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
except Exception as e:
    raise RuntimeError(f"Error loading MoveNet model: {e}")


# ------------------------------
# 📌 Utility Functions
# ------------------------------

def preprocess_image(image: Image.Image):
    """Converts uploaded image into MoveNet input format."""
    img = image.convert("RGB")
    img = img.resize((256, 256))
    img = np.array(img)
    img = img.astype("float32")
    img = img / 255.0
    return img[np.newaxis, ...]


def get_angle(a, b, c):
    """Returns angle between three points."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return float(angle)


# ------------------------------
# 📌 Main API Route
# ------------------------------

@app.post("/analyze")
async def analyze_posture(file: UploadFile = File(...)):
    """Returns posture data using MoveNet."""
    if not file:
        raise HTTPException(status_code=400, detail="No image uploaded")

    # Read file
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    # Preprocess
    input_tensor = preprocess_image(image)

    # Run inference
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    interpreter.set_tensor(input_index, input_tensor)
    interpreter.invoke()

    keypoints = interpreter.get_tensor(output_index)[0][0]

    # Extract useful named points
    kp = { 
        "nose": keypoints[0][:2].tolist(),
        "left_eye": keypoints[1][:2].tolist(),
        "right_eye": keypoints[2][:2].tolist(),
        "left_shoulder": keypoints[5][:2].tolist(),
        "right_shoulder": keypoints[6][:2].tolist(),
        "left_hip": keypoints[11][:2].tolist(),
        "right_hip": keypoints[12][:2].tolist()
    }

    # Basic posture metrics
    shoulder_angle = get_angle(kp["left_shoulder"], kp["nose"], kp["right_shoulder"])
    hip_angle = get_angle(kp["left_hip"], kp["nose"], kp["right_hip"])

    # Forward head estimate
    head_forward_shift = abs(kp["nose"][0] - ((kp["left_shoulder"][0] + kp["right_shoulder"][0]) / 2))

    # Lateral tilt
    shoulder_tilt = abs(kp["left_shoulder"][1] - kp["right_shoulder"][1])

    # Rotation metric
    rotation_estimate = abs(kp["left_shoulder"][0] - kp["right_shoulder"][0])

    # Output
    return {
        "success": True,
        "keypoints": kp,
        "metrics": {
            "shoulder_angle": shoulder_angle,
            "hip_angle": hip_angle,
            "forward_head_shift": head_forward_shift,
            "shoulder_tilt": shoulder_tilt,
            "rotation_estimate": rotation_estimate
        }
    }
