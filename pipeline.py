import os
import json
import cv2  # OpenCV for computer vision
from datetime import datetime
from PIL import Image
from ultralytics import YOLO  # YOLO object detection model
from google import genai  #  Google Gen AI SDK

# Load YOLOv8 Nano model
model = YOLO("yolov8n.pt")

# Initialize Gemini API Client
GEMINI_API_KEY = "AIzaSyDSvUD47qMi0rUFttCi01-HTQiHMxURqxM"
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# Load image
image_path = "traffic.jpg"
image = cv2.imread(image_path)

# Make sure image exists
if image is None:
    print(f"Error: Could not load image '{image_path}'")
    exit()

# Simulated camera metadata
ASSIGNED_CAMERA_ID = "CAM_DALLAS_MAIN_04"
CAMERA_GPS_COORDINATES = "32.7767, -96.7970"

# Run YOLO detection
results = model(image)
first_result = results[0]

for index, box in enumerate(first_result.boxes):

    class_id = int(box.cls[0])
    object_name = model.names[class_id]

    if object_name in ["car", "motorcycle", "bus", "truck"]:

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cropped_vehicle = image[y1:y2, x1:x2]

        if cropped_vehicle.size == 0:
            continue

        output_filename = f"vehicle_{index}_{object_name}.jpg"
        cv2.imwrite(output_filename, cropped_vehicle)

        # Convert image to rgb for Gemini
        rgb_image = cv2.cvtColor(cropped_vehicle, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)

        # GEMINI API
        prompt = (
            "Analyze this cropped vehicle image from a traffic camera. "
            "Identify its dominant paint color, broad vehicle body type (e.g., Sedan, SUV, Coupe, Truck), "
            "and its exact Make and Model (including year generation if possible). "
            "You must return your response as a raw JSON block containing exactly these three keys: "
            "'color', 'body_type', and 'make_model'."
        )

        try:
            response = gemini_client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[pil_image, prompt]
            )

            # Clean up the output string to ensure Python can read it as JSON
            raw_text = response.text.replace("```json", "").replace("```", "").strip()
            gemini_data = json.loads(raw_text)

        except Exception as e:
            print(f"Gemini API Error: {e}")
            # Fallback data if the API call fails or limits out
            gemini_data = {"color": "Unknown", "body_type": "Unknown", "make_model": "Unknown"}

        # -------------------------------------------------------------        # -------------------------------------------------------------
        vehicle_metadata = {
            "camera_id": ASSIGNED_CAMERA_ID,
            "gps": CAMERA_GPS_COORDINATES,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "crop_path": output_filename,
            "object_type": object_name,
            "color": gemini_data.get("color", "Unknown"),
            "body_type": gemini_data.get("body_type", "Unknown"),
            "predicted_make_model": gemini_data.get("make_model", "Unknown")
        }

        print(f"Successfully processed vehicle crop #{index}")
        print("--- Dynamic Metadata Profile ---")
        print(json.dumps(vehicle_metadata, indent=4))
