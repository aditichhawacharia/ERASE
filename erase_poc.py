"""
ERASE PoC: Edge Redaction and Surveillance Evaluation
Simulates an edge-computing camera mounted on a fixed traffic pole.
Processes a traffic video feed, capturing vehicle data and redacting pedestrians in real-time.
"""

import os
import sys
import time
import requests
import cv2

# Define constants
VIDEO_FILENAME = "sample_traffic.mp4"
VIDEO_URL = "https://github.com/intel-iot-devkit/sample-videos/raw/master/person-bicycle-car-detection.mp4"
MODEL_NAME = "yolov8n.pt"

def download_file(url, local_path):
    """Downloads a file from a URL to a local path with a progress bar."""
    print(f"[INFO] Sample video not found. Downloading from:\n      {url}")
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024
        downloaded = 0
        
        with open(local_path, 'wb') as file:
            for data in response.iter_content(block_size):
                file.write(data)
                downloaded += len(data)
                if total_size > 0:
                    percent = int(100 * downloaded / total_size)
                    bar = '=' * (percent // 2)
                    sys.stdout.write(f"\r[INFO] Download Progress: [{bar:<50}] {percent}% ({downloaded / (1024*1024):.2f} MB / {total_size / (1024*1024):.2f} MB)")
                    sys.stdout.flush()
        print("\n[INFO] Download complete.")
    except Exception as e:
        print(f"\n[ERROR] Failed to download sample video: {e}")
        print("[ERROR] Please place a sample video named 'sample_traffic.mp4' in this directory manually.")
        sys.exit(1)

def main():
    # 1. Download sample video if not present
    if not os.path.exists(VIDEO_FILENAME):
        download_file(VIDEO_URL, VIDEO_FILENAME)
        
    # Check if dependencies are available
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] 'ultralytics' is not installed. Please run 'pip install -r requirements.txt'")
        sys.exit(1)

    # 2. Load model
    print(f"[INFO] Loading YOLO model {MODEL_NAME}...")
    try:
        model = YOLO(MODEL_NAME)
    except Exception as e:
        print(f"[ERROR] Failed to load YOLO model: {e}")
        sys.exit(1)
    
    # 3. Open video
    print(f"[INFO] Opening video feed: {VIDEO_FILENAME}")
    cap = cv2.VideoCapture(VIDEO_FILENAME)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video file {VIDEO_FILENAME}")
        sys.exit(1)
        
    # Setup OpenCV window
    window_name = "ERASE PoC - Edge Traffic Camera & Privacy Redaction"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Frame rate smoothing variables
    fps = 0.0
    
    print("[INFO] Processing feed. Press 'q' inside the video window to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            # End of video: reset frame pointer to loop continuously
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
            
        start_time = time.time()
        h, w = frame.shape[:2]
        
        # Run inference using YOLOv8 (disable console print output for clean logs)
        results = model(frame, verbose=False)
        boxes = results[0].boxes
        
        vehicle_count = 0
        redacted_count = 0
        
        # Separate lists to ensure redactions are processed and rendered first (or vice-versa)
        # Redacting first ensures we don't draw green capture boxes over blurred regions
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            # Extract coordinates
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
            
            # Clamp coordinates to frame boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            # Ensure coordinates form a valid box
            if (x2 - x1) <= 0 or (y2 - y1) <= 0:
                continue

            if cls == 0:  # Humans (Privacy Redaction)
                redacted_count += 1
                roi = frame[y1:y2, x1:x2]
                
                if roi.size > 0:
                    # Determine appropriate kernel sizes (must be odd and positive)
                    # Use a heavy 99x99 Gaussian Blur, but scale down if the bounding box is extremely small
                    ksize_w = min(99, x2 - x1)
                    ksize_h = min(99, y2 - y1)
                    
                    # Ensure kernel dimensions are odd
                    ksize_w = ksize_w - 1 if ksize_w % 2 == 0 else ksize_w
                    ksize_h = ksize_h - 1 if ksize_h % 2 == 0 else ksize_h
                    
                    # Ensure minimum size of 3
                    ksize_w = max(3, ksize_w)
                    ksize_h = max(3, ksize_h)
                    
                    blurred = cv2.GaussianBlur(roi, (ksize_w, ksize_h), 0)
                    frame[y1:y2, x1:x2] = blurred
                
                # Draw red bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # Red text tag "REDACTED"
                label = "REDACTED"
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                label_y1 = max(0, y1 - label_h - 6)
                cv2.rectangle(frame, (x1, label_y1), (x1 + label_w + 10, label_y1 + label_h + 6), (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, label, (x1 + 5, label_y1 + label_h + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                
            elif cls in [2, 7]:  # Vehicles (Cars and Trucks)
                vehicle_count += 1
                
                # Draw green bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Green text tag "VEHICLE CAPTURE"
                label = "VEHICLE CAPTURE"
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                label_y1 = max(0, y1 - label_h - 6)
                cv2.rectangle(frame, (x1, label_y1), (x1 + label_w + 10, label_y1 + label_h + 6), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, label, (x1 + 5, label_y1 + label_h + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

        # Compute smoothed processing frame rate (FPS)
        curr_time = time.time()
        time_diff = curr_time - start_time
        if time_diff > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / time_diff)
            
        # Draw Premium HUD Header Overlay
        # 1. Semi-transparent banner
        hud_h = 60
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, hud_h), (12, 16, 22), cv2.FILLED)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        cv2.line(frame, (0, hud_h), (w, hud_h), (43, 58, 66), 1)
        
        # Define dynamic layout columns to fit smaller resolutions nicely
        col1_x = 20
        col2_x = max(240, w // 2 - 130)
        col3_x = max(385, w // 2 + 15)
        col4_x = w - 190
        
        # Column 1: System Title
        cv2.putText(frame, "ERASE // PRIVACY SHIELD", (col1_x, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "EDGE UNIT: POLE_CAM_01", (col1_x, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 160, 175), 1, cv2.LINE_AA)
        
        # Column 2: Vehicles Counter Card
        cv2.rectangle(frame, (col2_x, 12), (col2_x + 130, 48), (20, 25, 23), cv2.FILLED)
        cv2.rectangle(frame, (col2_x, 12), (col2_x + 130, 48), (0, 220, 0), 1)
        cv2.putText(frame, f"VEHICLES: {vehicle_count}", (col2_x + 10, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
        
        # Column 3: Redactions Counter Card
        cv2.rectangle(frame, (col3_x, 12), (col3_x + 130, 48), (25, 20, 20), cv2.FILLED)
        cv2.rectangle(frame, (col3_x, 12), (col3_x + 130, 48), (0, 0, 220), 1)
        cv2.putText(frame, f"REDACTED: {redacted_count}", (col3_x + 10, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
        
        # Column 4: System Status
        pulse = int((time.time() * 2.5) % 2)
        dot_color = (0, 255, 0) if pulse == 0 else (0, 120, 0)
        cv2.circle(frame, (col4_x + 10, 30), 4, dot_color, -1, cv2.LINE_AA)
        cv2.putText(frame, "SECURED", (col4_x + 22, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {fps:.1f}", (col4_x + 105, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Display the frame
        cv2.imshow(window_name, frame)
        
        # Break loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Resources released. Safe exit.")

if __name__ == "__main__":
    main()
