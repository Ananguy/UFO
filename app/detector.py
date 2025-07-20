from ultralytics import YOLO
import cv2
from datetime import datetime
import os

# Load lightweight YOLOv8 model
model = YOLO("models/yolov8n.pt")

# Initialize webcam
cap = cv2.VideoCapture(0)
ret, frame1 = cap.read()
ret, frame2 = cap.read()

# Set resolution for better speed
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Font
font = cv2.FONT_HERSHEY_SIMPLEX

# Create output directory
output_dir = "recordings"
os.makedirs(output_dir, exist_ok=True)

# VideoWriter object
out = None
recording = False

print("✅ Smart Surveillance with Recording Started...")

while cap.isOpened():
    # Display-only timestamp
    display_time = datetime.now().strftime("%d %b %Y | %I:%M:%S %p")

    # Motion detection
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 25, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    motion_detected = any(cv2.contourArea(c) > 500 for c in contours)

    if motion_detected:
        results = model(frame1, imgsz=640, conf=0.5, verbose=False)
        annotated_frame = results[0].plot()

        # Start recording if not already
        if not recording:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = os.path.join(output_dir, f"recording_{timestamp}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(filename, fourcc, 20.0, (640, 480))
            recording = True
            print(f"🎥 Recording started: {filename}")

        # Overlay
        cv2.putText(annotated_frame, "Motion  Detected!", (20, 35), font, 0.8, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"{display_time}", (20, 460), font, 0.6, (255, 255, 255), 2)

        cv2.imshow("YOLOv8 Smart Surveillance", annotated_frame)

        if recording:
            out.write(annotated_frame)

    else:
        if recording:
            print("🛑 Motion stopped. Recording saved.")
            recording = False
            out.release()

        cv2.putText(frame1, f"{display_time}", (20, 460), font, 0.6, (255, 255, 255), 2)
        cv2.imshow("YOLOv8 Smart Surveillance", frame1)

    # Update frames
    frame1 = frame2
    ret, frame2 = cap.read()
    if not ret:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🔒 Surveillance exited by user.")
        break

# Final cleanup
if recording:
    out.release()
cap.release()
cv2.destroyAllWindows()
