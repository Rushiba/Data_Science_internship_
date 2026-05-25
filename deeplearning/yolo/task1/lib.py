import cv2
from ultralytics import YOLO

# Load the ultra-lightweight YOLOv8 Nano model
model = YOLO("yolov8n.pt")

# Open webcam with the DirectShow flag for Windows stability
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Could not open camera with index 0. Trying index 1...")
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from camera.")
        break

    # Run detection (stream=True optimizes memory usage for video)
    results = model(frame, stream=True)

    # Plot the bounding boxes onto the frame
    annotated_frame = frame  # Fallback if no frames match
    for r in results:
        annotated_frame = r.plot()

    # Display the live result window
    cv2.imshow("YOLOv8 Live Detection", annotated_frame)

    # Press 'q' on your keyboard to close out cleanly
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
