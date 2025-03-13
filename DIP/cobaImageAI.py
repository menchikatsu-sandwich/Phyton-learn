from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load model YOLOv5 pre-trained
model = YOLO("yolov5s.pt")

# Load image
image_path = "mobil.jpg"  # Sesuaikan dengan lokasi file
image = cv2.imread(image_path)

# Run detection
results = model(image)

# Visualisasi hasil
detected_image = results.show()
cv2.imwrite("output.jpg", detected_image)

# Hitung jumlah mobil, motor, dan bus
detections = results.pandas().xyxy[0]  # Konversi hasil ke DataFrame
vehicle_counts = detections['name'].value_counts()

print(vehicle_counts)
