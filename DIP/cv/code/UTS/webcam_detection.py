# webcam_detection.py
import os
import cv2
import numpy as np
import argparse
import joblib
from ultralytics import YOLO

# -------------------------------
# ARGUMENT PARSER
# -------------------------------
parser = argparse.ArgumentParser(description="Deteksi wajah via kamera")
parser.add_argument('--use-knn', action='store_true', help='Gunakan KNN untuk klasifikasi wajah')
args = parser.parse_args()

# -------------------------------
# LOAD MODEL
# -------------------------------
MODEL_DIR = 'models'
YOLO_MODEL_PATH = os.path.join(MODEL_DIR, 'face_detection', 'weights', 'best.pt')
KNN_MODEL_PATH = os.path.join(MODEL_DIR, 'knn_model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')

# Load YOLO model
print("🚀 Memuat model YOLOv8...")
yolo_model = YOLO(YOLO_MODEL_PATH)

# Load KNN (jika dipilih)
knn_model = None
scaler = None
CLASS_NAMES = ['adi', 'emo', 'pius', 'rizky', 'santo']  # Sesuaikan dengan foldermu

if args.use_knn:
    print("🧠 Memuat model KNN...")
    knn_model = joblib.load(KNN_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

# -------------------------------
# WEBCAM DETECTION LOOP
# -------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Kamera tidak terbaca!")
    exit()

print("🎥 Kamera aktif. Tekan 'q' untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Deteksi dengan YOLO
    results = yolo_model(frame, conf=0.5, verbose=False)

    # Jika tidak ada deteksi
    if len(results) == 0 or len(results[0].boxes) == 0:
        cv2.putText(frame, "Tidak Diketahui", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box
            confs = result.boxes.conf.cpu().numpy()  # Confidence
            classes = result.boxes.cls.cpu().numpy()  # Class ID

            for i, (box, conf, cls_id) in enumerate(zip(boxes, confs, classes)):
                x1, y1, x2, y2 = map(int, box)
                label = CLASS_NAMES[int(cls_id)] if int(cls_id) < len(CLASS_NAMES) else "Tidak Diketahui"

                # Jika pakai KNN, lakukan prediksi ulang berdasarkan crop wajah
                if args.use_knn:
                    face_crop = frame[y1:y2, x1:x2]
                    if face_crop.size == 0:
                        continue
                    face_resize = cv2.resize(face_crop, (64, 64))
                    gray = cv2.cvtColor(face_resize, cv2.COLOR_BGR2GRAY)
                    feature = gray.flatten().reshape(1, -1)
                    feature_scaled = scaler.transform(feature)
                    knn_pred = knn_model.predict(feature_scaled)[0]
                    knn_label = CLASS_NAMES[knn_pred] if knn_pred < len(CLASS_NAMES) else "Tidak Diketahui"
                    label = knn_label  # Ganti label dengan hasil KNN

                # Gambar bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Teks di dalam box (pojok kiri atas dalam box)
                text = f"{label} ({conf:.2f})"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                text_x = x1 + 5
                text_y = y1 + text_size[1] + 5

                # Pastikan teks tidak keluar dari box
                if text_y > y2:
                    text_y = y2 - 5
                if text_x + text_size[0] > x2:
                    text_x = x2 - text_size[0] - 5

                # Background hitam agar teks jelas
                cv2.rectangle(frame, (text_x - 3, text_y - text_size[1] - 3),
                              (text_x + text_size[0] + 3, text_y + 3), (0, 0, 0), -1)
                cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow('Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("👋 Keluar dari aplikasi.")