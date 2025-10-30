# run_realtime.py
import cv2
import numpy as np
import tensorflow as tf

# --- Konfigurasi ---
MODEL_PATH = "models/face_model.h5"
LABELS_PATH = "labels.txt"
IMG_SIZE = (160, 160)
CONFIDENCE_THRESHOLD = 0.7  # naikkan threshold karena dataset kecil

# --- Muat model dan label ---
model = tf.keras.models.load_model(MODEL_PATH)
with open(LABELS_PATH, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- Inisialisasi kamera ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Tidak bisa buka kamera!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    detected_name = "Tidak Dikenali"  # default

    for (x, y, w, h) in faces:
        # Crop wajah
        face_roi = frame[y:y+h, x:x+w]
        
        # Preprocess
        face_resized = cv2.resize(face_roi, IMG_SIZE)
        face_normalized = face_resized / 255.0
        face_expanded = np.expand_dims(face_normalized, axis=0)

        # Prediksi
        pred = model.predict(face_expanded, verbose=0)
        idx = np.argmax(pred)
        confidence = pred[0][idx]

        if confidence >= CONFIDENCE_THRESHOLD:
            detected_name = class_names[idx]
        else:
            detected_name = "Tidak Dikenali"

        # Gambar bounding box
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        # Tampilkan nama di dalam bounding box
        cv2.putText(frame, detected_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    # Tampilkan nama di pojok kanan atas
    cv2.putText(frame, f"Detected: {detected_name}", 
                (frame.shape[1]-300, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)

    cv2.imshow("Face Recognition - Realtime", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()