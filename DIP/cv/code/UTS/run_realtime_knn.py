# run_realtime_knn.py
import cv2
import numpy as np
import joblib
from deepface import DeepFace

# --- Muat model ---
knn = joblib.load("models/knn_model.pkl")

# --- Inisialisasi kamera ---
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    detected_name = "Tidak Dikenali"
    confidence_percent = 0

    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]

        try:
            # Ekstraksi embedding wajah baru
            embedding = DeepFace.represent(
                img_path=face_roi,
                model_name="VGG-Face",
                detector_backend="opencv",
                enforce_detection=False
            )[0]["embedding"]

            # Prediksi dengan K-NN
            pred_proba = knn.predict_proba([embedding])[0]
            max_prob = np.max(pred_proba)
            predicted_label = knn.predict([embedding])[0]

            confidence_percent = int(max_prob * 100)

            if confidence_percent >= 60:  # threshold bisa diatur
                detected_name = predicted_label
            else:
                detected_name = "Tidak Dikenali"
                confidence_percent = 0

            # Gambar bounding box
            color = (0, 255, 0) if detected_name != "Tidak Dikenali" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Tampilkan nama + confidence di atas bounding box
            label_text = f"{detected_name} ({confidence_percent}%)"
            cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        except Exception as e:
            print("❌ Error:", e)
            continue

    # Tampilkan di pojok kanan atas
    status_text = f"Detected: {detected_name} ({confidence_percent}%)"
    cv2.putText(frame, status_text, 
                (frame.shape[1] - 400, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Face Recognition - Realtime", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()