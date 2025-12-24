# uas_yolo.py
import os
import shutil
from ultralytics import YOLO
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# -------------------------------
# CONFIGURASI
# -------------------------------
DATA_DIR = 'data'
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

CLASS_NAMES = [name for name in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, name)) and name != 'flow_frames']

USE_RCNN = False
USE_OPTICAL_FLOW = False
USE_VIOLA_JONES = False

# -------------------------------
# PREPROCESSING DATA + HAAR CASCADE (HANYA JALAN SEKALI)
# -------------------------------
def create_yolo_format():
    yolo_data_dir = 'yolo_data'
    os.makedirs(yolo_data_dir, exist_ok=True)
    os.makedirs(os.path.join(yolo_data_dir, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(yolo_data_dir, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(yolo_data_dir, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(yolo_data_dir, 'labels', 'val'), exist_ok=True)

    image_paths = []
    labels = []

    for class_id, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(DATA_DIR, class_name)
        for img_file in os.listdir(class_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(class_dir, img_file)
                image_paths.append(img_path)
                labels.append(class_id)

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, stratify=labels, random_state=42
    )

    # Hanya load Haar Cascade jika diperlukan
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    for split, paths, lbls in [('train', train_paths, train_labels), ('val', val_paths, val_labels)]:
        for i, (img_path, label) in enumerate(zip(paths, lbls)):
            dst_img = os.path.join(yolo_data_dir, 'images', split, f'{split}_{i}.jpg')
            dst_label = os.path.join(yolo_data_dir, 'labels', split, f'{split}_{i}.txt')

            # ⚡️ JIKA LABEL SUDAH ADA, SKIP DETEKSI (CEPAT!)
            if os.path.exists(dst_label):
                print(f"✅ Skip: label sudah ada untuk {os.path.basename(img_path)}")
                if not os.path.exists(dst_img):
                    shutil.copy(img_path, dst_img)
                continue

            # Salin gambar
            shutil.copy(img_path, dst_img)

            # Deteksi wajah hanya jika belum ada label
            img = cv2.imread(img_path)
            if img is None:
                print(f"⚠️ Gambar rusak: {img_path}")
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) == 0:
                print(f"⚠️ Tidak ada wajah terdeteksi di {img_path}. Skip.")
                os.remove(dst_label) if os.path.exists(dst_label) else None
                os.remove(dst_img) if os.path.exists(dst_img) else None
                continue

            # Ambil wajah pertama
            x, y, w_face, h_face = faces[0]
            x_center = (x + w_face / 2) / img.shape[1]
            y_center = (y + h_face / 2) / img.shape[0]
            bbox_w = w_face / img.shape[1]
            bbox_h = h_face / img.shape[0]

            with open(dst_label, 'w') as f:
                f.write(f"{label} {x_center} {y_center} {bbox_w} {bbox_h}\n")

    # Buat data.yaml (path relatif benar)
    data_yaml_path = os.path.join(yolo_data_dir, 'data.yaml')
    with open(data_yaml_path, 'w') as f:
        f.write("train: ./images/train\n")
        f.write("val: ./images/val\n")
        f.write(f"nc: {len(CLASS_NAMES)}\n")
        f.write(f"names: {CLASS_NAMES}\n")

    print(f"✅ Dataset siap. File: {data_yaml_path}")
    return yolo_data_dir

# -------------------------------
# TRAIN YOLOv8 (ADAPTIVE EPOCHS)
# -------------------------------
def train_yolo(yolo_data_dir):
    print("🚀 Training YOLOv8...")
    model = YOLO('yolov8n.pt')
    results = model.train(
        data=os.path.join(yolo_data_dir, 'data.yaml'),
        epochs=12,
        patience=10,           # Stop otomatis jika tidak improve
        imgsz=640,
        batch=16,
        device=0,
        project=MODEL_DIR,
        name='face_detection',
        save=True,
        exist_ok=True
    )
    print("✅ Training selesai.")

# -------------------------------
# TRAIN KNN (OPSIONAL)
# -------------------------------
def train_knn_classifier():
    print("🧠 Training KNN...")
    features, labels = [], []
    for class_id, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(DATA_DIR, class_name)
        for img_file in os.listdir(class_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img = cv2.imread(os.path.join(class_dir, img_file))
                if img is None: continue
                img = cv2.resize(img, (64, 64))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                features.append(gray.flatten())
                labels.append(class_id)

    X, y = np.array(features), np.array(labels)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_scaled, y)

    knn_path = os.path.join(MODEL_DIR, 'knn_model.pkl')
    scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
    joblib.dump(knn, knn_path)
    joblib.dump(scaler, scaler_path)
    print(f"✅ KNN disimpan di {MODEL_DIR}")

# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    print("📦 Membuat dataset YOLO (deteksi wajah sekali saja)...")
    yolo_data_dir = create_yolo_format()

    print("🔥 Training model...")
    train_yolo(yolo_data_dir)

    print("🧠 Training KNN...")
    train_knn_classifier()

    if USE_RCNN: print("⚠️ RCNN belum diimplementasikan.")
    if USE_OPTICAL_FLOW: print("⚠️ Optical Flow belum diimplementasikan.")
    if USE_VIOLA_JONES: print("⚠️ Viola-Jones belum diimplementasikan.")

    print("🎉 Selesai!")
    