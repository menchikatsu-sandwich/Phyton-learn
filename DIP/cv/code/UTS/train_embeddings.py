# train_embeddings.py
import cv2
import numpy as np
import os
from deepface import DeepFace  # pakai DeepFace untuk ekstraksi embedding
from sklearn.neighbors import KNeighborsClassifier

DATASET_DIR = "data"
EMBEDDINGS_PATH = "embeddings.npy"
LABELS_PATH = "labels.npy"

embeddings = []
labels = []

print("🚀 Mulai ekstraksi embedding...")

for person_name in os.listdir(DATASET_DIR):
    person_path = os.path.join(DATASET_DIR, person_name)
    if not os.path.isdir(person_path):
        continue
    
    print(f"📸 Proses: {person_name}")
    
    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        try:
            # Ekstraksi embedding
            embedding = DeepFace.represent(
                img_path=img_path,
                model_name="VGG-Face",  # atau "Facenet", "ArcFace"
                detector_backend="opencv",
                enforce_detection=False  # biarkan jika wajah tidak terdeteksi
            )[0]["embedding"]
            
            embeddings.append(embedding)
            labels.append(person_name)
            
        except Exception as e:
            print(f"⚠️ Error pada {img_name}: {e}")
            continue

# Simpan embeddings dan labels
np.save(EMBEDDINGS_PATH, np.array(embeddings))
np.save(LABELS_PATH, np.array(labels))

# Latih K-NN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(np.array(embeddings), np.array(labels))

# Simpan model K-NN
import joblib
joblib.dump(knn, "models/knn_model.pkl")

print(f"✅ Selesai! {len(embeddings)} embedding disimpan.")
print(f"✅ Model K-NN disimpan di: models/knn_model.pkl")