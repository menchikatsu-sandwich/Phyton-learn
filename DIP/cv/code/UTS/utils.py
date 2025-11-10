import cv2
import numpy as np
import os
from pathlib import Path
import pickle

# Viola-Jones Cascade Classifier untuk deteksi wajah
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def detect_faces_viola_jones(image, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
    """
    Deteksi wajah menggunakan Viola-Jones Cascade Classifier
    Returns: list of (x, y, w, h) untuk setiap wajah yang terdeteksi
    
    Parameters:
    - scale_factor: Semakin kecil = lebih akurat tapi lebih lambat
    - min_neighbors: Semakin rendah = lebih sensitif
    - min_size: Ukuran minimum wajah yang terdeteksi
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    return faces

def extract_face_roi(image, face_coords, padding=10):
    """
    Extract Region of Interest (ROI) dari wajah yang terdeteksi
    """
    x, y, w, h = face_coords
    # Tambah padding untuk context
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(image.shape[1] - x, w + 2 * padding)
    h = min(image.shape[0] - y, h + 2 * padding)
    
    face_roi = image[y:y+h, x:x+w]
    return face_roi, (x, y, w, h)

def extract_cnn_features(face_roi, model):
    """
    Extract fitur menggunakan CNN (Deep Learning)
    Menggunakan pre-trained model dari OpenCV
    """
    # Resize ke ukuran standar
    face_resized = cv2.resize(face_roi, (224, 224))
    
    # Normalisasi
    face_blob = cv2.dnn.blobFromImage(
        face_resized,
        scalefactor=1.0 / 255,
        size=(224, 224),
        mean=[104, 117, 123],
        swapRB=False,
        crop=False
    )
    
    # Extract features
    model.setInput(face_blob)
    features = model.forward()
    
    return features.flatten()

def calculate_optical_flow(prev_gray, curr_gray, prev_points):
    """
    Hitung Optical Flow untuk tracking motion
    Menggunakan Lucas-Kanade method
    """
    if prev_points is None or len(prev_points) == 0:
        return None, None
    
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )
    
    next_points, status, error = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, prev_points, None, **lk_params
    )
    
    return next_points, status

def load_dataset(data_dir='data'):
    """
    Load semua gambar dari dataset dan organize by person
    Returns: dict dengan struktur {person_name: [image_paths]}
    """
    dataset = {}
    
    if not os.path.exists(data_dir):
        print(f"Dataset directory '{data_dir}' tidak ditemukan!")
        return dataset
    
    for person_folder in os.listdir(data_dir):
        person_path = os.path.join(data_dir, person_folder)
        
        if not os.path.isdir(person_path):
            continue
        
        image_paths = []
        for img_file in os.listdir(person_path):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_paths.append(os.path.join(person_path, img_file))
        
        if image_paths:
            dataset[person_folder] = image_paths
            print(f"Loaded {len(image_paths)} images for {person_folder}")
    
    return dataset

def save_labels(labels, filename='labels.txt'):
    """
    Simpan label mapping ke file
    """
    with open(filename, 'w') as f:
        for idx, label in enumerate(labels):
            f.write(f"{idx},{label}\n")
    print(f"Labels saved to {filename}")

def load_labels(filename='labels.txt'):
    """
    Load label mapping dari file
    """
    labels = {}
    if not os.path.exists(filename):
        return labels
    
    with open(filename, 'r') as f:
        for line in f:
            idx, label = line.strip().split(',')
            labels[int(idx)] = label
    
    return labels

def save_model(model_data, filename='models/face_recognition_model.pkl'):
    """
    Simpan trained model ke file
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"Model saved to {filename}")

def load_model(filename='models/face_recognition_model.pkl'):
    """
    Load trained model dari file
    """
    if not os.path.exists(filename):
        return None
    
    with open(filename, 'rb') as f:
        model_data = pickle.load(f)
    
    return model_data

def euclidean_distance(feat1, feat2):
    """
    Hitung Euclidean distance antara dua feature vectors
    """
    return np.sqrt(np.sum((feat1 - feat2) ** 2))

def calculate_confidence(distance, threshold=0.6):
    """
    Hitung confidence score berdasarkan distance
    Semakin kecil distance, semakin tinggi confidence
    """
    if distance > threshold:
        return 0.0
    
    confidence = 1.0 - (distance / threshold)
    return max(0.0, min(1.0, confidence))

def draw_detection_box(image, face_coords, person_name, confidence, color=(0, 255, 0)):
    """
    Draw bounding box dan label pada image
    """
    x, y, w, h = face_coords
    
    # Draw rectangle
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    
    # Prepare label text
    label = f"{person_name} ({confidence:.2%})"
    
    # Get text size untuk background
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    
    # Draw background rectangle untuk text
    cv2.rectangle(
        image,
        (x, y - text_size[1] - 10),
        (x + text_size[0] + 10, y),
        color,
        -1
    )
    
    # Put text
    cv2.putText(
        image,
        label,
        (x + 5, y - 5),
        font,
        font_scale,
        (255, 255, 255),
        thickness
    )
    
    return image

def load_cnn_model():
    """
    Load pre-trained CNN model untuk feature extraction
    Menggunakan ResNet50 dari OpenCV DNN module
    """
    try:
        # Download model jika belum ada
        model_path = 'models/resnet50.caffemodel'
        config_path = 'models/resnet50.prototxt'
        
        if not os.path.exists(model_path) or not os.path.exists(config_path):
            print("Downloading pre-trained CNN model...")
            # Untuk simplicity, kita gunakan built-in model
            # Atau bisa download dari: https://github.com/opencv/opencv_3rdparty
            pass
        
        # Gunakan model yang lebih sederhana dan tersedia
        net = cv2.dnn.readNetFromCaffe(config_path, model_path)
        return net
    except:
        print("Warning: CNN model tidak ditemukan. Menggunakan feature extraction sederhana.")
        return None

def extract_simple_features(face_roi):
    """
    Extract fitur sederhana (untuk backward compatibility)
    Menggunakan histogram dan edge detection
    """
    # Resize
    face_resized = cv2.resize(face_roi, (128, 128))
    
    # Convert to grayscale
    if len(face_resized.shape) == 3:
        gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_resized
    
    # Histogram features
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    
    # Edge detection features
    edges = cv2.Canny(gray, 100, 200)
    edge_hist = cv2.calcHist([edges], [0], None, [256], [0, 256])
    edge_hist = cv2.normalize(edge_hist, edge_hist).flatten()
    
    # Combine features
    features = np.concatenate([hist, edge_hist])
    
    return features