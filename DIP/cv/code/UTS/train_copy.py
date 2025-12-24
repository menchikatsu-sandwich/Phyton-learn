#!/usr/bin/env python3
"""
Training script untuk Face Recognition System dengan CNN Neural Network
Menggunakan Viola-Jones untuk deteksi dan CNN untuk feature extraction
(versi diperbaiki untuk TensorFlow GPU support)
"""

import os
import cv2
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# import tensorflow & setup GPU growth
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Utils import
from utils import (
    detect_faces_viola_jones,
    extract_face_roi,
    load_dataset,
    save_labels,
    save_model,
)
import config

# PyTorch optional for preprocessing
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = torch.cuda.is_available()
    TORCH_DEVICE = torch.device('cuda' if TORCH_AVAILABLE else 'cpu')
    print(f"[Torch] cuda available: {TORCH_AVAILABLE}, device: {TORCH_DEVICE}")
except Exception as e:
    TORCH_AVAILABLE = False
    TORCH_DEVICE = None
    print(f"[Torch] import failed: {e}")

# MTCNN
try:
    from facenet_pytorch import MTCNN
    MTCNN_AVAILABLE = True
    _mtcnn = None
except Exception:
    MTCNN_AVAILABLE = False
    _mtcnn = None

def enable_tf_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[TensorFlow] GPU devices: {gpus}")
        except RuntimeError as e:
            print(f"[TensorFlow] GPU setup failed: {e}")
    else:
        print("[TensorFlow] No GPU device detected. Using CPU.")

def build_feature_extractor(input_shape=(128,128,3), trainable_base=False):
    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    base_model.trainable = trainable_base
    return base_model

def _init_mtcnn():
    global _mtcnn
    if not (MTCNN_AVAILABLE and TORCH_AVAILABLE):
        return False
    device = TORCH_DEVICE
    print(f"[MTCNN] Initializing on device: {device}")
    try:
        mt = MTCNN(keep_all=True, device=device)
        # warm-up
        dummy = torch.zeros((1,3,160,160), device=device)
        _ = mt(dummy)
        _mtcnn = mt
        print("[MTCNN] Warm-up complete.")
        return True
    except Exception as e:
        print(f"[MTCNN] init failed: {e}")
        _mtcnn = None
        return False

def prepare_training_data(data_dir='data'):
    global TORCH_AVAILABLE

    print("\n[1] Loading dataset…")
    dataset = load_dataset(data_dir)
    if not dataset:
        print("ERROR: dataset kosong!")
        return None, None, None, None

    print(f"Found {len(dataset)} people")

    tasks = []
    label_to_idx = {}
    for person_idx, (person_name, image_paths) in enumerate(dataset.items()):
        label_to_idx[person_name] = person_idx
        for img_path in image_paths:
            tasks.append((person_name, img_path, person_idx))

    print(f"  Total image files to check: {len(tasks)}")

    # init mtcnn if possible
    mt_ready = _init_mtcnn()

    def _process_image(task):
        person_name, img_path, person_idx = task
        try:
            image = cv2.imread(img_path)
            if image is None:
                return (person_name, img_path, person_idx, None, "cannot_read")

            h,w = image.shape[:2]
            max_dim = max(h,w)
            target_max = 640
            if max_dim > target_max:
                scale_det = target_max / float(max_dim)
                image_small = cv2.resize(image, (int(w*scale_det), int(h*scale_det)))
            else:
                scale_det = 1.0
                image_small = image.copy()

            gray_small = cv2.cvtColor(image_small, cv2.COLOR_BGR2GRAY)

            # face detection
            faces_small = []
            if mt_ready:
                rgb_small = cv2.cvtColor(image_small, cv2.COLOR_BGR2RGB)
                boxes, probs = _mtcnn.detect(rgb_small)
                if boxes is not None:
                    for b in boxes:
                        x1, y1, x2, y2 = [int(v) for v in b]
                        faces_small.append((x1, y1, x2 - x1, y2 - y1))
            if not faces_small:
                # fallback Viola-Jones
                faces_small = detect_faces_viola_jones(
                    image_small,
                    scale_factor=config.VIOLA_JONES_SCALE_FACTOR,
                    min_neighbors=config.VIOLA_JONES_MIN_NEIGHBORS,
                    min_size=config.VIOLA_JONES_MIN_SIZE
                )

            if not faces_small:
                return (person_name, img_path, person_idx, None, "no_face")

            fx,fy,fw,fh = max(faces_small, key=lambda f: f[2]*f[3])
            if scale_det != 1.0:
                fx = int(fx/scale_det)
                fy = int(fy/scale_det)
                fw = int(fw/scale_det)
                fh = int(fh/scale_det)

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            face_coords = (fx,fy,fw,fh)
            face_roi, _ = extract_face_roi(gray, face_coords, padding=config.FACE_ROI_PADDING)
            if face_roi is None:
                return (person_name, img_path, person_idx, None, "no_roi")

            return (person_name, img_path, person_idx, face_roi, "ok")
        except Exception as e:
            return (person_name, img_path, person_idx, None, f"error:{str(e)}")

    max_workers = min(8, (os.cpu_count() or 4))
    print(f"  Using ThreadPoolExecutor with max_workers={max_workers}")
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_process_image, t) for t in tasks]
        for f in as_completed(futures):
            results.append(f.result())

    rois = []
    roi_labels = []
    per_person_count = {}
    for person_name, img_path, person_idx, face_roi, status in results:
        if status == "ok":
            rois.append(face_roi)
            roi_labels.append(person_idx)
            per_person_count[person_name] = per_person_count.get(person_name,0) + 1
        else:
            if status not in ("no_face","cannot_read"):
                print(f"  ⚠ Skipped: {os.path.basename(img_path)} ({status})")

    if len(rois) == 0:
        print("ERROR: no ROIs extracted!")
        return None, None, None, None

    # preprocessing using PyTorch if available
    X = None
    y = np.array(roi_labels, dtype=np.int32)
    if TORCH_AVAILABLE:
        try:
            arrs = [np.asarray(r, dtype=np.float32) for r in rois]
            max_h = max(a.shape[0] for a in arrs)
            max_w = max(a.shape[1] for a in arrs)
            stacked = np.zeros((len(arrs), max_h, max_w), dtype=np.float32)
            for i,a in enumerate(arrs):
                h,w = a.shape[:2]
                stacked[i,0:h,0:w] = a
            t = torch.from_numpy(stacked).unsqueeze(1).to(TORCH_DEVICE)
            t = F.interpolate(t, size=(128,128), mode='bilinear', align_corners=False)
            t = t / 255.0
            t = t.repeat(1,3,1,1)
            t_cpu = t.to('cpu').numpy()
            X = np.transpose(t_cpu,(0,2,3,1)).astype('float32')
            print(f"[Preproc] PyTorch pipeline OK: samples={X.shape[0]}")
        except Exception as e:
            print(f"[Preproc] PyTorch pipeline failed: {e}. Falling back to CPU.")
            TORCH_AVAILABLE = False

    if X is None:
        print("[Preproc] CPU pipeline.")
        X_list = []
        for face_roi in rois:
            try:
                resized = cv2.resize(face_roi,(128,128))
                if resized.ndim == 2:
                    resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
                resized = resized.astype('float32')/255.0
                X_list.append(resized)
            except Exception as e:
                print(f"  ⚠ resize error: {e}")
        X = np.array(X_list, dtype='float32')

    num_classes = len(label_to_idx)
    print(f"[Data] samples:{X.shape[0]}, shape:{X.shape}, classes:{num_classes}")
    for name,cnt in per_person_count.items():
        print(f"  {name}: {cnt}")

    return X, y, label_to_idx, num_classes

def train_face_recognition_model(data_dir='data', output_model='models/face_recognition_model.pkl'):
    print("="*60)
    print("FACE RECOGNITION TRAINING - CNN NEURAL NETWORK")
    print("="*60)
    print(f"Epochs: {config.TRAINING_EPOCHS}, Batch: {config.BATCH_SIZE}, LR: {config.LEARNING_RATE}")

    enable_tf_gpu()

    X, y, label_to_idx, num_classes = prepare_training_data(data_dir)
    if X is None:
        return False

    print("[Model] building feature extractor …")
    base_model = build_feature_extractor(input_shape=(128,128,3), trainable_base=False)

    data_augmentation = keras.Sequential([
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.08),
        layers.RandomContrast(0.08),
    ], name='data_augmentation')

    inputs = layers.Input(shape=(128,128,3))
    x = data_augmentation(inputs)
    x = base_model(x, training=False)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs, outputs)

    optimizer = keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print(model.summary())

    print("[Train] start training …")
    start_time = time.time()
    history = model.fit(
        X, y,
        epochs=config.TRAINING_EPOCHS,
        batch_size=config.BATCH_SIZE,
        validation_split=0.2,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1)
        ],
        verbose=1
    )
    duration = time.time() - start_time

    print(f"[Train] done in {duration:.2f} sec")

    # extract feature extractor
    feature_extractor = keras.Model(inputs=base_model.input, outputs=base_model.output)

    print("[RefDB] generating reference features …")
    reference_db = {}
    class_similarity_thresholds = {}
    all_sims = []
    for person_name, idx in label_to_idx.items():
        mask = (y == idx)
        imgs = X[mask]
        if imgs.shape[0] == 0:
            continue
        features = feature_extractor.predict(imgs, verbose=0)
        mean_feat = np.mean(features, axis=0)
        norms = np.linalg.norm(features, axis=1)*(np.linalg.norm(mean_feat)+1e-10)
        sims = np.sum(features * mean_feat, axis=1)/(norms+1e-10)
        thresh = float(np.percentile(sims,5))
        class_similarity_thresholds[person_name] = thresh
        all_sims.extend(sims.tolist())
        reference_db[person_name] = {
            'mean_feature': mean_feat,
            'all_features': features,
            'label_idx': idx,
            'similarities': sims
        }
        print(f"  ✓ {person_name}: {features.shape[0]} samples")

    os.makedirs('models', exist_ok=True)
    feature_extractor.save('models/feature_extractor.h5')
    model_data = {
        'reference_db': reference_db,
        'label_to_idx': label_to_idx,
        'idx_to_label': {v:k for k,v in label_to_idx.items()},
        'training_history': history.history,
        'class_similarity_thresholds': class_similarity_thresholds,
        'global_similarity_threshold': float(np.percentile(all_sims,5))
    }
    save_model(model_data, output_model)
    save_labels(sorted(label_to_idx.keys()), 'labels.txt')

    print("="*60)
    print("TRAINING COMPLETED")
    print("="*60)
    print(f"Total people: {len(reference_db)}, total samples: {X.shape[0]}")
    if 'accuracy' in history.history and 'val_accuracy' in history.history:
        print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
        print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
    return True

if __name__ == "__main__":
    success = train_face_recognition_model()
    if not success:
        exit(1)
