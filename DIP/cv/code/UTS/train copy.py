#!/usr/bin/env python3
"""
Fixed training script untuk Face Recognition System dengan CNN Neural Network
Perbaikan utama:
 - Paksa MTCNN inisialisasi/warm-up ke GPU bila tersedia
 - Perbaikan alur preprocessing (tidak menimpa variabel, penggunaan rois -> X)
 - TensorFlow GPU memory growth diaktifkan
 - Penanganan fallback PyTorch preprocessing lebih aman
 - Logging lebih jelas
"""

import os
import time
import cv2
import numpy as np
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed

# utils harus menyediakan: detect_faces_viola_jones, extract_face_roi, load_dataset, save_labels, save_model
from utils import (
    detect_faces_viola_jones,
    extract_face_roi,
    load_dataset,
    save_labels,
    save_model,
)
import config

# TensorFlow / Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# PyTorch (optional)
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# facenet-pytorch MTCNN (optional)
try:
    from facenet_pytorch import MTCNN
    MTCNN_AVAILABLE = True
    _mtcnn = None
except Exception:
    MTCNN_AVAILABLE = False
    _mtcnn = None

def enable_tf_gpu_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for g in gpus:
                tf.config.experimental.set_memory_growth(g, True)
            print(f"[TF] Enabled memory growth for {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"[TF] Could not set memory growth: {e}")
    else:
        print("[TF] No GPUs found for TensorFlow")

def build_feature_extractor(input_shape=(128, 128, 3), trainable_base=False):
    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    base_model.trainable = bool(trainable_base)
    return base_model

def _init_mtcnn_if_needed():
    """
    Initialize global _mtcnn on appropriate device and warm up so that it loads to GPU.
    Returns: True if initialized and available, False otherwise.
    """
    global _mtcnn
    if not MTCNN_AVAILABLE or not TORCH_AVAILABLE:
        return False

    if _mtcnn is not None:
        return True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[MTCNN] Initializing MTCNN on device: {device}")

    try:
        mt = MTCNN(keep_all=True, device=device)
        # Warm-up call: pass a small random tensor to force weights onto device
        # MTCNN accepts torch tensors or PIL images; construct a dummy tensor
        try:
            dummy = torch.zeros((1, 3, 160, 160), device=device, dtype=torch.float32)
            # MTCNN callable will accept this tensor (may return None faces) — used to force device allocation
            _ = mt(dummy)  # warm-up
        except Exception:
            # fallback: run detect on a small numpy image (this will also initialize internals)
            dummy_np = (np.zeros((160, 160, 3), dtype=np.uint8))
            _ = mt.detect(dummy_np)
        _mtcnn = mt
        print("[MTCNN] Initialized and warmed up.")
        return True
    except Exception as e:
        print(f"[MTCNN] Initialization failed: {e}")
        _mtcnn = None
        return False

def prepare_training_data(data_dir='data'):
    """
    Prepare training data dari dataset.
    Mengembalikan: X (N,128,128,3), y (N,), label_to_idx (dict), num_classes (int)
    """
    print("\n[1] Loading dataset...")
    dataset = load_dataset(data_dir)

    if not dataset:
        print("ERROR: Dataset kosong! Pastikan folder 'data' ada dengan subfolder untuk setiap orang.")
        return None, None, None, None

    print(f"Found {len(dataset)} people in dataset")

    # build list of tasks
    tasks = []
    label_to_idx = {}
    for person_idx, (person_name, image_paths) in enumerate(dataset.items()):
        label_to_idx[person_name] = person_idx
        for img_path in image_paths:
            tasks.append((person_name, img_path, person_idx))

    print(f"  Total image files to check: {len(tasks)}")

    # ensure mtcnn init if available
    mtcnn_ready = False
    if MTCNN_AVAILABLE and TORCH_AVAILABLE:
        mtcnn_ready = _init_mtcnn_if_needed()
        if not mtcnn_ready:
            print("[WARN] facenet-pytorch MTCNN not ready; will fallback to Viola-Jones for detection.")

    # worker function
    def _process_image(task):
        person_name, img_path, person_idx = task
        try:
            image = cv2.imread(img_path)
            if image is None:
                return (person_name, img_path, person_idx, None, "cannot_read")

            h, w = image.shape[:2]
            max_dim = max(h, w)
            target_max = 640
            if max_dim > target_max:
                scale_det = target_max / float(max_dim)
                det_w = int(w * scale_det)
                det_h = int(h * scale_det)
                small = cv2.resize(image, (det_w, det_h))
            else:
                scale_det = 1.0
                small = image.copy()

            # detection
            faces_small = []
            # Prefer MTCNN if ready
            if mtcnn_ready and _mtcnn is not None:
                try:
                    rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                    boxes, probs = _mtcnn.detect(rgb_small)
                    if boxes is not None:
                        for b in boxes:
                            x1, y1, x2, y2 = [int(v) for v in b]
                            faces_small.append((x1, y1, x2 - x1, y2 - y1))
                except Exception as e:
                    # fallback to Viola-Jones
                    faces_small = detect_faces_viola_jones(
                        small,
                        scale_factor=config.VIOLA_JONES_SCALE_FACTOR,
                        min_neighbors=config.VIOLA_JONES_MIN_NEIGHBORS,
                        min_size=config.VIOLA_JONES_MIN_SIZE
                    )
            else:
                faces_small = detect_faces_viola_jones(
                    small,
                    scale_factor=config.VIOLA_JONES_SCALE_FACTOR,
                    min_neighbors=config.VIOLA_JONES_MIN_NEIGHBORS,
                    min_size=config.VIOLA_JONES_MIN_SIZE
                )

            if len(faces_small) == 0:
                return (person_name, img_path, person_idx, None, "no_face")

            # choose largest face in small image
            fx, fy, fw, fh = max(faces_small, key=lambda f: f[2] * f[3])
            # map coords back to original image
            if scale_det != 1.0:
                fx = int(fx / scale_det)
                fy = int(fy / scale_det)
                fw = int(fw / scale_det)
                fh = int(fh / scale_det)

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            face_coords = (fx, fy, fw, fh)
            face_roi, _ = extract_face_roi(gray, face_coords, padding=config.FACE_ROI_PADDING)

            if face_roi is None:
                return (person_name, img_path, person_idx, None, "no_roi")

            return (person_name, img_path, person_idx, face_roi, "ok")
        except Exception as e:
            return (person_name, img_path, person_idx, None, f"error:{str(e)}")

    # parallel processing (beware CPU usage)
    max_workers = min(8, (os.cpu_count() or 4))  # more conservative than 16
    print(f"[IO] Using ThreadPoolExecutor with max_workers={max_workers}")
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_process_image, t) for t in tasks]
        for f in as_completed(futures):
            try:
                results.append(f.result())
            except Exception as e:
                print(f"  Worker error: {e}")

    # collect ROIs and labels
    rois = []
    roi_labels = []
    per_person_count = {}
    for person_name, img_path, person_idx, face_roi, status in results:
        if status == "ok" and face_roi is not None:
            rois.append(face_roi)
            roi_labels.append(person_idx)
            per_person_count[person_name] = per_person_count.get(person_name, 0) + 1
            print(f"    ✓ {os.path.basename(img_path)} -> {person_name}")
        else:
            if status not in ("no_face", "cannot_read"):
                print(f"    ⚠ Skipped: {os.path.basename(img_path)} ({status})")

    if len(rois) == 0:
        print("\nERROR: No images extracted! Check your dataset.")
        return None, None, None, None

    # Preprocess ROIs: try PyTorch GPU pipeline first (if available)
    X = None
    y = np.array(roi_labels, dtype=np.int32)
    torch_pipeline_success = False

    if TORCH_AVAILABLE:
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"\n[Preproc] Using PyTorch for batched preprocessing. Device: {device}")

            arrs = [np.asarray(r, dtype=np.float32) for r in rois]
            max_h = max(a.shape[0] for a in arrs)
            max_w = max(a.shape[1] for a in arrs)
            stacked = np.zeros((len(arrs), max_h, max_w), dtype=np.float32)
            for i, a in enumerate(arrs):
                h, w = a.shape[:2]
                stacked[i, 0:h, 0:w] = a

            t = torch.from_numpy(stacked)  # (N, H, W)
            t = t.unsqueeze(1)  # (N,1,H,W)
            t = t.to(device)

            # Resize to (128,128) on GPU using bilinear
            t = F.interpolate(t, size=(128, 128), mode='bilinear', align_corners=False)

            # Normalize to [0,1]
            t = t / 255.0

            # Repeat to 3 channels
            t = t.repeat(1, 3, 1, 1)  # (N,3,128,128)

            # Move back to CPU numpy for TensorFlow training
            t_cpu = t.to('cpu').numpy()
            X = np.transpose(t_cpu, (0, 2, 3, 1)).astype('float32')  # (N,128,128,3)
            torch_pipeline_success = True
            print(f"[Preproc] PyTorch preprocessing success. Samples: {len(X)}")
        except Exception as e:
            print(f"[Preproc] PyTorch pipeline failed: {e}. Falling back to CPU OpenCV pipeline.")
            torch_pipeline_success = False

    if not torch_pipeline_success:
        print("\n[Preproc] Using CPU OpenCV resizing & conversion.")
        X_list = []
        for face_roi in rois:
            try:
                resized = cv2.resize(face_roi, (128, 128))
                if len(resized.shape) == 2:
                    resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
                resized = resized.astype('float32') / 255.0
                X_list.append(resized)
            except Exception as e:
                print(f"    ⚠ Resize error: {e}")
        X = np.array(X_list, dtype='float32')

    # final shapes check
    if X is None or X.shape[0] == 0:
        print("ERROR: No preprocessed images available.")
        return None, None, None, None

    num_classes = len(label_to_idx)
    print("\n  → Successfully extracted per-person sample counts:")
    for name, cnt in per_person_count.items():
        print(f"    {name}: {cnt}")

    print(f"\n[Data] Total samples: {len(X)} | Image shape: {X.shape} | Classes: {num_classes}")
    return X, y, label_to_idx, num_classes

def train_face_recognition_model(data_dir='data', output_model='models/face_recognition_model.pkl'):
    print("=" * 60)
    print("FACE RECOGNITION TRAINING - CNN NEURAL NETWORK (FIXED)")
    print("=" * 60)
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {config.TRAINING_EPOCHS}")
    print(f"  Batch Size: {config.BATCH_SIZE}")
    print(f"  Learning Rate: {config.LEARNING_RATE}")
    print(f"  Viola-Jones Min Neighbors: {config.VIOLA_JONES_MIN_NEIGHBORS}")

    # enable TF GPU growth
    enable_tf_gpu_growth()

    # Prepare data
    X, y, label_to_idx, num_classes = prepare_training_data(data_dir)
    if X is None:
        return False

    # Build feature extractor (MobileNetV2)
    print(f"\n[Model] Building feature extractor (MobileNetV2)...")
    base_model = build_feature_extractor(input_shape=(128, 128, 3), trainable_base=False)

    # Data augmentation
    data_augmentation = keras.Sequential([
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.08),
        layers.RandomContrast(0.08),
    ], name='data_augmentation')

    # Build final model
    inputs = layers.Input(shape=(128, 128, 3))
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

    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1)
    ]

    print(f"\n[Train] Training model (transfer learning) up to {config.TRAINING_EPOCHS} epochs...")
    start_time = time.time()
    history = model.fit(
        X, y,
        epochs=config.TRAINING_EPOCHS,
        batch_size=config.BATCH_SIZE,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    training_time = time.time() - start_time

    # Extract feature extractor (without classification head)
    feature_extractor = keras.Model(inputs=base_model.input, outputs=base_model.output)

    # Generate reference features
    print(f"\n[RefDB] Generating reference features...")
    reference_db = {}
    class_similarity_thresholds = {}
    all_similarities = []

    for person_name, person_idx in label_to_idx.items():
        person_mask = (y == person_idx)
        person_images = X[person_mask]
        if len(person_images) == 0:
            print(f"  ⚠ Warning: {person_name} has 0 images after preprocessing, skipping.")
            continue

        person_features = feature_extractor.predict(person_images, verbose=0)
        mean_feature = np.mean(person_features, axis=0)
        norms = np.linalg.norm(person_features, axis=1) * (np.linalg.norm(mean_feature) + 1e-10)
        sims = np.sum(person_features * mean_feature, axis=1) / (norms + 1e-10)
        class_thresh = float(np.percentile(sims, 5)) if len(sims) > 0 else 0.0
        class_similarity_thresholds[person_name] = class_thresh
        all_similarities.extend(sims.tolist())

        reference_db[person_name] = {
            'mean_feature': mean_feature,
            'all_features': person_features,
            'label_idx': person_idx,
            'similarities': sims
        }
        print(f"  ✓ {person_name}: {len(person_features)} samples")

    # Save models & DB
    print(f"\n[Save] Saving models...")
    os.makedirs('models', exist_ok=True)
    feature_extractor.save('models/feature_extractor.h5')

    model_data = {
        'reference_db': reference_db,
        'label_to_idx': label_to_idx,
        'idx_to_label': {v: k for k, v in label_to_idx.items()},
        'training_history': history.history,
        'class_similarity_thresholds': class_similarity_thresholds,
        'global_similarity_threshold': float(np.percentile(all_similarities, 5)) if len(all_similarities) > 0 else 0.5
    }
    save_model(model_data, output_model)
    save_labels(sorted(label_to_idx.keys()), 'labels.txt')

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    print(f"Feature extractor saved to: models/feature_extractor.h5")
    print(f"Model saved to: {output_model}")
    print(f"Labels saved to: labels.txt")
    print(f"Total people: {len(reference_db)}")
    print(f"Total training samples: {len(X)}")
    if 'accuracy' in history.history and 'val_accuracy' in history.history:
        print(f"\nFinal Training Accuracy: {history.history['accuracy'][-1]:.4f}")
        print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")

    return True

if __name__ == "__main__":
    success = train_face_recognition_model()
    if not success:
        exit(1)
