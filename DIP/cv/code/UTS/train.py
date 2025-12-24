#!/usr/bin/env python3
"""
Training script untuk Face Recognition System dengan CNN Neural Network
Menggunakan Viola-Jones untuk deteksi dan CNN untuk feature extraction
"""
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import cv2
import numpy as np
from utils import (
    detect_faces_viola_jones,
    extract_face_roi,
    load_dataset,
    save_labels,
    save_model,
)
from sklearn.preprocessing import StandardScaler
import pickle
import config
from tensorflow import keras
from tensorflow.keras import layers
import time

def build_feature_extractor(input_shape=(128, 128, 3), trainable_base=False):
    """
    Build feature extractor menggunakan transfer learning (MobileNetV2).
    Untuk dataset kecil, kita gunakan pretrained ImageNet weights dan freeze base model.
    """
    # Use MobileNetV2 as a strong feature extractor for small datasets
    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )

    # Freeze base model by default to avoid overfitting on small dataset
    base_model.trainable = bool(trainable_base)

    return base_model

def prepare_training_data(data_dir='data'):
    """
    Prepare training data dari dataset
    """
    print("\n[1] Loading dataset...")
    dataset = load_dataset(data_dir)
    
    if not dataset:
        print("ERROR: Dataset kosong! Pastikan folder 'data' ada dengan subfolder untuk setiap orang.")
        return None, None, None, None
    
    print(f"Found {len(dataset)} people in dataset")
    
    # Extract features dari semua gambar
    print("\n[2] Extracting faces dari training images...")
    all_images = []
    all_labels = []
    label_to_idx = {}
    idx_counter = 0
    
    for person_idx, (person_name, image_paths) in enumerate(dataset.items()):
        print(f"\n  Processing {person_name}...")
        label_to_idx[person_name] = person_idx
        
        valid_images = 0
        
        for img_path in image_paths:
            try:
                # Load image
                image = cv2.imread(img_path)
                if image is None:
                    print(f"    ⚠ Skipped: {os.path.basename(img_path)} (cannot read)")
                    continue
                
                # Convert to grayscale
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                faces = detect_faces_viola_jones(
                    image,
                    scale_factor=config.VIOLA_JONES_SCALE_FACTOR,
                    min_neighbors=config.VIOLA_JONES_MIN_NEIGHBORS,
                    min_size=config.VIOLA_JONES_MIN_SIZE
                )
                
                if len(faces) == 0:
                    print(f"    ⚠ Skipped: {os.path.basename(img_path)} (no face detected)")
                    continue
                
                # Extract ROI dari wajah pertama (terbesar)
                face_coords = max(faces, key=lambda f: f[2] * f[3])
                face_roi, _ = extract_face_roi(gray, face_coords, padding=config.FACE_ROI_PADDING)
                
                # Resize ke 128x128
                face_roi = cv2.resize(face_roi, (128, 128))

                # Jika gambar grayscale, ubah ke 3-channel (required untuk pretrained models)
                if len(face_roi.shape) == 2:
                    face_roi = cv2.cvtColor(face_roi, cv2.COLOR_GRAY2BGR)

                # Normalize pixel values ke range [0,1]
                face_roi = face_roi.astype('float32') / 255.0

                all_images.append(face_roi)
                all_labels.append(person_idx)
                valid_images += 1
                
                print(f"    ✓ {os.path.basename(img_path)}")
                
            except Exception as e:
                print(f"    ✗ Error processing {img_path}: {str(e)}")
                continue
        
        print(f"  → Successfully extracted {valid_images}/{len(image_paths)} images")
        
        if valid_images == 0:
            print(f"  WARNING: No valid images for {person_name}!")
    
    if len(all_images) == 0:
        print("\nERROR: No images extracted! Check your dataset.")
        return None, None, None, None
    
    # Convert to numpy arrays
    X = np.array(all_images)
    # Ensure shape is (N, 128, 128, 3)
    if X.ndim == 3:
        X = X.reshape(-1, 128, 128, 1)
        X = np.repeat(X, 3, axis=-1)

    y = np.array(all_labels)
    
    print(f"\n[3] Data preparation summary:")
    print(f"  Total samples: {len(X)}")
    print(f"  Image shape: {X.shape}")
    print(f"  Number of classes: {len(label_to_idx)}")
    
    return X, y, label_to_idx, len(label_to_idx)

def train_face_recognition_model(data_dir='data', output_model='models/face_recognition_model.pkl'):
    """
    Train face recognition model dengan CNN
    """
    print("=" * 60)
    print("FACE RECOGNITION TRAINING - CNN NEURAL NETWORK")
    print("=" * 60)
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {config.TRAINING_EPOCHS}")
    print(f"  Batch Size: {config.BATCH_SIZE}")
    print(f"  Learning Rate: {config.LEARNING_RATE}")
    print(f"  Viola-Jones Min Neighbors: {config.VIOLA_JONES_MIN_NEIGHBORS}")
    
    # Prepare data
    X, y, label_to_idx, num_classes = prepare_training_data(data_dir)
    
    if X is None:
        return False
    
    # Build feature extractor (transfer learning)
    print(f"\n[4] Building feature extractor (MobileNetV2)...")
    base_model = build_feature_extractor(input_shape=(128, 128, 3), trainable_base=False)

    # Data augmentation pipeline to help small dataset
    data_augmentation = keras.Sequential([
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.08),
        layers.RandomContrast(0.08),
    ], name='data_augmentation')

    # Build final model: augmentation -> base -> head
    inputs = layers.Input(shape=(128, 128, 3))
    x = data_augmentation(inputs)
    x = base_model(x, training=False)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs)

    # Compile model with a low learning rate
    optimizer = keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print(model.summary())

    # Callbacks: early stopping and LR reduction to avoid overfitting on tiny dataset
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1)
    ]

    # Train model
    print(f"\n[5] Training model (transfer learning) up to {config.TRAINING_EPOCHS} epochs...")
    print("Using data augmentation and early stopping to avoid overfitting on small dataset.\n")

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
    
    # Extract feature extractor model (without classification head)
    feature_extractor = keras.Model(
        inputs=base_model.input,
        outputs=base_model.output
    )
    
    # Generate features untuk reference database
    print(f"\n[6] Generating reference features...")
    reference_db = {}
    class_similarity_thresholds = {}
    all_similarities = []
    
    for person_name, person_idx in label_to_idx.items():
        person_mask = y == person_idx
        person_images = X[person_mask]
        
        # Extract features
        person_features = feature_extractor.predict(person_images, verbose=0)
        mean_feature = np.mean(person_features, axis=0)
        # Compute cosine similarities between each sample and the class centroid
        # similarity = (a . b) / (||a|| * ||b||)
        norms = np.linalg.norm(person_features, axis=1) * (np.linalg.norm(mean_feature) + 1e-10)
        sims = np.sum(person_features * mean_feature, axis=1) / (norms + 1e-10)

        # Determine a per-class similarity threshold (e.g., 5th percentile)
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
    
    # Save model
    print(f"\n[7] Saving models...")
    
    # Save feature extractor
    feature_extractor.save('models/feature_extractor.h5')
    
    # Save reference database
    model_data = {
        'reference_db': reference_db,
        'label_to_idx': label_to_idx,
        'idx_to_label': {v: k for k, v in label_to_idx.items()},
        'training_history': history.history,
        'class_similarity_thresholds': class_similarity_thresholds,
        'global_similarity_threshold': float(np.percentile(all_similarities, 5)) if len(all_similarities) > 0 else 0.5
    }
    
    save_model(model_data, output_model)
    
    # Save labels
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
    print(f"\nFinal Training Accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    
    return True

if __name__ == "__main__":
    success = train_face_recognition_model()
    if not success:
        exit(1)
