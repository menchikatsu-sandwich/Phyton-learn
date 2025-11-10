#!/usr/bin/env python3
"""
Real-time Face Recognition System dengan CNN
Menampilkan kamera dengan deteksi wajah dan confidence score
"""

import cv2
import numpy as np
import os
import tensorflow as tf
from utils import (
    detect_faces_viola_jones,
    extract_face_roi,
    load_model,
    euclidean_distance,
    calculate_confidence,
    draw_detection_box,
)
import config

class FaceRecognitionSystem:
    def __init__(self, model_path='models/face_recognition_model.pkl', 
                 feature_extractor_path='models/feature_extractor.h5',
                 confidence_threshold=None):
        """
        Initialize Face Recognition System dengan CNN
        """
        self.model_path = model_path
        self.feature_extractor_path = feature_extractor_path
        self.confidence_threshold = confidence_threshold or config.CONFIDENCE_THRESHOLD
        self.model_data = None
        self.feature_extractor = None
        self.detection_history = {}
        
        # Load model dan feature extractor
        self.load_trained_model()
    
    def load_trained_model(self):
        """
        Load trained model dan feature extractor
        """
        print("Loading trained model...")
        self.model_data = load_model(self.model_path)
        
        if self.model_data is None:
            print(f"ERROR: Model tidak ditemukan di {self.model_path}")
            print("Jalankan train.py terlebih dahulu!")
            exit(1)
        
        print("Loading feature extractor...")
        try:
            self.feature_extractor = tf.keras.models.load_model(self.feature_extractor_path)
            print(f"✓ Feature extractor loaded successfully")
        except Exception as e:
            print(f"ERROR: Cannot load feature extractor: {str(e)}")
            print("Pastikan train.py sudah dijalankan dengan sukses!")
            exit(1)
        
        print(f"✓ Model loaded successfully")
        print(f"  People in database: {len(self.model_data['reference_db'])}")
        print(f"  Confidence threshold: {self.confidence_threshold:.2%}")
        print(f"  Distance threshold: {config.DISTANCE_THRESHOLD}")
        # Load similarity thresholds if present (from training)
        self.class_similarity_thresholds = self.model_data.get('class_similarity_thresholds', {})
        self.global_similarity_threshold = self.model_data.get('global_similarity_threshold', 0.5)
        print(f"  Global similarity threshold: {self.global_similarity_threshold:.3f}")
    
    def recognize_face(self, face_roi):
        """
        Recognize wajah dari ROI menggunakan CNN features
        Returns: (person_name, confidence_score)
        """
        try:
            # Ensure face ROI is color (3-channel). If grayscale, convert to BGR.
            if face_roi is None or face_roi.size == 0:
                return "Unknown", 0.0

            if len(face_roi.shape) == 2 or face_roi.shape[2] == 1:
                face_color = cv2.cvtColor(face_roi, cv2.COLOR_GRAY2BGR)
            else:
                face_color = face_roi

            # Resize ke 128x128 (training resized to this)
            face_resized = cv2.resize(face_color, (128, 128))

            # Normalize pixel values to [0,1] - matches training preprocessing
            face_input = face_resized.astype('float32') / 255.0

            # Add batch dimension
            face_input = np.expand_dims(face_input, axis=0)

            # Extract features menggunakan feature_extractor (embedding)
            features = self.feature_extractor.predict(face_input, verbose=0)[0]
            
            # Compare dengan semua orang di database
            best_match = None
            best_confidence = 0.0
            best_distance = float('inf')
            
            for person_name, person_data in self.model_data['reference_db'].items():
                mean_feature = person_data['mean_feature']

                # Calculate cosine similarity between embeddings
                # cosine_sim = (a . b) / (||a|| * ||b||)
                num = float(np.dot(features, mean_feature))
                denom = (np.linalg.norm(features) * (np.linalg.norm(mean_feature) + 1e-10))
                sim = num / (denom + 1e-10)

                # Use similarity as score; higher is better
                if sim > best_confidence:
                    best_confidence = sim
                    best_match = person_name
            
            # Decide unknown vs known using class-specific threshold if available
            class_thresh = self.class_similarity_thresholds.get(best_match, None)
            threshold = class_thresh if class_thresh is not None else self.global_similarity_threshold

            if best_confidence >= threshold:
                # Scale confidence to [0,1] relative to threshold
                scaled_conf = (best_confidence - threshold) / (1.0 - threshold + 1e-10)
                scaled_conf = float(np.clip(scaled_conf, 0.0, 1.0))
                return best_match, scaled_conf
            else:
                return "Unknown", 0.0
        
        except Exception as e:
            print(f"Error in recognition: {str(e)}")
            return "Error", 0.0
    
    def smooth_detection(self, person_name, confidence, history_size=None):
        """
        Smooth detection results menggunakan history
        Untuk mengurangi flickering
        """
        if history_size is None:
            history_size = config.SMOOTHING_HISTORY_SIZE
        
        if person_name not in self.detection_history:
            self.detection_history[person_name] = []
        
        self.detection_history[person_name].append(confidence)
        
        # Keep only recent history
        if len(self.detection_history[person_name]) > history_size:
            self.detection_history[person_name].pop(0)
        
        # Return average confidence
        avg_confidence = np.mean(self.detection_history[person_name])
        return avg_confidence
    
    def run_realtime(self, camera_id=None):
        """
        Run real-time face recognition dari webcam
        """
        if camera_id is None:
            camera_id = config.CAMERA_ID
        
        print("\n" + "=" * 60)
        print("REAL-TIME FACE RECOGNITION - CNN POWERED")
        print("=" * 60)
        print("Press 'q' to quit")
        print("Press 's' to save screenshot")
        print("Press 'c' to show current config")
        print("=" * 60 + "\n")
        
        # Open camera
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("ERROR: Cannot open camera!")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
        
        frame_count = 0
        fps_start_time = cv2.getTickCount()
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("ERROR: Failed to read frame!")
                break
            
            frame_count += 1
            
            # Flip frame untuk mirror effect
            frame = cv2.flip(frame, 1)
            
            faces = detect_faces_viola_jones(
                frame,
                scale_factor=config.VIOLA_JONES_SCALE_FACTOR,
                min_neighbors=config.VIOLA_JONES_MIN_NEIGHBORS,
                min_size=config.VIOLA_JONES_MIN_SIZE
            )
            
            # Process setiap wajah yang terdeteksi
            for face_coords in faces:
                try:
                    # Extract face ROI
                    face_roi, adjusted_coords = extract_face_roi(
                        frame,
                        face_coords,
                        padding=config.FACE_ROI_PADDING
                    )
                    
                    # Recognize face
                    person_name, confidence = self.recognize_face(face_roi)
                    
                    # Smooth detection
                    smoothed_confidence = self.smooth_detection(person_name, confidence)
                    
                    if smoothed_confidence >= config.COLOR_HIGH_CONFIDENCE:
                        color = (0, 255, 0)  # Green - high confidence
                    elif smoothed_confidence >= config.COLOR_MEDIUM_CONFIDENCE:
                        color = (0, 165, 255)  # Orange - medium confidence
                    else:
                        color = (0, 0, 255)  # Red - low confidence
                    
                    # Draw detection box dengan nama dan confidence
                    frame = draw_detection_box(
                        frame,
                        adjusted_coords,
                        person_name,
                        smoothed_confidence,
                        color=color
                    )
                    
                except Exception as e:
                    print(f"Error processing face: {str(e)}")
                    continue
            
            # Calculate dan display FPS
            if frame_count % 30 == 0:
                fps_end_time = cv2.getTickCount()
                fps = cv2.getTickFrequency() / (fps_end_time - fps_start_time) * 30
                fps_start_time = cv2.getTickCount()
            
            # Display info
            cv2.putText(
                frame,
                f"Faces detected: {len(faces)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            
            # Show frame
            cv2.imshow('Face Recognition System - CNN', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('s'):
                filename = f"screenshot_{frame_count}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved: {filename}")
            elif key == ord('c'):
                self.print_config()
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "=" * 60)
        print("SYSTEM STOPPED")
        print("=" * 60)
    
    def print_config(self):
        """
        Print current configuration
        """
        print("\n" + "=" * 60)
        print("CURRENT CONFIGURATION")
        print("=" * 60)
        print(f"Confidence Threshold: {self.confidence_threshold:.2%}")
        print(f"Distance Threshold: {config.DISTANCE_THRESHOLD}")
        print(f"Viola-Jones Min Neighbors: {config.VIOLA_JONES_MIN_NEIGHBORS}")
        print(f"Viola-Jones Scale Factor: {config.VIOLA_JONES_SCALE_FACTOR}")
        print(f"Smoothing History Size: {config.SMOOTHING_HISTORY_SIZE}")
        print("=" * 60 + "\n")

def main():
    """
    Main function
    """
    # Check if model exists
    if not os.path.exists('models/face_recognition_model.pkl'):
        print("ERROR: Trained model not found!")
        print("Please run train.py first to train the model.")
        exit(1)
    
    if not os.path.exists('models/feature_extractor.h5'):
        print("ERROR: Feature extractor not found!")
        print("Please run train.py first to train the model.")
        exit(1)
    
    # Initialize system
    system = FaceRecognitionSystem(
        model_path='models/face_recognition_model.pkl',
        feature_extractor_path='models/feature_extractor.h5',
        confidence_threshold=config.CONFIDENCE_THRESHOLD
    )
    
    # Run real-time recognition
    system.run_realtime(camera_id=config.CAMERA_ID)

if __name__ == "__main__":
    main()
