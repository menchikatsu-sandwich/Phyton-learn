#!/usr/bin/env python3
"""
Real-time Face Recognition System dengan CNN
Menampilkan kamera dengan deteksi wajah dan confidence score
"""
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
import numpy as np
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

        # Optical flow state
        prev_gray = None
        flow_save_dir = 'flow_frames'
        os.makedirs(flow_save_dir, exist_ok=True)
        flow_frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("ERROR: Failed to read frame!")
                break
            
            frame_count += 1
            
            # Flip frame untuk mirror effect
            frame = cv2.flip(frame, 1)
            # Prepare grayscale frame for optical flow
            gray_for_flow = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Compute Farneback optical flow (between previous and current grayscale frames)
            flow_bgr = None
            try:
                if prev_gray is not None:
                    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray_for_flow, None,
                                                        pyr_scale=0.5, levels=3, winsize=15,
                                                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

                    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    hsv = np.zeros_like(frame)
                    hsv[..., 1] = 255
                    hsv[..., 0] = ang * 180 / np.pi / 2
                    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                    flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

                    # Show mean magnitude on the main frame (useful for debugging / liveness)
                    mean_mag = float(np.mean(mag))
                    cv2.putText(frame, f"Flow mean: {mean_mag:.3f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

                # update previous gray for next iteration
                prev_gray = gray_for_flow
            except Exception as e:
                print(f"Optical flow error: {e}")

            faces = detect_faces_viola_jones(
                frame,
                scale_factor=config.VIOLA_JONES_SCALE_FACTOR,
                min_neighbors=config.VIOLA_JONES_MIN_NEIGHBORS,
                min_size=config.VIOLA_JONES_MIN_SIZE
            )

            # Optional: simple hand detection + finger counting
            total_fingers = 0
            try:
                if config.HAND_DETECTION_ENABLED:
                    # Convert to YCrCb and threshold for skin color
                    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
                    skin_mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
                    # Morphological ops to clean mask
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel, iterations=2)
                    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

                    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    hand_contours = [c for c in contours if cv2.contourArea(c) >= config.HAND_MIN_CONTOUR_AREA]

                    for cnt in hand_contours:
                        # Bounding box
                        x, y, w, h = cv2.boundingRect(cnt)

                        # Convex hull and defects
                        hull = cv2.convexHull(cnt, returnPoints=False)
                        finger_count = 0
                        if hull is not None and len(hull) > 3:
                            defects = cv2.convexityDefects(cnt, hull)
                            if defects is not None:
                                for i in range(defects.shape[0]):
                                    s, e, f, depth = defects[i, 0]
                                    start = tuple(cnt[s][0])
                                    end = tuple(cnt[e][0])
                                    far = tuple(cnt[f][0])

                                    # Compute angle between start-far and end-far
                                    a = np.linalg.norm(np.array(start) - np.array(end))
                                    b = np.linalg.norm(np.array(start) - np.array(far))
                                    c = np.linalg.norm(np.array(end) - np.array(far))
                                    # cosine rule
                                    angle = 0.0
                                    if b * c != 0:
                                        angle = np.degrees(np.arccos((b**2 + c**2 - a**2) / (2 * b * c + 1e-10)))

                                    # depth is multiplied by 1/256 in OpenCV representation
                                    true_depth = depth / 256.0

                                    # heuristics: defect depth relative to contour perimeter and angle
                                    if true_depth > (config.HAND_DEFECT_DEPTH_RATIO * cv2.arcLength(cnt, True)) and angle < config.HAND_DEFECT_ANGLE_THRESH:
                                        finger_count += 1

                        # finger_count defects -> fingers approx = defects + 1
                        detected_fingers = min(finger_count + 1, 5) if finger_count > 0 else 0
                        total_fingers += detected_fingers

                        if config.HAND_SHOW_VIS:
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)
                            cv2.putText(frame, f"Fingers: {detected_fingers}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

                    # show total fingers
                    cv2.putText(frame, f"Total fingers: {total_fingers}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 220, 50), 2)
            except Exception as e:
                print(f"Hand detection error: {e}")
            
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

                    # Log detection to terminal
                    try:
                        print(f"[Frame {frame_count}] Detected: {person_name} | Confidence: {smoothed_confidence:.2%}")
                    except Exception:
                        # Keep logging non-blocking
                        pass
                    
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
            # Show optical flow visualization in separate window (if available)
            if flow_bgr is not None:
                cv2.imshow('Optical Flow (Farneback)', flow_bgr)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('s'):
                filename = f"screenshot_{frame_count}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved: {filename}")
            elif key == ord('f'):
                # Save last flow magnitude visualization (if computed)
                if flow_bgr is not None:
                    flow_filename = os.path.join(flow_save_dir, f"flow_{flow_frame_idx:05d}.png")
                    # Save the magnitude (V channel) as grayscale for compactness
                    mag = cv2.cvtColor(flow_bgr, cv2.COLOR_BGR2GRAY)
                    cv2.imwrite(flow_filename, mag)
                    print(f"Flow frame saved: {flow_filename}")
                    flow_frame_idx += 1
                else:
                    print("No optical flow frame available to save yet.")
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
