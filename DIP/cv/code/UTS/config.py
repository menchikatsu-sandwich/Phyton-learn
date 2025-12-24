"""
Configuration file untuk Face Recognition System
Ubah parameter di sini untuk tuning sistem
"""

# ============================================
# TRAINING CONFIGURATION
# ============================================

# Jumlah iterasi training untuk neural network
TRAINING_EPOCHS = 100

# Batch size untuk training
BATCH_SIZE = 8

# Learning rate untuk optimizer
LEARNING_RATE = 0.001

# Minimum neighbors untuk Viola-Jones detection
# Semakin tinggi = lebih strict, semakin rendah = lebih sensitif
VIOLA_JONES_MIN_NEIGHBORS = 4  # Default: 5, turunkan ke 4 untuk lebih sensitif

# Scale factor untuk Viola-Jones
# Semakin kecil = lebih akurat tapi lebih lambat
VIOLA_JONES_SCALE_FACTOR = 1.05  # Default: 1.1, turunkan untuk lebih akurat

# Minimum face size untuk deteksi
VIOLA_JONES_MIN_SIZE = (28, 28)  # Default: (30, 30)

# ============================================
# RECOGNITION CONFIGURATION
# ============================================

# Confidence threshold untuk recognition
# Semakin rendah = lebih mudah match, semakin tinggi = lebih strict
CONFIDENCE_THRESHOLD = 0.4  # Default: 0.5, turunkan ke 0.35 untuk lebih sensitif

# Distance threshold untuk feature matching
# Semakin tinggi = lebih mudah match
DISTANCE_THRESHOLD = 2.0  # Default: 1.5, naikkan ke 2.0 untuk lebih mudah match

# Smoothing history size (untuk mengurangi flickering)
SMOOTHING_HISTORY_SIZE = 4

# ============================================
# CAMERA CONFIGURATION
# ============================================

# Camera ID (0 = default camera)
CAMERA_ID = 0

# Camera resolution
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# ============================================
# FEATURE EXTRACTION CONFIGURATION
# ============================================

# Face ROI size untuk feature extraction
FACE_ROI_SIZE = (128, 128)

# Padding untuk face ROI
FACE_ROI_PADDING = 15

# Histogram bins
HISTOGRAM_BINS = 256

# Canny edge detection thresholds
CANNY_THRESHOLD1 = 100
CANNY_THRESHOLD2 = 200

# ============================================
# DISPLAY CONFIGURATION
# ============================================

# Color thresholds untuk confidence visualization
COLOR_HIGH_CONFIDENCE = 0.7  # Green
COLOR_MEDIUM_CONFIDENCE = 0.5  # Orange
COLOR_LOW_CONFIDENCE = 0.0  # Red

# Font settings
FONT_SCALE = 0.6
FONT_THICKNESS = 2

# ============================================
# HAND / MOTION CONFIGURATION
# ============================================
# Enable simple hand detection + finger counting (contour + convexity defects)
HAND_DETECTION_ENABLED = True

# Minimum contour area to be considered a hand
HAND_MIN_CONTOUR_AREA = 2000

# Convexity defect depth threshold (relative to contour perimeter)
HAND_DEFECT_DEPTH_RATIO = 0.01

# Maximum angle (degrees) between defect points to consider a finger gap
HAND_DEFECT_ANGLE_THRESH = 90

# Toggle visualization of per-hand bounding boxes and counts
HAND_SHOW_VIS = True
