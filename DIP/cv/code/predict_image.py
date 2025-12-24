import cv2
import numpy as np
import torch
import torchvision.models as models
import torch.nn as nn
import os
import argparse

# =====================================================
# CONFIG
# =====================================================
BASE_DIR = r"C:\Users\santo\Documents\coding\PY\DIP\cv"
MODEL_PATH = os.path.join(BASE_DIR, "resnet50_orange_disease_200perclass.pth")
CLASS_NAMES = ["healthy", "canker"]
# CLASS_NAMES = ["canker", "healthy", "multipleDiseases","nutrientDeficiency","youngHealthy"]
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================
# LOAD MODEL
# =====================================================
model = models.resnet50(weights=None)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, len(CLASS_NAMES)),
    nn.Softmax(dim=1)
)
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
# handle checkpoints saved as {'state_dict': ...}
if isinstance(ckpt, dict) and 'state_dict' in ckpt:
    ckpt = ckpt['state_dict']

# only keep params that match in name and shape
model_state = model.state_dict()
compatible_ckpt = {}
skipped_keys = []
for k, v in ckpt.items():
    if k in model_state and v.size() == model_state[k].size():
        compatible_ckpt[k] = v
    else:
        skipped_keys.append(k)

load_res = model.load_state_dict(compatible_ckpt, strict=False)
model.to(DEVICE)
model.eval()
if skipped_keys:
    print(f"[WARN] Skipped {len(skipped_keys)} parameter(s) due to name/shape mismatch. Example keys: {skipped_keys[:10]}")

# Normalisasi ImageNet
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# =====================================================
# INFERENCE FUNCTION
# =====================================================
def predict_image(image_path):
    # Baca gambar
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] Tidak bisa baca gambar: {image_path}")
        return

    orig_h, orig_w = frame.shape[:2]

    # Crop seluruh gambar sebagai "objek" (1 bounding box = full image)
    x1, y1, x2, y2 = 0, 0, orig_w, orig_h

    # Preprocess crop
    crop = frame[y1:y2, x1:x2]
    img = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    img = torch.from_numpy(img).to(DEVICE)

    # Prediksi
    with torch.no_grad():
        output = model(img)
        probs = output.cpu().numpy()[0]
        class_id = int(np.argmax(probs))
        conf = probs[class_id]

    # Gambar bounding box + label
    label = f"{CLASS_NAMES[class_id]} ({conf*100:.2f}%)"
    color = (0, 255, 0) if class_id == 0 else (0, 0, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1 + 5, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Simpan hasil
    output_path = os.path.splitext(image_path)[0] + "_result.jpg"
    cv2.imwrite(output_path, frame)
    print(f"[DONE] Hasil disimpan di: {output_path}")

    # Tampilkan (opsional)
    cv2.imshow("Hasil Deteksi", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Path ke gambar input (misal: sample.jpg)")
    args = parser.parse_args()

    predict_image(args.image_path)