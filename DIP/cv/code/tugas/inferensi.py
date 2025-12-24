# inferensi_lokal.py
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import matplotlib.pyplot as plt
import os
import random

print("🔍 Memuat model untuk inferensi lokal...")

# --- 1. Load model (tanpa GPU) ---
num_classes = len(open("classes.txt").readlines()) + 1

# Gunakan model yang SAMA seperti di Colab
model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=None)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Load bobot (pastikan file ada di sini!)
model.load_state_dict(torch.load("model_tomato_colab.pth", map_location="cpu"))
model.eval()  # penting: mode inferensi

# --- 2. Baca daftar kelas ---
with open("classes.txt", "r") as f:
    classes = ["__background__"] + [line.strip() for line in f.readlines()]

# --- 3. Pilih gambar (misal: ambil yang pertama di valid/images) ---
img_dir = "test/images"
img_files = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]
if not img_files:
    raise FileNotFoundError("❌ Tidak ada file .jpg di valid/images/")

# Bisa ganti dengan nama spesifik, misal: "IMG_0210_JPG.rf.88fef2f797fa888a725ec075a039aa57.jpg"
img_name = random.choice(img_files)
img_path = os.path.join(img_dir, img_name)
print(f"🖼️  Menguji: {img_name}")

# --- 4. Buka dan proses gambar ---
img = Image.open(img_path).convert("RGB")
from torchvision.transforms import ToTensor
img_tensor = ToTensor()(img).unsqueeze(0)  # tambah batch dimension

# --- 5. Deteksi objek ---
with torch.no_grad():
    prediction = model(img_tensor)[0]

# Ambil deteksi dengan skor > 0.5
threshold = 0.5
keep = prediction["scores"] > threshold
boxes = prediction["boxes"][keep].cpu().numpy()
labels = prediction["labels"][keep].cpu().numpy()
scores = prediction["scores"][keep].cpu().numpy()

# --- 6. Tampilkan hasil ---
plt.figure(figsize=(10, 8))
plt.imshow(img)

ax = plt.gca()
for box, label, score in zip(boxes, labels, scores):
    xmin, ymin, xmax, ymax = box
    w, h = xmax - xmin, ymax - ymin
    rect = plt.Rectangle((xmin, ymin), w, h, fill=False, edgecolor="red", linewidth=2)
    ax.add_patch(rect)
    text = f"{classes[label]}: {score:.2f}"
    ax.text(xmin, ymin - 5, text, color="yellow", fontsize=10,
            bbox=dict(facecolor="black", alpha=0.6))

plt.axis("off")
plt.tight_layout()

# Simpan hasil
output_path = "hasil_deteksi_lokal.jpg"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.show()

# Print info
print(f"\n✅ Hasil disimpan sebagai: {output_path}")
print(f"🔍 Ditemukan {len(boxes)} objek:")
for label, score in zip(labels, scores):
    print(f"  - {classes[label]} (skor: {score:.2f})")