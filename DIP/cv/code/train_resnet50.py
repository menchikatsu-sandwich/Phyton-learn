import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score

# =====================================================
# SET RANDOM SEED (opsional, agar hasil konsisten)
# =====================================================
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# =====================================================
# CONFIG
# =====================================================
BASE_DIR = r"C:\Users\santo\Documents\coding\PY\DIP\cv"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 50
NUM_SAMPLES_PER_CLASS = 500  # ← SESUAI PERMINTAAN: 200 GMBR/LABEL

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Using device: {device}")

# =====================================================
# CUSTOM DATASET CLASS (DENGAN RANDOM SAMPLING)
# =====================================================
class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_per_class=200):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Ambil semua folder kelas di tingkat pertama
        classes = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

        for class_name in classes:
            class_dir = os.path.join(root_dir, class_name)
            # Ambil hanya file gambar (abaikan folder/subfolder)
            all_files = [
                f for f in os.listdir(class_dir)
                if os.path.isfile(os.path.join(class_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            # Random sample
            sampled = random.sample(all_files, min(len(all_files), max_per_class))
            for img_name in sampled:
                self.image_paths.append(os.path.join(class_dir, img_name))
                self.labels.append(self.class_to_idx[class_name])

        print(f"[INFO] Total gambar: {len(self.image_paths)} | Kelas: {list(self.class_to_idx.keys())}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[SKIP] Gambar corrupt: {img_path}")
            # Skip ke gambar berikutnya
            return self.__getitem__((idx + 1) % len(self))
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# =====================================================
# DATA TRANSFORMS
# =====================================================
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# =====================================================
# LOAD DATASET
# =====================================================
print("[INFO] Loading dataset...")
train_dataset = CustomImageDataset(TRAIN_DIR, transform=train_transform, max_per_class=NUM_SAMPLES_PER_CLASS)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# =====================================================
# MODEL: RESNET50 PRETRAINED
# =====================================================
model = models.resnet50(weights='IMAGENET1K_V1')

# Freeze base model
for param in model.parameters():
    param.requires_grad = False

# Ganti head terakhir
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 5)  # 5 kelas
)

model.to(device)

# =====================================================
# LOSS & OPTIMIZER
# =====================================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=1e-4)

# =====================================================
# TRAINING LOOP (FULL 50 EPOCH)
# =====================================================
print("\n[START] Training dimulai...\n")
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    train_preds = []
    train_labels = []

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        train_preds.extend(preds.cpu().numpy())
        train_labels.extend(labels.cpu().numpy())

        # Tampilkan progres per batch (opsional)
        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch {batch_idx + 1}/{len(train_loader)}: Loss = {loss.item():.4f}")

    train_loss /= len(train_loader.dataset)
    train_acc = accuracy_score(train_labels, train_preds)

    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print("-" * 50)

# =====================================================
# SIMPAN MODEL
# =====================================================
model_save_path = os.path.join(BASE_DIR, "resnet50_orange_disease_200perclass.pth")
torch.save(model.state_dict(), model_save_path)
print(f"\n[DONE] Model disimpan di: {model_save_path}")