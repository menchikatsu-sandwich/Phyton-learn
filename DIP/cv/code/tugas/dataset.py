# dataset.py
import os
import torch
from PIL import Image
import xml.etree.ElementTree as ET
from torchvision.transforms import ToTensor

def get_class_to_idx():
    with open("classes.txt", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    return {name: i + 1 for i, name in enumerate(classes)}  # 0 = background

class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, split="train"):
        self.split = split
        # Ambil semua file .xml di VOC/split
        self.xml_files = [f for f in os.listdir(f"VOC/{split}") if f.endswith(".xml")]
        if not self.xml_files:
            raise RuntimeError(f"❌ Tidak ada file XML di VOC/{split}! Pastikan konversi sudah dijalankan.")
        print(f"✅ Dataset {split}: {len(self.xml_files)} sampel ditemukan.")

    def __getitem__(self, idx):
        xml_name = self.xml_files[idx]
        img_name = xml_name.replace(".xml", ".jpg")
        
        # Gambar ada di: train/images/, valid/images/, dll
        img_path = os.path.join(self.split, "images", img_name)
        xml_path = os.path.join("VOC", self.split, xml_name)

        # Baca gambar
        img = Image.open(img_path).convert("RGB")

        # Baca XML
        boxes, labels = [], []
        class_to_idx = get_class_to_idx()
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall("object"):
            name = obj.find("name").text
            if name not in class_to_idx:
                continue
            label = class_to_idx[name]
            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
            "iscrowd": torch.zeros(len(boxes), dtype=torch.int64)
        }
        return ToTensor()(img), target

    def __len__(self):
        return len(self.xml_files)