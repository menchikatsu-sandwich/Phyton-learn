import os
'''os environ digunakan untuk mengabaikan konflik yang terjadi antara beberapa library yang versinya tidak sesuai'''
os.environ["KMP_DUPLICATE_LTB_OK"] = "true"

'''gunakan api analytic'''
from ultralytics import YOLO
from pathlib import Path

def main():
  '''path.home ini dapat dihapus jika penyimpanan di dataset di drive D; tapi jangan lupa untuk menyesuaikan kembali'''
  dataset_yaml = Path.home() / "Documents" / "coding" / "PY" / "DIP" / "cv" / "orangeDiseaseyolov8" / "data.yaml"
  assert dataset_yaml.exists(), f"Dataset yaml not found at {dataset_yaml}"

  '''Gunakan pretrained YOLOV8 small dari coco'''
  model = YOLO("yolov8s.pt")
  model.train(
    data=str(dataset_yaml),
    epochs=100,
    imgsz=640,
    batch=16,
    patience=10,
    lr0=0.001,
    lrf=0.01,
    optimizer="AdamW",
    weight_decay=0.001,
    device=0
  )

if __name__ == "__main__":
  main()
  

  