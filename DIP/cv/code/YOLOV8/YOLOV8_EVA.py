import os
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pathlib import Path

os.environ["KMP_DUPLICATE_LTB_OK"] = "true"

def main():
    base_path = Path.home() / "Documents" / "coding" / "PY" / "DIP" / "cv"
    model_path = base_path / "code" / "YOLOV8" / "runs" / "detect" / "train" / "weights" / "best.pt"
    data_yaml = base_path / "orangeDiseaseyolov8" / "data.yaml"
    
    print(f"Using model path: {model_path}")
    print(f"Melakukan Evaluasi model dengan file data: {data_yaml}")
    
    model = YOLO(str(model_path))
    metrics = model.val(data=str(data_yaml), split="val", device=0)
    
    print("Ringkasan Evaluasi:")
    print(f"mAP50: {metrics.box.map50:.3f}")
    print(f"mAP50-95: {metrics.box.map:.3f}")
    print(f"Precision: {metrics.box.precision:.3f}")
    print(f"Recall: {metrics.box.r:.3f}")
    
    class_names = list(model.names.values())
    ap50_list = []
    precision_list = []
    recall_list = []
    
        # Ringkasan hasil global
    print("\nRingkasan Evaluasi Model:")
    print(f". mAP@0.5       : {metrics.box.map50:.3f}")
    print(f". mAP@0.5:0.95  : {metrics.box.map:.3f}")
    print(f". Precision     : {metrics.box.p:.3f}")
    print(f". Recall        : {metrics.box.r:.3f}")

    # Per Kelas
    class_names = list(model.names.values())
    ap50_list = []
    precision_list = []
    recall_list = []

    print("\nHasil Per Kelas:")
    for i, cls in enumerate(class_names):
        p, r, ap50, ap = metrics.box.class_result(i)
        print(f'. {cls_name<25} | Precision: {p:.3f} | Recall: {r:.3f} | AP50: {ap50:.3f} | AP: {ap:.3f}')

        precision_list.append(p)
        recall_list.append(r)
        ap50_list.append(ap50)

    # Visualisasi Grafik
    plot_bar_chart(class_names, ap50_list, "AP@0.5 per kelas", "AP@0.5", "ap50_per_class.png")
    plot_bar_chart(class_names, precision_list, "Precision per kelas", "Precision", "precision_per_class.png")
    plot_bar_chart(class_names, recall_list, "Recall per kelas", "Recall", "recall_per_class.png")

    preds = model.predict(source=str(base_path / "orangeDiseaseyolov8" / "test" / "images"), conf=0.25, save=True, device=0)
    
    if isinstance(preds, list) and hasattr(preds[0], 'save_dir'):
        print(f"Predicted images saved to: {preds[0].save_dir}")
    else:
        print("Prediction did not return expected results.")

def plot_bar_chart(class_names, values, title, ylabel, filename):
    plt.figure(figsize=(10, 6))
    plt.bar(class_names, values, color='skyblue')
    plt.ylim(0, 1)
    plt.title(title)
    plt.xlabel("Kelas")
    plt.ylabel(ylabel)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    save_path = Path.cwd() / filename
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")
    plt.close()
    
if __name__ == "__main__":
    main()