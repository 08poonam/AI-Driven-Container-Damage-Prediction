import os
import pandas as pd
from ultralytics import YOLO
from datetime import datetime

def export_predictions():
    model = YOLO("C:/Users/pooja/runs/detect/container_damage_80Epochs/weights/best.pt")
    test_dir = r'C:\Users\pooja\OneDrive\Desktop\ContainerDamageAnalysis\ImageTesting'

    print("🔍 Running predictions...")
    results = model.predict(source=test_dir, save=True, conf=0.25)

    data = []

    for r in results:
        image_path = r.path
        image_name = os.path.basename(image_path)

        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            width = x2 - x1
            height = y2 - y1
            area = width * height

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            aspect_ratio = width / height if height != 0 else 0
            damage_shape = "wide" if width > height else "tall"

            severity_score = round(area * conf, 2)
            severity_label = "High" if area > 20000 else "Low"

            data.append({
                "image": image_name,
                "class": model.names[cls],
                "confidence_percent": round(conf * 100, 2),
                "bbox_width": round(width, 2),
                "bbox_height": round(height, 2),
                "bbox_area": round(area, 2),
                "aspect_ratio": round(aspect_ratio, 3),
                "damage_shape": damage_shape,
                "center_x": round(center_x, 2),
                "center_y": round(center_y, 2),
                "severity_score": severity_score,
                "severity_label": severity_label
            })

    df = pd.DataFrame(data)
    df.to_csv("results_Predicted.csv", index=False)

    print("✅ Exported simplified results to results_for_powerbi.csv")
    print(df.head())

if __name__ == "__main__":
    export_predictions()
