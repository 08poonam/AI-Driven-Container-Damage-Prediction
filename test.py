# test_debug.py — Run trained YOLOv8 model on test images with detailed debug
# Author: Poonam Gupta | MSc IT Data Analysis Project

from ultralytics import YOLO
import pandas as pd
import os
import sys

def main():
    try:
        # 1️⃣ Load the trained model
        model_path = 'C:/Users/pooja/runs/detect/container_damage_80Epochs/weights/best.pt'
        if not os.path.exists(model_path):
            print(f"❌ Model file not found at: {model_path}")
            sys.exit(1)
        model = YOLO(model_path)
        print("✅ Model loaded successfully!")
        print(f"Classes: {model.names}")

        # 2️⃣ Predict on test images
        test_path = r'C:\Users\pooja\OneDrive\Desktop\ContainerDamageAnalysis\Model\static\ImageTesting'
        if not os.path.exists(test_path):
            print(f"❌ Test image folder not found: {test_path}")
            sys.exit(1)

        print("🔍 Running predictions on test dataset...")
        # Lower confidence threshold for debugging, match training size
        results = model.predict(
            source=test_path,
            save=True,
            conf=0.01,
            imgsz=(192,320)
        )
        print(results)
        print("✅ Predictions complete! Annotated images saved in runs/detect/predict/")

        # 3️⃣ Process predictions into a DataFrame
        records = []
        for result in results:
            image_name = os.path.basename(result.path)
            num_boxes = len(result.boxes)
            print(f"Image: {image_name}, Boxes detected: {num_boxes}")

            for i, box in enumerate(result.boxes):
                cls_id = int(box.cls)
                conf = float(box.conf)
                x_min, y_min, x_max, y_max = box.xyxy[0]
                class_name = result.names[cls_id] if cls_id in result.names else str(cls_id)
                print(f"  Box {i}: Class={class_name}, Conf={conf:.2f}, xyxy=({x_min:.1f},{y_min:.1f},{x_max:.1f},{y_max:.1f})")

                records.append({
                    'image': image_name,
                    'class': class_name,
                    'confidence': conf,
                    'x_min': float(x_min),
                    'y_min': float(y_min),
                    'x_max': float(x_max),
                    'y_max': float(y_max)
                })

        # 4️⃣ Export results to CSV
        if records:
            df = pd.DataFrame(records)
            output_csv = 'container_damage7_predictions_debug.csv'
            df.to_csv(output_csv, index=False)
            print(f"✅ Results saved to: {output_csv}")
            print(df.head())
        else:
            print("⚠️ No predictions found. Try lowering conf threshold or check your images.")

    except Exception as e:
        print(f"❌ Error during prediction: {e}")

if __name__ == '__main__':
    main()
