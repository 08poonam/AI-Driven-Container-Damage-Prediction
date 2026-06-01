from ultralytics import YOLO

def main():
    # Step 1: Load the model
    model = YOLO('yolo11n.pt')  # or yolov8s.pt if using v8

    # Step 2: Train the model
    results = model.train(
        data=r'C:\Users\pooja\OneDrive\Desktop\ContainerDamageAnalysis\dataset\data.yaml',
        epochs=80,
        imgsz=480,
        batch=8,
        name='container_damage_80Epochs'
    )

if __name__ == '__main__':
    main()
