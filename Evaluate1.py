from ultralytics import YOLO
def main():
    # Load the trained model (make sure the path is correct)
    model = YOLO('C:/Users/pooja/runs/detect/container_damage_50Epochs/weights/best.pt')

    # Evaluate model on the validation set (defined in your data.yaml)
    metrics = model.val()

    # Print results
    print(metrics)
if __name__ == '__main__':
    main()