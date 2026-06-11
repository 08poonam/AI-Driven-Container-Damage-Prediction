# AI-Driven Container Damage Prediction

An intelligent AI-driven system capable of detecting and predicting container damage with high accuracy using computer vision and deep learning techniques.

## 📋 Project Overview

This research project develops an automated solution for detecting and analyzing container damage using state-of-the-art YOLO (You Only Look Once) object detection models. The system leverages computer vision and data analytics to:

- **Detect** container damage in real-time from images
- **Predict** damage severity and classification
- **Analyze** damage patterns with detailed metrics
- **Export** results for business intelligence and dashboarding

## 🎯 Objectives

- Develop high-accuracy container damage detection using computer vision
- Build a scalable AI system for automated damage assessment
- Provide detailed analytics and severity scoring for damage prediction
- Create actionable insights through data visualization and reporting

## 🛠️ Technology Stack

- **Programming Language**: Python 3.x
- **Deep Learning Framework**: YOLO (YOLOv8s, YOLO11n)
- **Core Libraries**:
  - `ultralytics` - YOLO model training and inference
  - `pandas` - Data processing and analysis
  - `opencv` - Computer vision operations
- **Visualization**: Tableau (DashboardTab.twb)
- **Data Export**: CSV, PowerBI compatible formats

## 📁 Project Structure

```
AI-Driven-Container-Damage-Prediction/
├── train.py                          # Model training script
├── test.py                           # Inference and prediction testing
├── Evaluate1.py                      # Model evaluation metrics
├── Export_CSV.py                     # Results export to CSV
├── export_powerbi.py                 # PowerBI format export
├── yolov8s.pt                        # Pre-trained YOLOv8s model weights
├── yolo11n.pt                        # Pre-trained YOLO11n model weights
├── DashboardTab.twb                  # Tableau dashboard configuration
├── AI-Driven Container Damage Prediction.pdf  # Project documentation
│
├── Data Files (Predictions):
│   ├── container_damage_predictions.csv
│   ├── container_damage*_predictions_debug.csv
│   ├── container_damage_cleaned.csv
│   ├── results_Predicted.csv
│   ├── results_for_powerbi.csv
│   └── container_damage6_cleaned.csv
│
└── README.md                         # This file
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install ultralytics pandas opencv-python torch
```

### 1. Training the Model

```bash
python train.py
```

**Configuration** (edit `train.py`):
- Data path: Path to your YOLO-formatted dataset (data.yaml)
- Epochs: 80 (configurable)
- Image size: 480x480
- Batch size: 8

### 2. Running Predictions

```bash
python test.py
```

This script will:
- Load the trained model
- Run inference on test images
- Generate bounding box predictions
- Export results to CSV

### 3. Evaluating Model Performance

```bash
python Evaluate1.py
```

Generates validation metrics:
- Precision, Recall, F1-score
- Mean Average Precision (mAP)
- Confusion matrices

### 4. Exporting Results

```bash
python Export_CSV.py
```

Exports comprehensive predictions with damage analysis:
- Confidence scores
- Bounding box coordinates
- Damage severity estimation
- Damage shape classification

## 📊 Output Files

### Prediction CSV Columns

| Column | Description |
|--------|-------------|
| `image` | Input image filename |
| `class` | Detected damage class |
| `confidence_percent` | Detection confidence (0-100) |
| `bbox_width` | Bounding box width |
| `bbox_height` | Bounding box height |
| `bbox_area` | Total detection area |
| `aspect_ratio` | Width to height ratio |
| `damage_shape` | Classification (wide/tall) |
| `center_x` | Damage center X coordinate |
| `center_y` | Damage center Y coordinate |
| `severity_score` | Computed severity metric |
| `severity_label` | Risk level (High/Low) |

## 🎨 Visualization & Dashboarding

### Tableau Dashboard

The project includes `DashboardTab.twb` for:
- Real-time damage detection visualization
- Severity distribution analysis
- Trend analysis over time
- Geographic/facility-based insights

**To use**:
1. Open Tableau Desktop
2. Load `DashboardTab.twb`
3. Connect to `results_for_powerbi.csv` or PowerBI export

## 📈 Key Features

✅ **High Accuracy Detection** - YOLOv8s/YOLO11n models with 80+ epochs training  
✅ **Real-time Inference** - Fast prediction on batch or single images  
✅ **Damage Severity Scoring** - Quantitative metrics for damage assessment  
✅ **Comprehensive Analytics** - Area, aspect ratio, position analysis  
✅ **Easy Integration** - CSV exports for BI tools and reporting  
✅ **Extensible Framework** - Modular design for enhancement  

## 🔧 Configuration Guide

### Model Selection

- **YOLOv8s**: Balanced performance and accuracy (recommended for production)
- **YOLO11n**: Lightweight, faster inference for edge devices

Edit model path in scripts:
```python
model = YOLO('yolov8s.pt')  # or 'yolo11n.pt'
```

### Inference Parameters

```python
results = model.predict(
    source=test_path,
    save=True,
    conf=0.25,        # Confidence threshold (0-1)
    imgsz=(480, 480)  # Input image size
)
```

## 📝 Usage Examples

### Single Image Prediction

```python
from ultralytics import YOLO

model = YOLO('yolov8s.pt')
results = model.predict(source='image.jpg', conf=0.25)
print(results[0].boxes)  # Access bounding boxes
```

### Batch Processing

```python
results = model.predict(source='path/to/images/', save=True)
for result in results:
    print(f"Image: {result.path}, Detections: {len(result.boxes)}")
```

## 📊 Expected Performance Metrics

Based on training with 80 epochs:
- **Precision**: > 85%
- **Recall**: > 80%
- **mAP@0.5**: > 0.82
- **Inference Speed**: ~20-50ms per image (depending on hardware)

## 🔄 Data Pipeline

```
Raw Images → Model Inference → Prediction Extraction 
    ↓
Severity Analysis → Feature Engineering → CSV Export
    ↓
PowerBI/Tableau Import → Dashboard Visualization
    ↓
Business Insights & Reports
```

## 🤝 Contributing

To extend this project:

1. Add custom damage classes in dataset annotation
2. Modify severity scoring logic in `Export_CSV.py`
3. Integrate additional preprocessing in `test.py`
4. Enhance dashboard visualizations in Tableau

## 📄 Documentation

Refer to `AI-Driven Container Damage Prediction.pdf` for:
- Detailed research methodology
- Dataset description
- Model architecture details
- Results and findings
- Recommendations

## 🛟 Troubleshooting

### Model not found
```
❌ Model file not found at: [path]
```
**Solution**: Verify the path in script matches your trained model location

### No predictions detected
```
⚠️ No predictions found. Try lowering conf threshold
```
**Solution**: Reduce `conf` parameter (e.g., from 0.25 to 0.01) or check image quality

### CSV export issues
```
Error during export
```
**Solution**: Ensure test images exist at specified path and model has sufficient permissions

## 📞 Contact & Support

**Author**: Poonam Gupta  
**Project**: MSc IT Data Analysis Research  
**Repository**: [AI-Driven-Container-Damage-Prediction](https://github.com/08poonam/AI-Driven-Container-Damage-Prediction)

## 📜 License

This project is provided for research and educational purposes.

---

**Last Updated**: June 2026  
**Status**: Active Development  
**Python Version**: 3.8+
