
import pandas as pd
import os

def main():
    input_csv = 'container_damage6_predictions_debug.csv'
    output_csv = 'container_damage6_cleaned.csv'

    if not os.path.exists(input_csv):
        print(f"❌ File not found: {input_csv}")
        return

    print("📥 Loading model predictions...")
    df = pd.read_csv(input_csv)

    print("🧠 Processing data for Power BI...")
    df['bbox_width'] = df['x_max'] - df['x_min']
    df['bbox_height'] = df['y_max'] - df['y_min']
    df['bbox_area'] = df['bbox_width'] * df['bbox_height']
    df['confidence_percent'] = (df['confidence'] * 100).round(2)

    df['normalized_area'] = df['bbox_area'] / df['bbox_area'].max()

    df = df.round({
        'x_min': 2, 'y_min': 2, 'x_max': 2, 'y_max': 2, 
        'bbox_area': 2, 'normalized_area': 4
    })

    df = df[['image', 'class', 'confidence_percent', 
             'bbox_width', 'bbox_height', 'bbox_area', 'normalized_area']]

    df.to_csv(output_csv, index=False)
    print(f"✅ Cleaned data exported successfully to {output_csv}")
    print(df.head(10))

if __name__ == '__main__':
    main()
