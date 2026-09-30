import pandas as pd
import os

def main():
    print("Preprocessing dataset...")
    df = pd.read_csv("data/processed/dataset_features_final.csv")
    
    # Drop rows with missing sequence features (if any)
    df = df.dropna()
    
    # Map EC classes to 0-5
    unique_classes = sorted(df['ec_class'].unique())
    class_map = {c: i for i, c in enumerate(unique_classes)}
    df['label'] = df['ec_class'].map(class_map)
    
    # Check class balance
    print("Class distribution:")
    print(df['ec_name'].value_counts())
    
    # Save the cleaned dataset
    df.to_csv("data/processed/dataset_clean.csv", index=False)
    print(f"Preprocessed dataset saved with {len(df)} samples.")

if __name__ == "__main__":
    main()
