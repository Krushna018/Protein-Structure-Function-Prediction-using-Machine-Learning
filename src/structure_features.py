import pandas as pd
import os

def main():
    print("Extracting structure features...")
    # Load dataset with sequence features
    df = pd.read_csv("data/processed/dataset_with_seq_features.csv")
    
    # Structure features are already calculated during data collection
    # We will just ensure they are clean and properly typed
    df['helix_fraction'] = df['helix_fraction'].fillna(0.0)
    df['sheet_fraction'] = df['sheet_fraction'].fillna(0.0)
    df['coil_fraction'] = df['coil_fraction'].fillna(1.0)
    
    df.to_csv("data/processed/dataset_features_final.csv", index=False)
    print("Structure features finalized.")

if __name__ == "__main__":
    main()
