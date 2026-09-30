import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import numpy as np
import os

def calculate_sequence_features(seq):
    try:
        pa = ProteinAnalysis(seq)
        
        # Amino acid composition (20 standard AA)
        aa_comp = pa.amino_acids_percent
        
        # MW and pI
        mw = pa.molecular_weight()
        pi = pa.isoelectric_point()
        
        # Secondary structure fraction (sequence-based prediction by ProtParam)
        # Note: this is different from true structural features! We'll just include it as a basic feature.
        seq_helix, seq_turn, seq_sheet = pa.secondary_structure_fraction()
        
        # Hydrophobicity (GRAVY)
        gravy = pa.gravy()
        
        # Aromaticity
        aromaticity = pa.aromaticity()
        
        # Instability index
        instability = pa.instability_index()
        
        features = {
            'seq_length': len(seq),
            'mw': mw,
            'pi': pi,
            'gravy': gravy,
            'aromaticity': aromaticity,
            'instability': instability,
            'seq_helix_propensity': seq_helix,
            'seq_turn_propensity': seq_turn,
            'seq_sheet_propensity': seq_sheet
        }
        
        # Add AA comp
        for aa, perc in aa_comp.items():
            features[f'aa_{aa}'] = perc
            
        return features
    except Exception as e:
        print(f"Failed to calculate sequence features: {e}")
        # Fallback for weird sequences
        return None

def main():
    print("Extracting sequence features...")
    df = pd.read_csv("data/raw/dataset.csv")
    
    feature_list = []
    valid_indices = []
    for idx, row in df.iterrows():
        seq = row['sequence']
        feats = calculate_sequence_features(seq)
        if feats is not None:
            feature_list.append(feats)
            valid_indices.append(idx)
            
    df_valid = df.loc[valid_indices].reset_index(drop=True)
    df_features = pd.DataFrame(feature_list)
    
    # Combine
    df_seq = pd.concat([df_valid, df_features], axis=1)
    
    os.makedirs("data/processed", exist_ok=True)
    df_seq.to_csv("data/processed/dataset_with_seq_features.csv", index=False)
    print(f"Saved sequence features for {len(df_seq)} proteins.")

if __name__ == "__main__":
    main()
