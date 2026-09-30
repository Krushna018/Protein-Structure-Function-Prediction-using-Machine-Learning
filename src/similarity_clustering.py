import pandas as pd
import numpy as np
import os

def get_kmers(seq, k=3):
    return set([seq[i:i+k] for i in range(len(seq)-k+1)])

def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0

def cluster_sequences(df, threshold=0.15):
    # threshold=0.15 on 3-mers roughly corresponds to ~30-40% sequence identity
    clusters = []
    
    print("Computing similarity clusters...")
    for idx, row in df.iterrows():
        seq = row['sequence']
        kmers = get_kmers(seq)
        
        assigned = False
        for c_idx, cluster in enumerate(clusters):
            center_kmers = cluster['center_kmers']
            if jaccard_similarity(kmers, center_kmers) > threshold:
                cluster['members'].append(idx)
                assigned = True
                break
                
        if not assigned:
            clusters.append({
                'center_idx': idx,
                'center_kmers': kmers,
                'members': [idx]
            })
            
    print(f"Formed {len(clusters)} clusters from {len(df)} sequences.")
    
    # Assign cluster ID to dataframe
    cluster_ids = np.zeros(len(df), dtype=int)
    for c_id, cluster in enumerate(clusters):
        for member_idx in cluster['members']:
            cluster_ids[member_idx] = c_id
            
    df['cluster_id'] = cluster_ids
    return df

def main():
    print("Running sequence similarity clustering...")
    df = pd.read_csv("data/processed/dataset_clean.csv")
    df = cluster_sequences(df)
    
    df.to_csv("data/processed/dataset_clustered.csv", index=False)
    print("Clustering complete.")

if __name__ == "__main__":
    main()
