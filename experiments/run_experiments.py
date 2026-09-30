import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

from src.models import get_logistic_regression, get_random_forest, get_mlp
from src.training import train_mlp, predict_mlp

def split_by_cluster(df, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    clusters = df['cluster_id'].unique()
    np.random.shuffle(clusters)
    
    test_clusters = set(clusters[:int(len(clusters) * test_size)])
    
    train_df = df[~df['cluster_id'].isin(test_clusters)]
    test_df = df[df['cluster_id'].isin(test_clusters)]
    return train_df, test_df

def evaluate_model(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_f1': f1_score(y_true, y_pred, average='macro'),
        'weighted_f1': f1_score(y_true, y_pred, average='weighted'),
        'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='macro', zero_division=0)
    }

def run_experiment(exp_name, df, feature_cols, model_type, split_type):
    print(f"--- Running {exp_name} ({model_type}, {split_type}) ---")
    
    X = df[feature_cols].values
    y = df['label'].values
    
    if split_type == 'random':
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    else: # similarity-aware
        train_df, test_df = split_by_cluster(df, test_size=0.2, random_state=42)
        X_train = train_df[feature_cols].values
        y_train = train_df['label'].values
        X_test = test_df[feature_cols].values
        y_test = test_df['label'].values
        
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    if model_type == 'Logistic Regression':
        model = get_logistic_regression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
    elif model_type == 'Random Forest':
        model = get_random_forest()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
    elif model_type == 'Neural Network':
        # Need val set for early stopping
        X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)
        model = get_mlp(input_dim=X_train.shape[1], num_classes=6)
        model = train_mlp(model, X_tr, y_tr, X_val, y_val, epochs=200, batch_size=64, lr=1e-3, patience=15)
        y_pred = predict_mlp(model, X_test)
        
    metrics = evaluate_model(y_test, y_pred)
    
    # Save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {exp_name}')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.savefig(f'results/confusion_matrices/{exp_name}.png')
    plt.close()
    
    # Save feature importance for random forest
    if model_type == 'Random Forest':
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:20] # top 20
        
        plt.figure(figsize=(10,6))
        plt.title(f"Feature Importances: {exp_name}")
        plt.bar(range(len(indices)), importances[indices], align="center")
        plt.xticks(range(len(indices)), [feature_cols[i] for i in indices], rotation=90)
        plt.xlim([-1, len(indices)])
        plt.tight_layout()
        plt.savefig(f'results/feature_importance/{exp_name}.png')
        plt.close()
        
    print(f"Results for {exp_name}: Acc={metrics['accuracy']:.4f}, Macro-F1={metrics['macro_f1']:.4f}")
    return metrics

def main():
    print("Starting experiments...")
    df = pd.read_csv("data/processed/dataset_clustered.csv")
    
    seq_features = [col for col in df.columns if col.startswith('aa_') or col in ['seq_length', 'mw', 'pi', 'gravy', 'aromaticity', 'instability', 'seq_helix_propensity', 'seq_turn_propensity', 'seq_sheet_propensity']]
    str_features = ['helix_fraction', 'sheet_fraction', 'coil_fraction']
    all_features = seq_features + str_features
    
    experiments = [
        ('E1_Seq_LR_Rand', seq_features, 'Logistic Regression', 'random'),
        ('E2_Seq_RF_Rand', seq_features, 'Random Forest', 'random'),
        ('E3_Seq_NN_Rand', seq_features, 'Neural Network', 'random'),
        
        ('E4_Str_LR_Rand', str_features, 'Logistic Regression', 'random'),
        ('E5_Str_RF_Rand', str_features, 'Random Forest', 'random'),
        ('E6_Str_NN_Rand', str_features, 'Neural Network', 'random'),
        
        ('E7_Both_LR_Rand', all_features, 'Logistic Regression', 'random'),
        ('E8_Both_RF_Rand', all_features, 'Random Forest', 'random'),
        ('E9_Both_NN_Rand', all_features, 'Neural Network', 'random'),
        
        ('E10_Both_RF_SimAware', all_features, 'Random Forest', 'similarity-aware'),
        ('E11_Both_NN_SimAware', all_features, 'Neural Network', 'similarity-aware'),
    ]
    
    results = []
    for exp_name, feats, mod, split in experiments:
        metrics = run_experiment(exp_name, df, feats, mod, split)
        metrics['Experiment'] = exp_name
        metrics['Features'] = 'Sequence' if feats == seq_features else ('Structure' if feats == str_features else 'Both')
        metrics['Model'] = mod
        metrics['Split'] = split
        results.append(metrics)
        
    df_res = pd.DataFrame(results)
    df_res.to_csv("results/metrics/experiment_results.csv", index=False)
    print("All experiments completed successfully.")

if __name__ == "__main__":
    main()
