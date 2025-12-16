import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src.dataset.dataset import TrainDataset
from src.models.TCN import UniversalTrafficModel

from config import (
    SEQ_LEN, EMBARGO, HORIZON,
    AGGREGATE, INPUT_DIM, HIDDEN_CHANNELS, DROPOUT, USE_ATTENTION, BATCH_SIZE
)

FEATURES_PATH = "/Data/barbados/barbados-traffic-analysis-challenge/data/features/final_features.csv"
TRAIN_CSV_PATH = "/Data/barbados/barbados-traffic-analysis-challenge/data/Train.csv"
MODEL_PATH = "/Data/barbados/barbados-traffic-analysis-challenge/previous_models/best_model_classic_aggregate_True_attn_76.pth" 
OUTPUT_DIR = "/Data/barbados/plots/"

N_REPEATS = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    """Recreates the model architecture using imported config and loads trained weights."""
    print(f"Loading model from {MODEL_PATH}...")
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

    model = UniversalTrafficModel(
        input_dim=INPUT_DIM, 
        hidden_channels=HIDDEN_CHANNELS,
        seq_len=SEQ_LEN, 
        prediction_steps=HORIZON, 
        num_classes=4, 
        dropout=DROPOUT, 
        use_attention=USE_ATTENTION,
        aggregate=AGGREGATE
    ).to(DEVICE)
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

def get_feature_names():
    """Reads the CSV header to get the list of feature names."""
    df = pd.read_csv(FEATURES_PATH, nrows=1)
    return [c for c in df.columns if c.startswith("feature_")]

def score_model(model, loader):
    """
    Runs inference and returns the competition metric: 0.7 * F1 + 0.3 * Accuracy
    """
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs) 
            preds = torch.argmax(outputs, dim=-1)
            all_preds.append(preds.cpu().numpy().flatten())
            all_targets.append(targets.cpu().numpy().flatten())
            
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    
    return (0.7 * f1) + (0.3 * acc)

def calculate_permutation_importance():
    print(f"--- STARTING ROBUST FEATURE IMPORTANCE (Repeats={N_REPEATS}) ---")
    
    model = load_model()
    
    print("Loading dataset...")
    ds = TrainDataset(FEATURES_PATH, TRAIN_CSV_PATH, SEQ_LEN,
                      EMBARGO, HORIZON, split_ratio=0.8, mode='test', aggregate=AGGREGATE)
    
    loader = DataLoader(ds, batch_size=2048, shuffle=False)
    
    print("Calculating Baseline Score...")
    baseline_score = score_model(model, loader)
    print(f"Baseline Score: {baseline_score:.4f}")
    
    feature_names = get_feature_names()
    importance_results = []
    
    print(f"Analyzing {len(feature_names)} features...")
    
    for i, feat_name in enumerate(tqdm(feature_names, desc="Permuting Features")):
        feature_scores = []
        
        for _ in range(N_REPEATS):
            all_preds = []
            all_targets = []
            
            with torch.no_grad():
                for inputs, targets in loader:
                    inputs = inputs.to(DEVICE)
                    
                    shuffled_inputs = inputs.clone()
                    perm_idx = torch.randperm(inputs.size(0))
                    shuffled_inputs[:, :, i] = inputs[perm_idx, :, i]
                    
                    outputs = model(shuffled_inputs)
                    preds = torch.argmax(outputs, dim=-1)
                    
                    all_preds.append(preds.cpu().numpy().flatten())
                    all_targets.append(targets.cpu().numpy().flatten())
            
            y_pred = np.concatenate(all_preds)
            y_true = np.concatenate(all_targets)
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='macro')
            score = (0.7 * f1) + (0.3 * acc)
            feature_scores.append(score)
        
        avg_permuted_score = np.mean(feature_scores)
        importance_drop = baseline_score - avg_permuted_score
        
        importance_results.append({
            'feature': feat_name,
            'importance': importance_drop,
            'avg_permuted_score': avg_permuted_score
        })

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    df_imp = pd.DataFrame(importance_results)
    df_imp = df_imp.sort_values('importance', ascending=False)
    
    csv_path = os.path.join(OUTPUT_DIR, "feature_importance_robust.csv")
    df_imp.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    
    print("\n=== TOP 10 MOST IMPORTANT FEATURES (Averaged) ===")
    print(df_imp[['feature', 'importance']].head(10))
    
    plt.figure(figsize=(10, 5))
    
    plot_data = df_imp.head(10)
    
    sns.barplot(
        data=plot_data,
        x='importance',
        y='feature',
        hue='feature',
        palette='magma',
        legend=False,
        width=0.7
    )
    
    plt.title(f"Top 10 Robust Feature Importance (Avg of {N_REPEATS} Permutations)")
    plt.xlabel("Performance Drop (Baseline - Permuted)")
    plt.tight_layout()
    
    plot_path = os.path.join(OUTPUT_DIR, "feature_importance_robust.png")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    calculate_permutation_importance()