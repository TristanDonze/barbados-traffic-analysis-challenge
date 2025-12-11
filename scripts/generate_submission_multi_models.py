import os, sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.models.TCN import MultiHeadSequenceModel
from src.dataset.dataset import TrainDataset, TestDataset

FEATURES_PATH = "data/features/final_features.csv"
TRAIN_CSV_PATH = "data/Train.csv"
SEQ_LEN = 15
HORIZON = 7
EMBARGO = 0
BATCH_SIZE = 32
LR = 0.001
EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


INV_MAP = {0: "free flowing", 1: "light delay", 2: "moderate delay", 3: "heavy delay"}
CAM_NAMES = ["Norman Niles #1", "Norman Niles #2", "Norman Niles #3", "Norman Niles #4"]

def generate_submission_from_models():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    OUTPUT_CSV = "output/submission_ensemble.csv"
    
    train_ds_temp = TrainDataset(FEATURES_PATH, TRAIN_CSV_PATH, SEQ_LEN, 0, HORIZON, mode='train', aggregate=True)
    train_stats = (train_ds_temp.mean, train_ds_temp.std)
    del train_ds_temp
    
    test_ds = TestDataset(
        FEATURES_PATH, "data/TestInputSegments.csv", SEQ_LEN, 
        aggregate=True, stats=train_stats
    )
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    models = []
    for fold in range(5):
        m = MultiHeadSequenceModel(
            input_dim=328, hidden_channels=[64, 64, 64, 64], 
            seq_len=SEQ_LEN, prediction_steps=HORIZON, num_streams=4, num_classes=4
        ).to(DEVICE)
        
        path = f"models/best_model_fold_{fold}.pth"
        m.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=False))
        m.eval()
        models.append(m)
    
    print(f"Loaded {len(models)} models for ensemble.")

    submission_rows = []
    CAM_NAMES = ["Norman Niles #1", "Norman Niles #2", "Norman Niles #3", "Norman Niles #4"]
    INV_MAP = {0: "free flowing", 1: "light delay", 2: "moderate delay", 3: "heavy delay"}

    with torch.no_grad():
        for inputs, last_ids in tqdm(test_loader, desc="Ensemble Inference"):
            inputs = inputs.to(DEVICE)
            
            fold_outputs = []
            for m in models:
                fold_outputs.append(m(inputs))
            
            ensemble_logits = torch.stack(fold_outputs).mean(dim=0)
            
            preds = torch.argmax(ensemble_logits, dim=-1).cpu().numpy()
            last_ids = last_ids.numpy()

            for b in range(inputs.size(0)):
                current_last_id = last_ids[b]
                batch_preds = preds[b]
                
                for c_idx, cam_name in enumerate(CAM_NAMES):
                    time_indices = [2, 3, 4, 5, 6] 
                    for t_idx in time_indices:
                        pred_label = INV_MAP[batch_preds[c_idx, t_idx]]
                        target_id_num = current_last_id + (t_idx + 1)
                        row_id = f"time_segment_{target_id_num}_{cam_name}_congestion_enter_rating"
                        
                        submission_rows.append({
                            "ID": row_id, "Target": pred_label, "Target_Accuracy": pred_label
                        })

    df = pd.DataFrame(submission_rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print("Ensemble submission saved.")

if __name__ == "__main__":
    generate_submission_from_models()