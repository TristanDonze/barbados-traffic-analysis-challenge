import os, sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)


import torch
import pandas as pd
from torch.utils.data import DataLoader
from src.models.TCN import UniversalTrafficModel
from src.dataset.dataset import TrainDataset, TestDataset
from config import SEQ_LEN, HORIZON, BATCH_SIZE, INPUT_DIM, AGGREGATE, HIDDEN_CHANNELS


FEATURES_PATH = "data/features/final_features.csv"
TRAIN_CSV_PATH = "data/Train.csv"
TEST_INPUT_PATH = "data/TestInputSegments.csv"
OUTPUT_CSV = "output/submission.csv"

INV_MAP = {0: "free flowing", 1: "light delay", 2: "moderate delay", 3: "heavy delay"}
CAM_NAMES = ["Norman Niles #1", "Norman Niles #2", "Norman Niles #3", "Norman Niles #4"]

def generate_submission():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    MODEL_PATH = f"models/best_model_classic_aggregate_{AGGREGATE}.pth"
    
    train_ds_temp = TrainDataset(
        FEATURES_PATH, 
        TRAIN_CSV_PATH, 
        SEQ_LEN, 
        0, 
        HORIZON, 
        mode='train', 
        aggregate=AGGREGATE
    )
    train_stats = (train_ds_temp.mean, train_ds_temp.std)
    del train_ds_temp

    test_ds = TestDataset(
        FEATURES_PATH, 
        TEST_INPUT_PATH, 
        SEQ_LEN, 
        aggregate=AGGREGATE, 
        stats=train_stats
    )
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    model = UniversalTrafficModel(
        input_dim=INPUT_DIM, 
        hidden_channels=HIDDEN_CHANNELS,
        seq_len=SEQ_LEN, 
        prediction_steps=HORIZON, 
        num_classes=4, 
        dropout=0.3,
        aggregate=AGGREGATE
    ).to(DEVICE)
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False))
    model.eval()

    submission_rows = []

    with torch.no_grad():
        for inputs, last_ids, batch_cam_names in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs) 
            preds = torch.argmax(outputs, dim=-1).cpu().numpy()
            
            last_ids = last_ids.numpy()

            for b in range(inputs.size(0)):
                current_last_id = last_ids[b]
                batch_preds = preds[b]
                
                if AGGREGATE:
                    iterator = enumerate(CAM_NAMES) 
                else:
                    cam_name_from_loader = batch_cam_names[b]
                    iterator = [(0, cam_name_from_loader)]

                for stream_idx, cam_name in iterator:
                    time_indices = [2, 3, 4, 5, 6] 
                    
                    for t_idx in time_indices:
                        pred_class = batch_preds[stream_idx, t_idx]
                        pred_label = INV_MAP[pred_class]
                        
                        target_id_num = current_last_id + (t_idx + 1)
                        row_id = f"time_segment_{target_id_num}_{cam_name}_congestion_enter_rating"
                        
                        submission_rows.append({
                            "ID": row_id,
                            "Target": pred_label,
                            "Target_Accuracy": pred_label
                        })

    df_submission = pd.DataFrame(submission_rows)
    df_submission.to_csv(OUTPUT_CSV, index=False)

if __name__ == "__main__":
    generate_submission()