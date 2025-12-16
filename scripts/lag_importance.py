import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src.dataset.dataset import TrainDataset
from src.models.TCN import UniversalTrafficModel
from config import (
    SEQ_LEN, EMBARGO, HORIZON,
    AGGREGATE, INPUT_DIM, HIDDEN_CHANNELS, DROPOUT, USE_ATTENTION, BATCH_SIZE
)

FEATURES_PATH="/Data/barbados/barbados-traffic-analysis-challenge/data/features/final_features.csv"
TRAIN_CSV_PATH="/Data/barbados/barbados-traffic-analysis-challenge/data/Train.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(fold=0):
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

    # Data paths
    FEATURES_PATH="/Data/barbados/barbados-traffic-analysis-challenge/data/features/final_features.csv"
    TRAIN_CSV_PATH="/Data/barbados/barbados-traffic-analysis-challenge/data/Train.csv"

    # Load your best model
    path = f"/Data/barbados/barbados-traffic-analysis-challenge/previous_models/best_model_classic_aggregate_True_attn_76.pth" 
    # OR if using k-fold: f"models/best_model_fold_{fold}_aggregate_{AGGREGATE}.pth"
    
    if not os.path.exists(path):
        print(f"Warning: Model not found at {path}")
        return None
        
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model

def visualize_attention():
    if not USE_ATTENTION:
        print("Error: USE_ATTENTION is set to False in config. This analysis requires an attention layer.")
        return

    print("--- ANALYZING TEMPORAL ATTENTION ---")
    model = load_model()
    if model is None: return

    # Load a small batch of validation data
    ds = TrainDataset(FEATURES_PATH, TRAIN_CSV_PATH, SEQ_LEN,
                      EMBARGO, HORIZON, split_ratio=0.8, mode='test', aggregate=AGGREGATE)
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    
    inputs, targets = next(iter(loader))
    inputs = inputs.to(DEVICE)

    # --- Hook into the Attention Layer ---
    # We need to capture the 'attn_weights' from the forward pass of self.mha
    attention_weights = []

    def hook_fn(module, input, output):
        # output of MHA is (attn_output, attn_output_weights)
        # attn_output_weights shape: (Batch, Target_Seq, Source_Seq)
        # In self-attention, Target_Seq == Source_Seq == SEQ_LEN
        weights = output[1]
        attention_weights.append(weights.detach().cpu().numpy())

    # Register hook on the MultiheadAttention layer
    handle = model.mha.register_forward_hook(hook_fn)

    # Run Inference
    with torch.no_grad():
        _ = model(inputs)

    # Remove hook
    handle.remove()

    # --- Process Weights ---
    # weights shape: (Batch, Seq_Len, Seq_Len)
    # We want to know: For the LAST time step (prediction time), which past steps were attended to?
    atts = np.concatenate(attention_weights, axis=0) # (Batch, Seq, Seq)
    
    # Average across the batch
    avg_atts = np.mean(atts, axis=0) # (Seq, Seq)

    # We are mostly interested in the attention of the *last* time step (row -1)
    # i.e., "When predicting T=0, how much did I look at T=-1, T=-2...?"
    last_step_attention = avg_atts[-1, :] 

    # --- Plotting ---
    plt.figure(figsize=(12, 5))
    
    # Create x-axis labels (e.g., t-14, t-13 ... t-0)
    time_steps = np.arange(SEQ_LEN)
    labels = [f"t-{((SEQ_LEN-1)-i)}" for i in time_steps]
    
    sns.barplot(x=labels, y=last_step_attention, palette="viridis")
    plt.title("Average Temporal Attention (Last Step Focus)")
    plt.xlabel("Time Lag (History)")
    plt.ylabel("Attention Weight")
    plt.grid(axis='y', alpha=0.3)
    
    output_path = "/Data/barbados/plots/temporal_attention_analysis.png"
    os.makedirs("data/analysis", exist_ok=True)
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    
    # Interpretation
    max_idx = np.argmax(last_step_attention)
    print(f"\nInsight: The model pays most attention to {labels[max_idx]}.")
    if last_step_attention[-1] > 0.5:
        print("Observation: The model is highly 'reactive' (mostly looks at the most recent step).")
    else:
        print("Observation: The model uses 'long-term memory' (distributes attention across history).")

if __name__ == "__main__":
    visualize_attention()