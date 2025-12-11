import os, sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

from src.dataset.dataset import TrainDataset
from src.models.TCN import UniversalTrafficModel


from config import DROPOUT, SEQ_LEN, EMBARGO, HORIZON, BATCH_SIZE, INPUT_DIM, AGGREGATE, USE_ATTENTION, HIDDEN_CHANNELS, LR, EPOCHS

FEATURES_PATH = "data/features/final_features.csv"
TRAIN_CSV_PATH = "data/Train.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available(
) else "mps" if torch.backends.mps.is_available() else "cpu")

torch.manual_seed(42)

# class EMA:
#     def __init__(self, model, decay=0.999):
#         self.decay = decay
#         self.shadow = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
#     def update(self, model):
#         for n, p in model.named_parameters():
#             if p.requires_grad:
#                 self.shadow[n] = (1.0 - self.decay) * p.detach() + self.decay * self.shadow[n]
#     def apply_shadow(self, model):
#         self.backup = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
#         for n, p in model.named_parameters():
#             if p.requires_grad:
#                 p.data.copy_(self.shadow[n])
#     def restore(self, model):
#         for n, p in model.named_parameters():
#             if p.requires_grad:
#                 p.data.copy_(self.backup[n])

# class FocalLoss(nn.Module):
#     def __init__(self, gamma=2.0, weight=None, reduction='mean'):
#         super().__init__()
#         self.gamma = gamma
#         self.weight = weight
#         self.reduction = reduction
#     def forward(self, inputs, targets):
#         # inputs: (N, C), targets: (N,)
#         logp = torch.nn.functional.log_softmax(inputs, dim=-1)
#         p = logp.exp()
#         ce = torch.nn.functional.nll_loss(logp, targets, weight=self.weight, reduction='none')
#         pt = p.gather(1, targets.unsqueeze(1)).squeeze(1)
#         loss = ce * ((1 - pt) ** self.gamma)
#         return loss.mean() if self.reduction=='mean' else loss.sum()


def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=-1)
            all_preds.append(preds.cpu().numpy().flatten())
            all_targets.append(targets.cpu().numpy().flatten())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    final_score = (0.7 * f1) + (0.3 * acc)

    return final_score, f1, acc


def train():
    print(f"Running on device: {DEVICE}")

    train_ds = TrainDataset(FEATURES_PATH, TRAIN_CSV_PATH, SEQ_LEN,
                            EMBARGO, HORIZON, split_ratio=0.8, mode='train', aggregate=AGGREGATE)
    train_stats = (train_ds.mean, train_ds.std)
    val_ds = TrainDataset(FEATURES_PATH, TRAIN_CSV_PATH, SEQ_LEN, EMBARGO,
                          HORIZON, split_ratio=0.8, mode='test', aggregate=AGGREGATE, stats=train_stats)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    all_targets = []
    for _, y in train_ds:
        all_targets.append(y.numpy().flatten())
    y_all = np.concatenate(all_targets)
    classes = np.unique(y_all)
    weights = compute_class_weight(
        class_weight='balanced', classes=classes, y=y_all)
    weights_tensor = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
    print(f"Weights: {weights}")

    model = UniversalTrafficModel(
        input_dim=INPUT_DIM, hidden_channels=HIDDEN_CHANNELS,
        seq_len=SEQ_LEN, prediction_steps=HORIZON, num_classes=4, dropout=DROPOUT, use_attention=USE_ATTENTION,
        aggregate=AGGREGATE
    ).to(DEVICE)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Infos : \n - Input Dim : {INPUT_DIM} \n - Aggregate : {AGGREGATE} \n - Use Attention : {USE_ATTENTION} \n - Model has {num_params} trainable parameters.\n - Batch Size : {BATCH_SIZE}\n - Hidden Channels : {HIDDEN_CHANNELS}\n - Dropout : {DROPOUT}")

    criterion = nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=0.05)
    # criterion = FocalLoss(gamma=1.0, weight=weights_tensor)
    scaler = GradScaler(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(optimizer, max_lr=LR, steps_per_epoch=steps_per_epoch, epochs=EPOCHS)
    # ema = EMA(model, decay=0.999)

    best_score = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    print("\n--- Starting Training ---")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            with autocast(device_type=DEVICE.type):
                outputs = model(inputs)
                loss = criterion(outputs.view(-1, 4), targets.view(-1))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        score, f1, acc = evaluate(model, val_loader, DEVICE)


        if score > best_score:
            best_score = score
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), f"models/best_model_classic_aggregate_{AGGREGATE}.pth")
            print(
                f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Score: {score:.4f} (F1: {f1:.4f} Acc : {acc:.4f}) Learning Rate : {optimizer.param_groups[0]['lr']:.6f} [BEST]")
        else:
            print(
                f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Score: {score:.4f} (F1: {f1:.4f} Acc : {acc:.4f}) Learning Rate : {optimizer.param_groups[0]['lr']:.6f}")

    print(f"\nTraining Complete. Best Score: {best_score:.4f}")

    model.load_state_dict(best_model_wts)


if __name__ == "__main__":
    train()
