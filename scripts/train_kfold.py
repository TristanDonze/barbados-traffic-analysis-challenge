import os, sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

from src.dataset.dataset import TrainDataset
from src.models.TCN import UniversalTrafficModel

from config import DROPOUT, SEQ_LEN, EMBARGO, HORIZON, BATCH_SIZE, INPUT_DIM, USE_ATTENTION, AGGREGATE, HIDDEN_CHANNELS, LR, EPOCHS_PER_FOLD, N_SPLITS


FEATURES_PATH = "data/features/final_features.csv"
TRAIN_CSV_PATH = "data/Train.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available(
) else "mps" if torch.backends.mps.is_available() else "cpu")

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        outputs_flat = outputs.view(-1, outputs.shape[-1])
        targets_flat = targets.view(-1)

        loss = criterion(outputs_flat, targets_flat)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs_flat.data, 1)
        total += targets_flat.size(0)
        correct += (predicted == targets_flat).sum().item()

    return running_loss / len(loader), 100 * correct / total


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


def train_kfold():
    print(f"Running K-Fold ({N_SPLITS} folds) on device: {DEVICE}")

    full_ds = TrainDataset(
        FEATURES_PATH, TRAIN_CSV_PATH, SEQ_LEN, EMBARGO, HORIZON,
        split_ratio=1.0, mode='train', aggregate=AGGREGATE
    )

    kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    all_targets = []
    for _, y in full_ds:
        all_targets.append(y.numpy().flatten())
    y_all = np.concatenate(all_targets)
    classes = np.unique(y_all)
    weights = compute_class_weight(
        class_weight='balanced', classes=classes, y=y_all)
    weights_tensor = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
    print(f"Global Class Weights: {weights}")

    results = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(range(len(full_ds)))):
        print(f"\n=== FOLD {fold+1}/{N_SPLITS} ===")

        train_sub = Subset(full_ds, train_idx)
        val_sub = Subset(full_ds, val_idx)

        train_loader = DataLoader(
            train_sub, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_sub, batch_size=BATCH_SIZE, shuffle=False)

        model = UniversalTrafficModel(
            input_dim=INPUT_DIM, hidden_channels=HIDDEN_CHANNELS,
            seq_len=SEQ_LEN, prediction_steps=HORIZON, num_classes=4, dropout=DROPOUT, use_attention=USE_ATTENTION,
            aggregate=AGGREGATE
        ).to(DEVICE)
        
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model has {num_params} trainable parameters.")

        criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-4)

        best_score = 0.0
        best_model_wts = copy.deepcopy(model.state_dict())

        for epoch in range(EPOCHS_PER_FOLD):
            model.train()
            running_loss = 0.0

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs.view(-1, 4), targets.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0)
                optimizer.step()
                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)
            score, f1, acc = evaluate(model, val_loader, DEVICE)
            scheduler.step(score)

            if score > best_score:
                best_score = score
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(),
                           f"models/best_model_fold_{fold}_aggregate_{AGGREGATE}.pth")

            if epoch % 5 == 0:
                print(
                    f"  Ep {epoch} | Loss: {avg_loss:.3f} | Score: {score:.3f} (F1: {f1:.4f} Acc : {acc:.4f}) Learning Rate : {optimizer.param_groups[0]['lr']:.6f} ")

        print(f"Fold {fold+1} Best Score: {best_score:.4f}")
        results.append(best_score)

    print(f"\n--- K-FOLD COMPLETE ---")
    print(f"Average Score: {np.mean(results):.4f} (+/- {np.std(results):.4f})")


if __name__ == "__main__":
    train_kfold()