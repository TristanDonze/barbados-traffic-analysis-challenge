import os
import sys
import joblib
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from joblib import Parallel, delayed

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src.dataset.dataset import TrainDataset
from config import SEQ_LEN, EMBARGO, HORIZON, AGGREGATE, INPUT_DIM

FEATURES_PATH = "/Data/barbados/barbados-traffic-analysis-challenge/data/features/final_features.csv"
TRAIN_CSV_PATH = "/Data/barbados/barbados-traffic-analysis-challenge/data/Train.csv"

USE_GPU = True 
N_JOBS_PARALLEL = 4

LGBM_PARAMS = {
    'n_estimators': 500,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'objective': 'multiclass',
    'num_class': 4,
    'metric': 'multi_logloss',
    'random_state': 42,
    'class_weight': 'balanced',
    'device': 'gpu' if USE_GPU else 'cpu',
    'gpu_platform_id': 0,
    'gpu_device_id': 0,
    'max_bin': 63,
    'subsample': 0.8,
    'colsample_bytree': 0.6,
    'n_jobs': 1 if N_JOBS_PARALLEL > 1 else -1,
    'verbose': -1
}

class SuppressStdOutErr:
    """
    Context manager to suppress low-level C/C++ stdout and stderr.
    This kills the annoying '1 warning generated' messages from the GPU compiler.
    """
    def __enter__(self):
        self.save_stdout = os.dup(1)
        self.save_stderr = os.dup(2)
        
        self.devnull = os.open(os.devnull, os.O_WRONLY)
        
        os.dup2(self.devnull, 1)
        os.dup2(self.devnull, 2)

    def __exit__(self, exc_type, exc_value, traceback):
        os.dup2(self.save_stdout, 1)
        os.dup2(self.save_stderr, 2)
        
        os.close(self.devnull)
        os.close(self.save_stdout)
        os.close(self.save_stderr)

def train_single_model(params, X, y, idx):
    """
    Helper function to train one model (for one target column).
    Now wrapped in the silencer to keep output clean.
    """
    with SuppressStdOutErr():
        clf = lgb.LGBMClassifier(**params)
        clf.fit(X, y)
    return clf

class FastMultiOutputClassifier:
    def __init__(self, model_params, n_jobs=4):
        self.params = model_params
        self.n_jobs = n_jobs
        self.estimators_ = []

    def fit(self, X, y):
        n_outputs = y.shape[1]
        print(f"Training {n_outputs} models in parallel (Batch size: {self.n_jobs})...")
        
        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(train_single_model)(self.params, X, y[:, i], i)
            for i in tqdm(range(n_outputs), desc="Parallel Boosting")
        )

    def predict(self, X):
        preds = []
        for clf in self.estimators_:
            preds.append(clf.predict(X))
        return np.column_stack(preds)

def load_and_flatten_data(mode='train', split_ratio=0.8, stats=None):
    print(f"Loading {mode} data...")
    ds = TrainDataset(
        features_path=FEATURES_PATH,
        train_path=TRAIN_CSV_PATH,
        seq_len=SEQ_LEN,
        embargo=EMBARGO,
        horizon=HORIZON,
        split_ratio=split_ratio,
        mode=mode,
        aggregate=AGGREGATE,
        stats=stats
    )
    
    loader = DataLoader(ds, batch_size=4096, shuffle=False, num_workers=4)
    
    X_list = []
    y_list = []
    
    for inputs, targets in tqdm(loader, desc=f"Flattening {mode} set"):
        batch_size = inputs.shape[0]
        X_flat = inputs.view(batch_size, -1).numpy()
        y_np = targets.view(batch_size, -1).numpy()
        X_list.append(X_flat)
        y_list.append(y_np)
        
    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0), ds.mean, ds.std

def calculate_weighted_score(y_true, y_pred):
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    acc = accuracy_score(y_true_flat, y_pred_flat)
    f1 = f1_score(y_true_flat, y_pred_flat, average='macro')
    return (0.7 * f1) + (0.3 * acc), f1, acc

def train_fast():
    print(f"--- FAST GBOOST TRAINING (GPU={USE_GPU}) ---")
    
    X_train, y_train, train_mean, train_std = load_and_flatten_data(mode='train')
    train_stats = (train_mean, train_std)
    X_val, y_val, _, _ = load_and_flatten_data(mode='test', stats=train_stats)
    
    print(f"Train Shape: {X_train.shape}, Val Shape: {X_val.shape}")

    model = FastMultiOutputClassifier(LGBM_PARAMS, n_jobs=N_JOBS_PARALLEL)
    model.fit(X_train, y_train)
    
    print("\n--- EVALUATING ---")
    y_pred_val = model.predict(X_val)
    score, f1, acc = calculate_weighted_score(y_val, y_pred_val)
    
    print(f"Validation Results:")
    print(f"Final Score: {score:.4f} (F1: {f1:.4f}, Acc: {acc:.4f})")
    
    os.makedirs("models", exist_ok=True)
    model_path = f"models/lgbm_fast_agg_{AGGREGATE}.joblib"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_fast()