import os
import time
import warnings
from ast import literal_eval

import numpy as np
import pandas as pd

import joblib
from sklearn.base import clone
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score


FEATURES_PATH = "/Data/barbados/barbados-traffic-analysis-challenge/data/features/final_features.csv"
TRAIN_PATH    = "/Data/barbados/barbados-traffic-analysis-challenge/data/Train.csv"

TARGET_COL = "congestion_enter_rating"
SEQ_LEN    = 15
HORIZON    = 7
EMBARGO    = 0

RANDOM_STATE = 42
TRAIN_SIZE   = 0.85

OUT_DIR = "models_logreg_elasticnet_fixed"
os.makedirs(OUT_DIR, exist_ok=True)

FIXED_PARAMS = dict(
    clf__C=0.3,
    clf__l1_ratio=0.2,
    clf__class_weight="balanced",
)

SOLVER_VERBOSE = 1

warnings.filterwarnings("ignore", message=".*'multi_class' was deprecated.*", category=FutureWarning)


def combined_score(y_true, y_pred):
    f1 = f1_score(y_true, y_pred, average="macro")
    acc = accuracy_score(y_true, y_pred)
    return 0.7 * f1 + 0.3 * acc


def _parse_possible_list_cell(x):
    if isinstance(x, str) and x.startswith("[") and x.endswith("]"):
        try:
            return literal_eval(x)
        except Exception:
            return x
    return x


def expand_list_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    obj_cols = [c for c in df.columns if df[c].dtype == object]

    for c in obj_cols:
        df[c] = df[c].apply(_parse_possible_list_cell)
        nonnull = df[c].dropna()
        first = nonnull.iloc[0] if len(nonnull) else None

        if isinstance(first, (list, tuple, np.ndarray)):
            arr = df[c].apply(lambda v: list(v) if isinstance(v, (list, tuple, np.ndarray)) else []).tolist()
            max_len = max((len(v) for v in arr), default=0)
            arr = [v + [np.nan] * (max_len - len(v)) for v in arr]
            new = pd.DataFrame(arr, columns=[f"{c}_{i}" for i in range(max_len)])
            df = df.drop(columns=[c]).join(new)

    return df


def load_merged_df(features_path=FEATURES_PATH, train_path=TRAIN_PATH) -> pd.DataFrame:
    df_feat = pd.read_csv(features_path)
    df_meta = pd.read_csv(train_path)

    df_feat.columns = df_feat.columns.str.strip()
    df_meta.columns = df_meta.columns.str.strip()

    if "video" in df_feat.columns and "videos" not in df_feat.columns:
        df_feat = df_feat.rename(columns={"video": "videos"})

    df_feat = expand_list_columns(df_feat)

    keep_meta = ["videos", "time_segment_id", "view_label", TARGET_COL]
    df = df_feat.merge(df_meta[keep_meta], on="videos", how="inner")

    rating_map = {"free flowing": 0, "light delay": 1, "moderate delay": 2, "heavy delay": 3}
    df[TARGET_COL] = df[TARGET_COL].map(rating_map).astype(int)

    df["time_segment_id"] = df["time_segment_id"].astype(int)
    df = df.sort_values(["view_label", "time_segment_id"]).reset_index(drop=True)
    return df


def pick_feature_cols(df: pd.DataFrame):
    drop = {"videos", "video_path", "original_video_path", "time_segment_id", "view_label", TARGET_COL}
    cols = []
    for c in df.columns:
        if c in drop:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def build_lagged_windows(df, feature_cols, seq_len=SEQ_LEN, embargo=EMBARGO, horizon=HORIZON):
    X_list, Y_list, G_list = [], [], []

    feature_names = []
    for pos in range(seq_len):
        lag = (seq_len - 1) - pos
        for f in feature_cols:
            feature_names.append(f"{f}__lag{lag}")

    for view, g in df.groupby("view_label", sort=False):
        g = g.sort_values("time_segment_id").reset_index(drop=True)

        ids = g["time_segment_id"].values
        breaks = np.where(np.diff(ids) != 1)[0]
        starts = np.r_[0, breaks + 1]
        ends   = np.r_[breaks + 1, len(g)]

        Xv = g[feature_cols].values.astype(np.float32)
        yv = g[TARGET_COL].values.astype(np.int64)
        vv = g["videos"].values

        req = seq_len + embargo + horizon
        for s, e in zip(starts, ends):
            if (e - s) < req:
                continue

            for i in range(s, e - req + 1):
                x_win = Xv[i : i + seq_len].reshape(-1)
                y_fut = yv[i + seq_len + embargo : i + seq_len + embargo + horizon]
                group = vv[i + seq_len - 1]
                X_list.append(x_win)
                Y_list.append(y_fut)
                G_list.append(group)

    X = np.asarray(X_list)
    Y = np.asarray(Y_list)
    groups = np.asarray(G_list)
    return X, Y, groups, feature_names


def explain_and_save(model: Pipeline, feature_names, out_prefix, top_k=20):
    clf = model.named_steps["clf"]
    coefs = clf.coef_
    imp = np.mean(np.abs(coefs), axis=0)

    rows = []
    for name, val in zip(feature_names, imp):
        base, lag_part = name.split("__lag")
        lag = int(lag_part)
        rows.append((name, base, lag, float(val)))

    df_imp = pd.DataFrame(rows, columns=["feature_lag", "base_feature", "lag", "importance"])
    df_imp = df_imp.sort_values("importance", ascending=False)

    by_base = df_imp.groupby("base_feature", as_index=False)["importance"].sum().sort_values("importance", ascending=False)
    by_lag  = df_imp.groupby("lag", as_index=False)["importance"].sum().sort_values("lag", ascending=True)

    print("\nTop feature×lag drivers:")
    print(df_imp.head(top_k).to_string(index=False))

    df_imp.to_csv(out_prefix + "_feature_lag.csv", index=False)
    by_base.to_csv(out_prefix + "_by_base.csv", index=False)
    by_lag.to_csv(out_prefix + "_by_lag.csv", index=False)


def main():
    print("Loading and merging...")
    df = load_merged_df(FEATURES_PATH, TRAIN_PATH)
    feature_cols = pick_feature_cols(df)

    print(f"Merged df shape: {df.shape}")
    print(f"Num feature columns: {len(feature_cols)}")

    X, Y, groups, feature_names = build_lagged_windows(df, feature_cols, SEQ_LEN, EMBARGO, HORIZON)
    print(f"Lagged X shape: {X.shape} Y shape: {Y.shape} groups: {groups.shape}")

    gss = GroupShuffleSplit(n_splits=1, train_size=TRAIN_SIZE, random_state=RANDOM_STATE)
    tr_idx, va_idx = next(gss.split(X, Y[:, 0], groups=groups))
    X_tr, X_va = X[tr_idx], X[va_idx]
    Y_tr, Y_va = Y[tr_idx], Y[va_idx]
    G_tr, G_va = groups[tr_idx], groups[va_idx]

    print(f"Train windows: {len(tr_idx)} | Val windows: {len(va_idx)}")
    print(f"Unique train videos: {len(np.unique(G_tr))} | Unique val videos: {len(np.unique(G_va))}")

    base_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            solver="saga",
            penalty="elasticnet",
            max_iter=100,
            verbose=SOLVER_VERBOSE,
            n_jobs=1,
            random_state=RANDOM_STATE
        ))
    ])

    print("\nFixed params:", FIXED_PARAMS)

    preds_va = np.zeros_like(Y_va)
    models = []

    all_t0 = time.time()
    for k in range(HORIZON):
        print("\n" + "=" * 60)
        print(f"HORIZON k={k+1}/{HORIZON} (predict t+{k+1})")
        print("=" * 60)

        model = clone(base_pipe).set_params(**FIXED_PARAMS)

        t0 = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] Fit started...")
        model.fit(X_tr, Y_tr[:, k])
        print(f"[{time.strftime('%H:%M:%S')}] Fit finished. elapsed={time.time()-t0:.1f}s")

        y_hat = model.predict(X_va)
        preds_va[:, k] = y_hat

        step_score = combined_score(Y_va[:, k], y_hat)
        step_f1 = f1_score(Y_va[:, k], y_hat, average="macro")
        step_acc = accuracy_score(Y_va[:, k], y_hat)

        print(f"Hold-out: score={step_score:.4f} | macroF1={step_f1:.4f} | acc={step_acc:.4f}")

        model_path = os.path.join(OUT_DIR, f"logreg_elasticnet_h{k+1}.joblib")
        joblib.dump({"model": model, "feature_names": feature_names, "params": FIXED_PARAMS}, model_path)
        print("Saved:", model_path)

        out_prefix = os.path.join(OUT_DIR, f"importance_h{k+1}")
        explain_and_save(model, feature_names, out_prefix, top_k=20)

        models.append(model)

    flat_score = combined_score(Y_va.reshape(-1), preds_va.reshape(-1))
    flat_f1 = f1_score(Y_va.reshape(-1), preds_va.reshape(-1), average="macro")
    flat_acc = accuracy_score(Y_va.reshape(-1), preds_va.reshape(-1))

    print("\n" + "=" * 60)
    print("FINAL (all horizons flattened)")
    print(f"score={flat_score:.4f} | macroF1={flat_f1:.4f} | acc={flat_acc:.4f}")
    print(f"Total time: {(time.time()-all_t0)/60:.1f} min")
    print("=" * 60)


if __name__ == "__main__":
    main()