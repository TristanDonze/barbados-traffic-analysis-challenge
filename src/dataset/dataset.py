import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset, DataLoader
from ast import literal_eval

class TrainDataset(Dataset):
    def __init__(self, features_path, train_path, seq_len, embargo, horizon, 
                 split_ratio=0.8, mode='train', aggregate=False, stats=None, random_state=42):
        self.seq_len = seq_len
        self.embargo = embargo
        self.horizon = horizon
        self.aggregate = aggregate

        df_feat = pd.read_csv(features_path)
        df_meta = pd.read_csv(train_path)

        df_feat.columns = df_feat.columns.str.strip()
        df_meta.columns = df_meta.columns.str.strip()
        for col in df_feat.columns:
            if df_feat[col].dtype == object:
                first = df_feat[col].iloc[0]
                if isinstance(first, str) and first.startswith('['):
                    df_feat[col] = df_feat[col].apply(lambda x: literal_eval(x) if isinstance(x, str) else x)

        df = pd.merge(df_feat, df_meta[['videos', 'time_segment_id', 'view_label',
                      'congestion_enter_rating']], on='videos', how='inner')

        rating_map = {"free flowing": 0, "light delay": 1, "moderate delay": 2, "heavy delay": 3}
        target_col = 'congestion_enter_rating'
        if target_col in df.columns:
            df[target_col] = df[target_col].map(rating_map).astype(int)

        feature_cols = [c for c in df.columns if c.startswith("feature_")]

        if self.aggregate:
            df['time_segment_id'] = df['time_segment_id'].astype(int)
            df_pivot = df.pivot(index='time_segment_id', columns='view_label', values=feature_cols)
            df_pivot.columns = [f"{c[0]}_{c[1]}" for c in df_pivot.columns]
            
            meta_pivot = df.pivot(index='time_segment_id', columns='view_label', values=target_col)
            cam_names = sorted(meta_pivot.columns)
            meta_pivot = meta_pivot[cam_names]
            
            video_pivot = df.pivot(index='time_segment_id', columns='view_label', values='videos')
            group_series = video_pivot[cam_names[0]] 

            self.data = df_pivot.join(meta_pivot).join(group_series.rename("group_id")).sort_index()

            x_arrays = []
            for col in df_pivot.columns:
                s = self.data[col]
                valid_val = s.iloc[0]
                if isinstance(valid_val, (list, np.ndarray)):
                    x_arrays.append(np.stack(s.values))
                else:
                    x_arrays.append(s.values.reshape(-1, 1))

            self.X = np.concatenate(x_arrays, axis=1).astype(np.float32) if x_arrays else np.empty((0, 0), dtype=np.float32)
            self.y = self.data[cam_names].values.astype(np.int64)
            self.ids = self.data.index.values
            
            self.video_groups = self.data["group_id"].values 

        else:
            df = df.sort_values(['view_label', 'time_segment_id'])
            x_arrays = []
            for col in feature_cols:
                col_data = df[col].values
                if len(col_data) > 0 and isinstance(col_data[0], (list, np.ndarray)):
                    x_arrays.append(np.stack(col_data))
                else:
                    x_arrays.append(col_data.reshape(-1, 1))

            self.X = np.concatenate(x_arrays, axis=1).astype(np.float32) if x_arrays else np.empty((0, 0), dtype=np.float32)
            self.y = df[[target_col]].values.astype(np.int64)
            self.ids = df['time_segment_id'].values
            self.groups = df['view_label'].values 
            
            self.video_groups = df['videos'].values

        if stats is None:
            self.mean = np.mean(self.X, axis=0)
            self.std = np.std(self.X, axis=0) + 1e-8
        else:
            self.mean, self.std = stats
        self.X = (self.X - self.mean) / self.std

        all_indices = []
        window_groups = []

        if self.aggregate:
            diff_ids = self.ids[1:] - self.ids[:-1]
            is_break = (diff_ids != 1)
            starts = np.concatenate(([0], np.where(is_break)[0] + 1))
            ends = np.concatenate((np.where(is_break)[0] + 1, [len(self.ids)]))
            
            req = self.seq_len + self.embargo + self.horizon
            for s, e in zip(starts, ends):
                if (e - s) >= req:
                    for i in range((e - s) - req + 1):
                        start_idx = s + i
                        all_indices.append((start_idx, start_idx + self.seq_len + self.embargo))
                        window_groups.append(self.video_groups[start_idx])
        else:
            diff_ids = self.ids[1:] - self.ids[:-1]
            diff_group = (self.groups[1:] != self.groups[:-1])
            is_break = (diff_ids != 1) | diff_group
            starts = np.concatenate(([0], np.where(is_break)[0] + 1))
            ends = np.concatenate((np.where(is_break)[0] + 1, [len(self.ids)]))

            req = self.seq_len + self.embargo + self.horizon
            for s, e in zip(starts, ends):
                if (e - s) >= req:
                    for i in range((e - s) - req + 1):
                        start_idx = s + i
                        all_indices.append((start_idx, start_idx + self.seq_len + self.embargo))
                        window_groups.append(self.video_groups[start_idx])
        
        if split_ratio is None:
            raise ValueError("split_ratio must be provided")
        if isinstance(split_ratio, float) and split_ratio >= 1.0:
            train_idx = np.arange(len(all_indices))
            val_idx = np.array([], dtype=int)
        else:
            gss = GroupShuffleSplit(n_splits=1, train_size=split_ratio, random_state=random_state)
            train_idx, val_idx = next(gss.split(all_indices, groups=window_groups))
        
        if mode == 'train':
            self.indices = [all_indices[i] for i in train_idx]
        else:
            self.indices = [all_indices[i] for i in val_idx]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i, t = self.indices[idx]
        y_val = torch.tensor(self.y[t:t+self.horizon], dtype=torch.long)
        if len(y_val.shape) == 2:
             y_val = y_val.permute(1, 0)
        return torch.tensor(self.X[i:i+self.seq_len], dtype=torch.float32), y_val


class TestDataset(Dataset):
    def __init__(self, features_path, test_path, seq_len, aggregate=False, stats=None):
        self.seq_len = seq_len
        self.aggregate = aggregate

        df_feat = pd.read_csv(features_path)
        df_test = pd.read_csv(test_path)
        
        df_feat.columns = df_feat.columns.str.strip()
        df_test.columns = df_test.columns.str.strip()

        for col in df_feat.columns:
            if df_feat[col].dtype == object:
                first = df_feat[col].iloc[0]
                if isinstance(first, str) and first.startswith('['):
                    df_feat[col] = df_feat[col].apply(lambda x: literal_eval(x) if isinstance(x, str) else x)

        df = pd.merge(df_feat, df_test[['videos', 'time_segment_id', 'view_label', 'ID_enter']], on='videos', how='inner')

        feature_cols = [c for c in df.columns if c.startswith("feature_")]

        if self.aggregate:
            df['time_segment_id'] = df['time_segment_id'].astype(int)
            df_pivot = df.pivot(index='time_segment_id', columns='view_label', values=feature_cols)
            df_pivot.columns = [f"{c[0]}_{c[1]}" for c in df_pivot.columns]
            
            self.data = df_pivot.sort_index()
            
            x_arrays = []
            for col in df_pivot.columns:
                s = self.data[col]
                valid_val = s.iloc[0]
                if isinstance(valid_val, (list, np.ndarray)):
                    x_arrays.append(np.stack(s.values))
                else:
                    x_arrays.append(s.values.reshape(-1, 1))
            
            self.X = np.concatenate(x_arrays, axis=1).astype(np.float32)
            self.ids = self.data.index.values
            
            self.cam_names = None 

        else:
            df = df.sort_values(['view_label', 'time_segment_id'])
            
            x_arrays = []
            for col in feature_cols:
                col_data = df[col].values
                if len(col_data) > 0 and isinstance(col_data[0], (list, np.ndarray)):
                    x_arrays.append(np.stack(col_data))
                else:
                    x_arrays.append(col_data.reshape(-1, 1))
            
            self.X = np.concatenate(x_arrays, axis=1).astype(np.float32)
            self.ids = df['time_segment_id'].values
            
            self.cam_names = df['view_label'].values 
            
            self.groups = df['view_label'].values

        if stats is not None:
            self.mean, self.std = stats
            self.X = (self.X - self.mean) / self.std

        self.indices = []
        if self.aggregate:
            ids = self.ids
            diff_ids = ids[1:] - ids[:-1]
            is_break = (diff_ids != 1)
            starts = np.concatenate(([0], np.where(is_break)[0] + 1))
            ends = np.concatenate((np.where(is_break)[0] + 1, [len(ids)]))
            for s, e in zip(starts, ends):
                if (e - s) >= self.seq_len:
                    for i in range((e - s) - self.seq_len + 1):
                        self.indices.append(s + i)
        else:
            ids = self.ids
            groups = self.groups
            diff_ids = ids[1:] - ids[:-1]
            diff_group = (groups[1:] != groups[:-1])
            is_break = (diff_ids != 1) | diff_group
            starts = np.concatenate(([0], np.where(is_break)[0] + 1))
            ends = np.concatenate((np.where(is_break)[0] + 1, [len(ids)]))
            for s, e in zip(starts, ends):
                if (e - s) >= self.seq_len:
                    for i in range((e - s) - self.seq_len + 1):
                        self.indices.append(s + i)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        
        X_tensor = torch.tensor(self.X[i:i+self.seq_len], dtype=torch.float32)
        
        last_time_idx = i + self.seq_len - 1
        last_segment_id = self.ids[last_time_idx]
        
        cam_name = "ALL" if self.aggregate else self.cam_names[last_time_idx]
        
        return X_tensor, last_segment_id, cam_name


if __name__ == "__main__":
    FEATURES = "data/features/final_features.csv"
    TRAIN = "data/Train.csv"
    TEST = "data/TestInputSegments.csv"

    SEQ_LEN = 15
    EMBARGO_PARAM = 0
    HORIZON_PARAM = 7
    AGG = True
    SPLIT = 0.8

    print(f"--- LEAKAGE CHECK (Aggregate={AGG}) ---")

    print("Loading Train Split...")
    train_ds = TrainDataset(
        FEATURES, TRAIN, SEQ_LEN, EMBARGO_PARAM, HORIZON_PARAM, 
        split_ratio=SPLIT, mode='train', aggregate=AGG
    )
    
    print("Loading Val Split...")
    val_ds = TrainDataset(
        FEATURES, TRAIN, SEQ_LEN, EMBARGO_PARAM, HORIZON_PARAM, 
        split_ratio=SPLIT, mode='test', aggregate=AGG
    )

    print(f"\nTrain samples: {len(train_ds)}")
    print(f"Val samples:   {len(val_ds)}")

    train_indices_set = set(train_ds.indices)
    val_indices_set = set(val_ds.indices)
    
    idx_intersection = train_indices_set.intersection(val_indices_set)
    
    print(f"\n[Test 1] Window Overlap Check")
    if len(idx_intersection) == 0:
        print("SUCCESS: No common windows found between Train and Val.")
    else:
        print(f"FAILURE: Found {len(idx_intersection)} overlapping windows!")

    
    train_videos = set()
    for s, _ in train_ds.indices:
        video_id = train_ds.video_groups[s]
        train_videos.add(video_id)
        
    val_videos = set()
    for s, _ in val_ds.indices:
        video_id = val_ds.video_groups[s]
        val_videos.add(video_id)
        
    video_intersection = train_videos.intersection(val_videos)
    
    print(f"\n[Test 2] Video Leakage Check")
    print(f"Unique Videos in Train: {len(train_videos)}")
    print(f"Unique Videos in Val:   {len(val_videos)}")
    
    if len(video_intersection) == 0:
        print("SUCCESS: Train and Val sets contain completely different videos.")
    else:
        print(f"FAILURE: Found {len(video_intersection)} videos present in BOTH sets!")
        print(f"Leaking videos: {list(video_intersection)[:5]}...")

    print("\n[Test 3] DataLoader Smoke Test")
    try:
        ldr = DataLoader(train_ds, batch_size=2, shuffle=True)
        bx, by = next(iter(ldr))
        print(f"Train Batch X: {bx.shape}, Y: {by.shape} - OK")
        
        ldr_val = DataLoader(val_ds, batch_size=2)
        bx_v, by_v = next(iter(ldr_val))
        print(f"Val Batch X:   {bx_v.shape}, Y: {by_v.shape} - OK")
    except Exception as e:
        print(f"Batch Error: {e}")