# Roundabout Traffic Prediction

<table>
    <tr>
        <td valign="top">
            <img src="https://zindi-public-release.s3.eu-west-2.amazonaws.com/uploads/image_attachment/image/3037/98cefecb-3edb-42b2-b7b3-855a793b098c.png" alt="Roundabout Congestion" style="max-width:400px;height:auto;display:block;margin:0 auto;"/>
        </td>
        <td valign="top">
            This project is an entry to the Zindi competition: <a href="https://zindi.africa/competitions/barbados-traffic-analysis-challenge">Barbados Traffic Analysis Challenge</a>.
            It aims to forecast congestion levels 5 minutes into the future from synchronized multi‑camera video streams by extracting robust traffic‑flow features from raw video and predicting congestion categories using a hybrid TCN-Attention architecture.
            <br><br>
            <strong>Performance:</strong> Top 10 Finish (9th/109) with a score of 0.6304.
        </td>
    </tr>
</table>

## Objective

The goal is to predict short-term traffic congestion at the Norman Niles roundabout in Barbados. Using raw video data from four synchronized cameras (each monitoring an approach/exit), the system must:
1.  Analyze a **15-minute** observation window.
2.  Respect a **2-minute** operational embargo (gap).
3.  Forecast congestion levels for the following **5 minutes** on all approaches.

## Data

The dataset consists of video recordings split into 1-minute segments.
* **Input:** 4 synchronized MP4 streams (`normanniles1` to `normanniles4`).
* **Labels:** Categorical congestion ratings provided for training segments.

### Congestion Levels
The target variable is categorized into four classes:
1.  **Free flowing**
2.  **Light delay**
3.  **Moderate delay**
4.  **Heavy delay**

*Note: The dataset is highly imbalanced, with "Free flowing" being the dominant class.*

## Approach

Our solution moves away from end-to-end black-box video models. Instead, we utilize a multi-stage pipeline that engineers interpretable features representing traffic physics, followed by a sequence model that captures temporal dependencies.

### 1. Feature Extraction Pipeline
We transform raw video into tabular time-series data in three stages:

* **Stage 1: Low-Level Signals**
    * **Object Detection:** YOLO + ByteTrack to detect and track vehicles (Cars, Buses, Trucks) across frames.
    * **Occupancy:** Background subtraction within manually masked regions (Entry, Roundabout, Exit) to determine spatial density.
    * **Context:** Cyclic time-of-day and day-of-week encoding.
* **Stage 2: Behavioral Features**
    * Derivation of zone-level statistics: Vehicle counts per zone, average speed estimation (via pixel distance), and stopped vehicle counts.
* **Stage 3: Multi-View Aggregation**
    * Time-series signals are aggregated into 1-minute summary statistics (mean, std, trend).
    * **Crucial Step:** Features from all 4 cameras are concatenated into a single global state vector $X_t$ to capture the interaction between different arms of the roundabout.

### 2. Modeling (TCN + Attention)
We feed the sequence of aggregated features ($L=15$ steps) into a hybrid architecture:
* **Backbone:** A **Temporal Convolutional Network (TCN)** with dilated causal convolutions to extract temporal patterns without looking into the future.
* **Attention:** A **Multi-Head Self-Attention** module to adaptively weight historical time steps (e.g., focusing on traffic conditions 4 minutes ago).
* **Output Heads:** View-specific fully connected layers to predict the 4 congestion classes for the next 5 steps.

### 3. Training Strategy
* **Loss:** Weighted Cross-Entropy to handle class imbalance.
* **Ensemble:** Grouped Stratified 5-Fold Cross-Validation. The final prediction is an average of 5 models trained on different video splits.

## Evaluation

The competition uses a weighted multi-metric evaluation:

* **Macro-F1 (70%):** Measures performance averaged equally across all 4 classes. This penalizes models that ignore rare classes (Moderate/Heavy delay).
* **Accuracy (30%):** Measures the overall percentage of correct predictions.

**Our Public Leaderboard Score:** `0.631` (vs Baseline XGBoost `0.521`).

## Installation and Usage

1.  **Environment Setup**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Data Preparation**
    Ensure your data is organized as shown in the file tree below.
    ```bash
    # Downloads dataset (requires API key or manual setup)
    python scripts/download_dataset_kaggle.py
    ```

3.  **Feature Extraction**
    Run the pipeline stages sequentially to generate the tabular dataset.
    ```bash
    # Stage 1: Yolo & Occupancy
    python scripts/parallelized_features_extraction_stage1.py
    # Stage 2: Behavior & Zone stats
    python scripts/extract_features_stage2.py
    # Stage 3: Aggregation & Normalization
    python scripts/extract_features_stage3.py
    ```

4.  **Training**
    Train the K-Fold ensemble.
    ```bash
    python scripts/train_kfold.py
    ```

5.  **Inference**
    Generate the submission file.
    ```bash
    python scripts/generate_submission_multi_models.py
    ```

## Repository Structure
```
.
├── data/                       # Dataset storage (videos, CSVs)
├── features/                   # Output folder for extracted CSV features
│   ├── stage1/                 # Raw detection/occupancy logs
│   ├── stage2/                 # Derived behavioral features
│   └── final_features.csv      # The final training input
├── models/                     # Saved PyTorch model weights (Best folds)
├── notebooks/                  # EDA, Mask creation, and Loss analysis
├── output/                     # Submission CSVs and training logs
├── pretrained_models/          # Weights for YOLO/Dino
├── scripts/                    # Executable scripts for the pipeline
│   ├── extract_features_*.py   # Feature engineering scripts
│   ├── train_kfold.py          # Main training script (TCN+Attention)
│   ├── train_classic.py        # Baseline training script
│   └── ...
├── src/                        # Core source code
│   ├── dataset/                # PyTorch Dataset loaders
│   ├── features/               # Feature extraction logic (YOLO, OpenCV)
│   ├── mask/                   # ROI Mask generation and utilities
│   ├── models/                 # Neural Network definitions (TCN.py)
│   └── utils/                  # Helper functions
└── README.md
```