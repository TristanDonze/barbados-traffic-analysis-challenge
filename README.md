# Roundabout Traffic Prediction

<table>
    <tr>
        <td valign="top">
            <img src="https://zindi-public-release.s3.eu-west-2.amazonaws.com/uploads/image_attachment/image/3037/98cefecb-3edb-42b2-b7b3-855a793b098c.png" alt="Roundabout Congestion" style="max-width:400px;height:auto;display:block;margin:0 auto;"/>
        </td>
        <td valign="top">
            This project is an entry to the Zindi competition: <a href="https://zindi.africa/competitions/barbados-traffic-analysis-challenge">Barbados Traffic Analysis Challenge</a>.
            It aims to forecast congestion levels 15 minutes into the future from synchronized multi‑camera video streams by extracting robust traffic‑flow features from raw video and predicting balanced congestion categories using classical or pre‑trained approaches (no backpropagation during training or inference).
        </td>
    </tr>
</table>



## Objective

Predict congestion ratings at different roundabout entry and exit points. The model outputs one of four congestion classes for each time segment.

The challenge does **not allow backpropagation** during training or inference.
The pipeline is therefore designed to rely on feature extraction and classical or pre-trained model strategies rather than end-to-end deep learning approaches.

## Data

The dataset provides:

* Four synchronized video streams of the same roundabout
* Time segmented labels for congestion at entry and exit points
* A target label representing congestion severity

### Congestion Levels

* Free flow
* Light delay
* Moderate delay
* Heavy delay

## Approach

W.I.P


## Evaluation

This challenge uses multi-metric evaluation. There are two error metrics: F1 and Accuracy.

The score on the leaderboard is the weighted mean of two metrics:
* Macro-F1 (70%): measures how well your model performs across all four congestion classes, treating each class equally. This is important because some classes may appear less often in the data.
* Accuracy (30%): measures the overall percentage of correct predictions across all samples.

This means the model should aim for high accuracy and balanced performance across all classes.

For every row in the dataset, submission files should contain 3 columns: id, Target_F1 and Target_Accuracy.
F1 is calculated from the column Target.
Accuracy is calculated from the column Target_Accuracy.

## Repository Structure

```
barbados-traffic-analysis-challenge/
│
├── data/                     # Raw and processed data (not included)
├── notebooks/                # Experiment and exploration notebooks
├── src/
│   ├── preprocessing/        # Video segmentation and cleaning
│   ├── features/             # Computer vision feature extraction
│   ├── models/               # Training and inference scripts
│   └── utils/                # Helper functions
│
├── output/                   # Model outputs and submissions
├── .gitignore                # Git ignore file
├── build_dataset.ipynb       # Notebook to build the dataset
└── README.md                 # Project documentation
```