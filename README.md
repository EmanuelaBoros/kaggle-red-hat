# Kaggle Red Hat Business Value Prediction

Legacy Kaggle experiment code for the **Predicting Red Hat Business Value** competition. The task is to predict the binary `outcome` for each activity using activity-level features joined with person-level attributes.

The repository contains a single starter script, `simple_starter.py`, which reads the Kaggle CSV files, performs simple preprocessing, trains an XGBoost binary classifier, and writes a submission file.

## Repository Contents

```text
.
├── simple_starter.py
└── README.md
```

## Expected Data Layout

Kaggle data is not included in this repository. Download the competition files and place them in a local `data/` folder:

```text
data/
├── act_train.csv
├── act_test.csv
└── people.csv
```

The script expects:

- `act_train.csv` with `activity_id`, `people_id`, activity/date features, and `outcome`.
- `act_test.csv` with the same activity features but without `outcome`.
- `people.csv` with person-level features joined by `people_id`.

## Workflow

`simple_starter.py` runs the full pipeline:

1. Read train/test activity files and people metadata.
2. Merge activity rows with people rows on `people_id`.
3. Encode boolean columns as `0`/`1`.
4. Encode categorical object columns with integer mappings.
5. Convert date columns into `year`, `month`, and weekday-style `day` features.
6. Drop unused columns.
7. Train an XGBoost model with `binary:logistic`.
8. Evaluate validation AUC.
9. Write a timestamped Kaggle submission file.

## Features
The script processes activity and people columns such as:

```text
date_x, activity_category, char_1_x, ..., char_10_x,
date_y, char_1_y, ..., char_38
```

After preprocessing, categorical values are replaced by integer ids, booleans are mapped to integers, and dates are expanded into numeric date parts.

## Model

The active model is XGBoost:

```python
objective = "binary:logistic"
eval_metric = "auc"
max_depth = 3
subsample = 0.7
colsample_bytree = 0.7
num_boost_round = 500
early_stopping_rounds = 50
```

The training set is split into train/validation subsets using `train_test_split`, and validation performance is reported with ROC AUC.

## Requirements

This is legacy code and uses older APIs:

- `np.str`, now deprecated.
- `sklearn.cross_validation.train_test_split`, now replaced by `sklearn.model_selection.train_test_split`.
- Older XGBoost parameters such as `silent`.

Original-style dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn xgboost
```

For modern Python environments, update the deprecated NumPy and scikit-learn calls before running.

## Usage

From the repository root:

```bash
mkdir -p data
# place act_train.csv, act_test.csv, and people.csv in data/
python simple_starter.py
```

The script writes a file named like:

```text
submission_<score>_<YYYY-MM-DD-HH-MM>.csv
```

with the Kaggle submission columns:

```text
activity_id,outcome
```

## Notes

- The code is an experiment snapshot, not a packaged library.
- Kaggle data is not included and must be downloaded separately.
- The categorical encoding is fitted independently on train/test columns in the current script; for stricter reproducibility, use a shared mapping fitted on the combined values.
- Several exploratory date-distribution and correlation analyses are included as commented code at the bottom of the script.

