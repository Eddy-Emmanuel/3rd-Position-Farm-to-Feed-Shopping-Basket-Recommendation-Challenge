# 3rd Position — Farm to Feed Shopping Basket Recommendation Challenge

🏆 **3rd place solution** for the *Farm to Feed Shopping Basket Recommendation Challenge*.

Predicts whether a customer will purchase a given product in the next 1 and 2 weeks (classification), and how much quantity they'll buy (regression), for Farm to Feed's Kenyan customer base.

## Repository Structure

```
.
├── Train.csv                  # Raw training transactions
├── Test.csv                   # Raw test transactions
├── sku_data.csv                # Product/SKU metadata
├── feat_eng_ph1.py             # Stage 1: EDA + core feature engineering
├── feat_eng_ph2.py             # Stage 2: additional leak-free features
├── model_training.py           # Stage 3: model training + ensembling + submission
├── requirements.txt
└── README.md
```

The pipeline has three stages, meant to be run in order:

```
feat_eng_ph1.py  →  feat_eng_ph2.py  →  model_training.py
```

## Pipeline Overview

### 1. `feat_eng_ph1.py` — EDA & Core Feature Engineering
Loads the raw data, explores it, and builds the first generation of features.

- **Inputs:** `Train.csv`, `Test.csv`, `sku_data.csv`
- **What it does:**
  - EDA: missing values, target distributions, class imbalance, numeric/categorical summaries, temporal coverage, customer account age
  - Historical purchase features (rolling counts, purchase rate, average quantity)
  - Customer activity, product popularity, and customer preference aggregations
  - Temporal/seasonal features: cyclical encoding, Kenyan public holidays & festive periods, school terms, agricultural seasons
  - Advanced pattern features: purchase streaks, volatility, product lifecycle, ratio/interaction features
  - Missing value handling and target encoding of categorical features
- **Outputs:** `train_processed.csv`, `test_processed.csv`

### 2. `feat_eng_ph2.py` — Additional Leak-Free Features
Builds a second round of features directly from the raw transactions, engineered to avoid target leakage, then merges them onto the Phase 1 output.

- **Inputs:** `Train.csv`, `Test.csv` (raw), plus `train_processed.csv` / `test_processed.csv` from Phase 1
- **What it does:**
  - Momentum & regularity features (leak-free)
  - Product diversity & basket-composition features
  - Exponential time-decay and price features
  - Additional high-value engineered features
  - Missing-value imputation for the new features
  - Merges the new feature set onto the Phase 1 processed data
- **Intermediate outputs:** `train_enhanced_features_v2.csv`, `test_enhanced_features_v2.csv`
- **Final outputs:** `train_final_with_new_features.csv`, `test_final_with_new_features.csv`

### 3. `model_training.py` — Model Training & Ensembling
Trains the final models and produces the submission file.

- **Inputs:** `train_final_with_new_features.csv`, `test_final_with_new_features.csv`
- **Feature pipelines:**
  - **Pipeline 1** (used for 1-week classification, 1-week regression, 2-week regression): baseline + 8 category target-encodings + 8 recency/momentum features
  - **Pipeline 2** (used for 2-week classification only): baseline + 8 recency/momentum features, with **all** target/label encodings removed to avoid leakage
- **Models per task:**

  | Task | Models | Notes |
  |---|---|---|
  | 1-week purchase (classification) | LightGBM (63%) + Neural Net (37%), ensembled | Pipeline 1 features |
  | 2-week purchase (classification) | LightGBM (79%) + Neural Net (21%), ensembled | Pipeline 2 features, 8-week exponential time decay weighting |
  | 1-week quantity (regression) | XGBoost | Pipeline 1 features, log1p target transform |
  | 2-week quantity (regression) | XGBoost | Pipeline 1 features, log1p target transform, 8-week time decay weighting |

- **Neural net architecture:** Dense(256) → BN → Dropout(0.3) → Dense(128) → BN → Dropout(0.3) → Dense(64) → BN → Dropout(0.2) → Dense(32) → Dropout(0.2) → Dense(1, sigmoid)
- **Validation checks before saving:** no NaNs/infinite values, classification outputs clipped to [0, 1], regression outputs clipped to ≥ 0, ID uniqueness, correct row count
- **Output:** a timestamped/tagged `submission_*.csv` with columns `ID`, `Target_purchase_next_1w`, `Target_qty_next_1w`, `Target_purchase_next_2w`, `Target_qty_next_2w`

## Requirements

See `requirements.txt`. Core dependencies:

```
pandas
numpy
matplotlib
seaborn
scikit-learn
lightgbm
xgboost
tensorflow
```

## Usage

Run the three scripts in order (they were developed as notebook cells, so a Jupyter/Colab/Kaggle environment is recommended, especially for `feat_eng_ph1.py` which uses the `%matplotlib inline` magic):

```bash
# 1. Core feature engineering + EDA
python feat_eng_ph1.py

# 2. Additional leak-free features (merges onto Phase 1 output)
python feat_eng_ph2.py

# 3. Train models and generate submission
python model_training.py
```

Make sure `Train.csv`, `Test.csv`, and `sku_data.csv` are present in the working directory before running Phase 1.

## Notes

- Random seeds are fixed (`42`) across NumPy and TensorFlow for reproducibility.
- A fixed list of low-value features (identified during earlier experimentation) is dropped at the start of `model_training.py` before the two feature pipelines are built.
- Time-decay weighting (8-week half-life-style exponential decay) is used for the 2-week tasks to emphasize recent behavior.

## Result

Finished **3rd place** in the Farm to Feed Shopping Basket Recommendation Challenge.
