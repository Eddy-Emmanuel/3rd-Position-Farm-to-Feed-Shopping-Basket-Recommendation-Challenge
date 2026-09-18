# ============================================================================
# CELL 1: SETUP & IMPORTS
# ============================================================================

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, mean_absolute_error
import gc
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("FINAL ENSEMBLE SUBMISSION - ALL 4 TASKS")
print("="*80)

print("\nLibrary Versions:")
print(f"  Pandas: {pd.__version__}")
print(f"  NumPy: {np.__version__}")
print(f"  LightGBM: {lgb.__version__}")
print(f"  XGBoost: {xgb.__version__}")
print(f"  TensorFlow: {tf.__version__}")

print(f"\n Hardware:")
print(f"  TensorFlow GPU: {len(tf.config.list_physical_devices('GPU')) > 0}")

# Set seeds
np.random.seed(42)
tf.random.set_seed(42)

print("\n Libraries imported and seeds set")

# ============================================================================
# CELL 2: LOAD & CLEAN DATA
# ============================================================================

print("\n" + "="*80)
print("LOADING DATA")
print("="*80)

# Load training data
train_df = pd.read_csv('train_final_with_new_features.csv')
train_df['week_start'] = pd.to_datetime(train_df['week_start'])

print(f"✓ Train loaded: {train_df.shape}")

# Load test data
test_df = pd.read_csv('test_final_with_new_features.csv')
test_df['week_start'] = pd.to_datetime(test_df['week_start'])

print(f"✓ Test loaded: {test_df.shape}")

print("\n" + "="*80)
print("REMOVING UNHELPFUL FEATURES")
print("="*80)

# Remove features that didn't help (Groups C-G)
features_to_remove = [
    'pct_customer_spending_this_product',
    'product_rank_in_customer_basket',
    'customer_avg_basket_size',
    'customer_total_qty_8w',
    'product_popularity_4w',
    'customer_vs_market_qty_ratio',
    'customer_category_diversity',
    'exp_weighted_purchase_rate',
    'exp_weighted_avg_qty',
    'price_vs_avg',
    'purchase_volatility_8w',
    'std_days_between_purchases',
    'avg_orders_per_week_4w',
    'order_frequency_momentum',
    'is_bulk_buyer',
    'spend_momentum_4w',
    'spend_volatility_8w',
    'product_growth_rate_4w',
    'product_reorder_rate_8w',
    'purchase_interval_stability',
    'weeks_since_first_purchase'
]

for feat in features_to_remove:
    if feat in train_df.columns:
        train_df.drop(feat, axis=1, inplace=True)
    if feat in test_df.columns:
        test_df.drop(feat, axis=1, inplace=True)

print(f"\n✓ Train shape after cleanup: {train_df.shape}")
print(f"✓ Test shape after cleanup: {test_df.shape}")

# ============================================================================
# CELL 3: PREPARE FEATURE SETS (YOUR EXACT PIPELINES)
# ============================================================================

print("\n" + "="*80)
print("PREPARING FEATURE SETS")
print("="*80)

# ============================================================================
# PIPELINE 1: FOR 1W CLASS, 1W REG, 2W REG
# (Keeps 8 category target encodings)
# ============================================================================
print("\n" + "="*80)
print("PIPELINE 1: FOR 1W CLASS, 1W REG, 2W REG")
print("="*80)

train_pipeline1 = train_df.copy()
test_pipeline1 = test_df.copy()

print("\n" + "="*60)
print("REMOVING ID ENCODINGS & ALL LABEL ENCODINGS")
print("="*60)

# ID-related target encodings to remove (causes leakage)
id_encodings_to_remove = [
    'ID_target_encoded',
    'product_unit_variant_id_target_encoded',
    'product_id_target_encoded',
    'product_grade_variant_id_target_encoded'
]

# Get ALL label-encoded features dynamically
label_encoded_features = [col for col in train_pipeline1.columns if 'label_encoded' in col.lower()]

# Combine
features_to_remove_p1 = list(set(id_encodings_to_remove + label_encoded_features))

print(f"\nRemoving features:")
print(f"  - ID target encodings: {len(id_encodings_to_remove)}")
print(f"  - Label encodings: {len(label_encoded_features)}")
print(f"  - Total: {len(features_to_remove_p1)}")

removed_count = 0
for feat in features_to_remove_p1:
    if feat in train_pipeline1.columns:
        train_pipeline1.drop(feat, axis=1, inplace=True)
        removed_count += 1
    if feat in test_pipeline1.columns:
        test_pipeline1.drop(feat, axis=1, inplace=True)

print(f"\n Successfully removed: {removed_count} features")
print(f" Train shape: {train_pipeline1.shape}")
print(f" Test shape: {test_pipeline1.shape}")

# Show remaining target-encoded features (should be 8 category ones)
remaining_target_encoded = [col for col in train_pipeline1.columns if 'target_encoded' in col.lower()]
print(f"\n✓ Remaining TARGET-ENCODED features ({len(remaining_target_encoded)}):")
for feat in remaining_target_encoded:
    print(f"  - {feat}")

# Remove object columns
print("\n" + "="*60)
print("REMOVING OBJECT COLUMNS")
print("="*60)

object_cols_p1 = train_pipeline1.select_dtypes(include=['object']).columns.tolist()
cols_to_drop_p1 = [col for col in object_cols_p1 
                   if col not in ['ID', 'customer_id', 'product_unit_variant_id', 
                                 'week_start', 'customer_created_at']]

if len(cols_to_drop_p1) > 0:
    print(f"\nRemoving {len(cols_to_drop_p1)} object columns:")
    for col in cols_to_drop_p1:
        print(f"  - {col}")
    train_pipeline1.drop(cols_to_drop_p1, axis=1, inplace=True)
    test_pipeline1.drop(cols_to_drop_p1, axis=1, inplace=True)

print(f"\n Final train shape: {train_pipeline1.shape}")
print(f" Final test shape: {test_pipeline1.shape}")

# Prepare features
exclude_cols = [
    'customer_id', 'product_unit_variant_id', 'week_start',
    'Target_purchase_next_1w', 'Target_qty_next_1w',
    'Target_purchase_next_2w', 'Target_qty_next_2w',
    'ID', 'customer_created_at', 'product_grade_variant_id', 'product_id'
]

feature_cols_pipeline1 = [col for col in train_pipeline1.columns if col not in exclude_cols]

# Check for non-numeric
non_numeric_cols_p1 = train_pipeline1[feature_cols_pipeline1].select_dtypes(exclude=[np.number]).columns.tolist()
if len(non_numeric_cols_p1) > 0:
    print(f"\n Removing {len(non_numeric_cols_p1)} non-numeric features")
    feature_cols_pipeline1 = [col for col in feature_cols_pipeline1 if col not in non_numeric_cols_p1]

print(f"\nPIPELINE 1 FEATURES READY:")
print(f"  Total features: {len(feature_cols_pipeline1)}")
print(f"  Includes: Baseline + 8 new (recency+momentum) + 8 target-encoded")

# Extract data for pipeline 1
X_train_pipeline1 = train_pipeline1[feature_cols_pipeline1].values
X_test_pipeline1 = test_pipeline1[feature_cols_pipeline1].values

print(f"  X_train shape: {X_train_pipeline1.shape}")
print(f"  X_test shape: {X_test_pipeline1.shape}")

# ============================================================================
# PIPELINE 2: FOR 2W CLASSIFICATION (CLEAN)
# (NO encodings at all)
# ============================================================================
print("\n" + "="*80)
print("PIPELINE 2: FOR 2W CLASSIFICATION (CLEAN)")
print("="*80)

train_pipeline2 = train_df.copy()
test_pipeline2 = test_df.copy()

print("\n" + "="*60)
print("REMOVING ALL TARGET & LABEL ENCODINGS")
print("="*60)

# Get ALL encoded features dynamically
all_target_encoded = [col for col in train_pipeline2.columns if 'target_encoded' in col.lower()]
all_label_encoded = [col for col in train_pipeline2.columns if 'label_encoded' in col.lower()]
all_encoded_features = list(set(all_target_encoded + all_label_encoded))

print(f"\nRemoving features:")
print(f"  - Target-encoded: {len(all_target_encoded)}")
print(f"  - Label-encoded: {len(all_label_encoded)}")
print(f"  - Total: {len(all_encoded_features)}")

removed_count_p2 = 0
for feat in all_encoded_features:
    if feat in train_pipeline2.columns:
        train_pipeline2.drop(feat, axis=1, inplace=True)
        removed_count_p2 += 1
    if feat in test_pipeline2.columns:
        test_pipeline2.drop(feat, axis=1, inplace=True)

print(f"\n Successfully removed: {removed_count_p2} encoded features")
print(f" Train shape: {train_pipeline2.shape}")
print(f" Test shape: {test_pipeline2.shape}")

# Verify no encodings remain
remaining_encoded = [col for col in train_pipeline2.columns 
                     if 'target_encoded' in col.lower() or 'label_encoded' in col.lower()]
if len(remaining_encoded) > 0:
    print(f"\n⚠ WARNING: {len(remaining_encoded)} encoded features still present!")
    for feat in remaining_encoded:
        print(f"  - {feat}")
else:
    print("\n NO ENCODINGS REMAINING - CLEAN DATASET")

# Remove object columns
print("\n" + "="*60)
print("REMOVING OBJECT COLUMNS")
print("="*60)

object_cols_p2 = train_pipeline2.select_dtypes(include=['object']).columns.tolist()
cols_to_drop_p2 = [col for col in object_cols_p2 
                   if col not in ['ID', 'customer_id', 'product_unit_variant_id', 
                                 'week_start', 'customer_created_at']]

if len(cols_to_drop_p2) > 0:
    print(f"\nRemoving {len(cols_to_drop_p2)} object columns:")
    for col in cols_to_drop_p2:
        print(f"  - {col}")
    train_pipeline2.drop(cols_to_drop_p2, axis=1, inplace=True)
    test_pipeline2.drop(cols_to_drop_p2, axis=1, inplace=True)

print(f"\n Final train shape: {train_pipeline2.shape}")
print(f" Final test shape: {test_pipeline2.shape}")

# Define features
feature_cols_pipeline2 = [col for col in train_pipeline2.columns if col not in exclude_cols]

# Check for non-numeric
non_numeric_cols_p2 = train_pipeline2[feature_cols_pipeline2].select_dtypes(exclude=[np.number]).columns.tolist()
if len(non_numeric_cols_p2) > 0:
    print(f"\n⚠ Removing {len(non_numeric_cols_p2)} non-numeric features")
    feature_cols_pipeline2 = [col for col in feature_cols_pipeline2 if col not in non_numeric_cols_p2]

print(f"\n PIPELINE 2 FEATURES READY:")
print(f"  Total features: {len(feature_cols_pipeline2)}")
print(f"  Includes: Baseline + 8 new (recency+momentum) - NO encodings")

# Extract data for pipeline 2
X_train_pipeline2 = train_pipeline2[feature_cols_pipeline2].values
X_test_pipeline2 = test_pipeline2[feature_cols_pipeline2].values

print(f"  X_train shape: {X_train_pipeline2.shape}")
print(f"  X_test shape: {X_test_pipeline2.shape}")

# ============================================================================
# VERIFY 8 NEW FEATURES IN BOTH PIPELINES
# ============================================================================
print("\n" + "="*80)
print("VERIFYING 8 NEW FEATURES")
print("="*80)

new_features_8 = [
    'days_since_last_purchase_any',
    'days_since_last_purchase_this_product',
    'consecutive_weeks_purchased',
    'consecutive_weeks_not_purchased',
    'purchase_momentum_4w',
    'quantity_momentum_4w',
    'is_regular_customer_this_product',
    'avg_days_between_purchases'
]

found_p1 = [f for f in new_features_8 if f in feature_cols_pipeline1]
found_p2 = [f for f in new_features_8 if f in feature_cols_pipeline2]

print(f"\nPipeline 1: {len(found_p1)}/8 new features found")
print(f"Pipeline 2: {len(found_p2)}/8 new features found")

if len(found_p1) < 8:
    print("\n WARNING: Pipeline 1 missing some new features!")
if len(found_p2) < 8:
    print("\nWARNING: Pipeline 2 missing some new features!")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print(" ALL FEATURE SETS PREPARED")
print("="*80)

print(f"\nPipeline 1 (1W Class, 1W Reg, 2W Reg):")
print(f"  Features: {len(feature_cols_pipeline1)}")
print(f"  = Baseline + 8 target encodings + 8 new")

print(f"\nPipeline 2 (2W Class only):")
print(f"  Features: {len(feature_cols_pipeline2)}")
print(f"  = Baseline + 8 new (NO encodings)")

print("\n Using YOUR exact pipeline logic")

# ============================================================================
# CELL 4: MODEL PARAMETERS
# ============================================================================

print("\n" + "="*80)
print("DEFINING MODEL PARAMETERS")
print("="*80)

# 1W Classification - Optimized parameters
params_lgb_class_1w =  {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'verbosity': -1,
    'seed': 42,
    'learning_rate': 0.011858906685575274,
    'num_leaves': 66,
    'max_depth': 4,
    'min_child_samples': 88,
    'lambda_l1': 1.2465962536551158,
    'lambda_l2': 0.6617960497052984,
    'feature_fraction': 0.4381350101716142,
    'bagging_fraction': 0.5865893930293973,
    'bagging_freq': 3,
    'scale_pos_weight': 47.5
}
num_boost_round_1w_class = 355

# 2W Classification - Use hybrid params
params_lgb_class_2w = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'verbosity': -1,
    'seed': 42,
    'learning_rate': 0.011858906685575274,
    'num_leaves': 66,
    'max_depth': 4,
    'min_child_samples': 88,
    'lambda_l1': 1.2465962536551158,
    'lambda_l2': 0.6617960497052984,
    'feature_fraction': 0.4381350101716142,
    'bagging_fraction': 0.5865893930293973,
    'bagging_freq': 3,
    'scale_pos_weight': 35
}
num_boost_round_2w_class = 350

# XGBoost Regression Parameters
params_xgb_reg = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'tree_method': 'hist',
    'learning_rate': 0.05,
    'max_depth': 6,
    'min_child_weight': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.1,
    'reg_alpha': 1.0,
    'reg_lambda': 1.0,
    'seed': 42,
    'verbosity': 0
}
num_boost_round_reg = 500

# Neural Network Parameters
nn_params = {
    'learning_rate': 0.001,
    'epochs': 50,
    'batch_size': 1024,
    'early_stopping_patience': 10
}

print(" Parameters defined:")
print(f"  1W Classification: {num_boost_round_1w_class} rounds")
print(f"  2W Classification: {num_boost_round_2w_class} rounds")
print(f"  Regressions: {num_boost_round_reg} rounds")
print(f"  Neural Networks: Up to {nn_params['epochs']} epochs")

# ============================================================================
# CELL 5: HELPER FUNCTIONS
# ============================================================================

print("\n" + "="*80)
print("DEFINING HELPER FUNCTIONS")
print("="*80)

def calculate_time_weights(df, decay_weeks=8):
    """Calculate exponential decay time weights"""
    max_week = df['week_start'].max()
    weights = []
    for week in df['week_start']:
        weeks_ago = (max_week - week).days / 7
        weight = np.exp(-weeks_ago / decay_weeks)
        weights.append(weight)
    weights = np.array(weights)
    weights = weights / weights.mean()
    return weights

def create_nn_classification_model(input_dim):
    """Create neural network for binary classification"""
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=nn_params['learning_rate']),
        loss='binary_crossentropy',
        metrics=['AUC']
    )
    
    return model

# ============================================================================
# CELL 6: TRAIN LIGHTGBM MODELS (ALL 4 TASKS)
# ============================================================================

print("\n" + "="*80)
print("TRAINING LIGHTGBM MODELS")
print("="*80)

# Extract targets from pipeline 1
y_train_1w_class = train_pipeline1['Target_purchase_next_1w'].values
y_train_1w_reg = train_pipeline1['Target_qty_next_1w'].values
y_train_2w_reg = train_pipeline1['Target_qty_next_2w'].values

# Extract target from pipeline 2
y_train_2w_class = train_pipeline2['Target_purchase_next_2w'].values

# ============================================================================
# 1W Classification (Pipeline 1 - WITH encodings)
# ============================================================================
print("\n" + "="*80)
print("1. TRAINING 1W CLASSIFICATION")
print("="*80)

print(f"Using Pipeline 1: {len(feature_cols_pipeline1)} features (with 8 target encodings)")
print(f"Target: {y_train_1w_class.mean()*100:.2f}% positive")

dtrain_1w_class = lgb.Dataset(X_train_pipeline1, label=y_train_1w_class)

print(f"\nTraining LightGBM for {num_boost_round_1w_class} rounds...")

lgb_model_1w_class = lgb.train(
    params_lgb_class_1w,
    dtrain_1w_class,
    num_boost_round=num_boost_round_1w_class,
    callbacks=[lgb.log_evaluation(50)]
)

print(f"✓ 1W Classification trained")

del dtrain_1w_class
gc.collect()

# ============================================================================
# 2W Classification (Pipeline 2 - NO encodings, with 8-week decay)
# ============================================================================
print("\n" + "="*80)
print("2. TRAINING 2W CLASSIFICATION")
print("="*80)

print(f"Using Pipeline 2: {len(feature_cols_pipeline2)} features (NO encodings)")
print(f"Target: {y_train_2w_class.mean()*100:.2f}% positive")

# Calculate time weights
train_weights_2w = calculate_time_weights(train_pipeline2, decay_weeks=8)

print(f"Time weights: min={train_weights_2w.min():.3f}, max={train_weights_2w.max():.3f}")

dtrain_2w_class = lgb.Dataset(X_train_pipeline2, label=y_train_2w_class, weight=train_weights_2w)

print(f"\nTraining LightGBM for {num_boost_round_2w_class} rounds (with 8-week decay)...")

lgb_model_2w_class = lgb.train(
    params_lgb_class_2w,
    dtrain_2w_class,
    num_boost_round=num_boost_round_2w_class,
    callbacks=[lgb.log_evaluation(50)]
)

print(f"✓ 2W Classification trained (with time weights)")

del dtrain_2w_class
gc.collect()

# ============================================================================
# 1W Regression (Pipeline 1 - WITH encodings)
# ============================================================================
print("\n" + "="*80)
print("3. TRAINING 1W REGRESSION")
print("="*80)

print(f"Using Pipeline 1: {len(feature_cols_pipeline1)} features (with 8 target encodings)")
print(f"Target: mean={y_train_1w_reg.mean():.2f}, median={np.median(y_train_1w_reg):.2f}")

# Log transform
y_train_1w_reg_log = np.log1p(y_train_1w_reg)

dtrain_1w_reg = xgb.DMatrix(X_train_pipeline1, label=y_train_1w_reg_log)

print(f"\nTraining XGBoost for {num_boost_round_reg} rounds...")

xgb_model_1w_reg = xgb.train(
    params_xgb_reg,
    dtrain_1w_reg,
    num_boost_round=num_boost_round_reg,
    verbose_eval=50
)

print(f"✓ 1W Regression trained")

del dtrain_1w_reg
gc.collect()

# ============================================================================
# 2W Regression (Pipeline 1 - WITH encodings, with 8-week decay)
# ============================================================================
print("\n" + "="*80)
print("4. TRAINING 2W REGRESSION")
print("="*80)

print(f"Using Pipeline 1: {len(feature_cols_pipeline1)} features (with 8 target encodings)")
print(f"Target: mean={y_train_2w_reg.mean():.2f}, median={np.median(y_train_2w_reg):.2f}")

# Log transform
y_train_2w_reg_log = np.log1p(y_train_2w_reg)

# Calculate time weights
train_weights_2w_reg = calculate_time_weights(train_pipeline1, decay_weeks=8)

print(f"Time weights: min={train_weights_2w_reg.min():.3f}, max={train_weights_2w_reg.max():.3f}")

dtrain_2w_reg = xgb.DMatrix(X_train_pipeline1, label=y_train_2w_reg_log, weight=train_weights_2w_reg)

print(f"\nTraining XGBoost for {num_boost_round_reg} rounds (with 8-week decay)...")

xgb_model_2w_reg = xgb.train(
    params_xgb_reg,
    dtrain_2w_reg,
    num_boost_round=num_boost_round_reg,
    verbose_eval=50
)

print(f" 2W Regression trained (with time weights)")

del dtrain_2w_reg
gc.collect()

print("\n" + "="*80)
print(" ALL LIGHTGBM MODELS TRAINED")
print("="*80)

# ============================================================================
# CELL 7: TRAIN NEURAL NETWORKS (2 CLASSIFICATION TASKS)
# ============================================================================

print("\n" + "="*80)
print("TRAINING NEURAL NETWORKS")
print("="*80)

# ============================================================================
# 1W Classification Neural Network (Pipeline 1 - WITH encodings)
# ============================================================================
print("\n" + "="*80)
print("1. TRAINING 1W CLASSIFICATION (NEURAL NETWORK)")
print("="*80)

print(f"Using Pipeline 1: {len(feature_cols_pipeline1)} features (with 8 target encodings)")

# Scale features
scaler_1w_class = StandardScaler()
X_train_1w_class_scaled = scaler_1w_class.fit_transform(X_train_pipeline1)

print(f"   Train samples: {X_train_1w_class_scaled.shape}")
print(f"   Positive rate: {y_train_1w_class.mean()*100:.2f}%")

# Calculate class weights
pos_weight_1w = (y_train_1w_class == 0).sum() / (y_train_1w_class == 1).sum()
class_weight_1w = {0: 1.0, 1: pos_weight_1w}

print(f"   Class weight for positive class: {pos_weight_1w:.1f}")

# Create model
nn_model_1w_class = create_nn_classification_model(X_train_1w_class_scaled.shape[1])

print("\n📋 Model Architecture:")
nn_model_1w_class.summary()

# Callbacks
early_stop_1w = callbacks.EarlyStopping(
    monitor='loss',
    patience=nn_params['early_stopping_patience'],
    mode='min',
    restore_best_weights=True,
    verbose=1
)

reduce_lr_1w = callbacks.ReduceLROnPlateau(
    monitor='loss',
    factor=0.5,
    patience=5,
    mode='min',
    min_lr=1e-6,
    verbose=1
)

# Train
print("\n Training Neural Network...")
history_1w = nn_model_1w_class.fit(
    X_train_1w_class_scaled, y_train_1w_class,
    epochs=nn_params['epochs'],
    batch_size=nn_params['batch_size'],
    class_weight=class_weight_1w,
    callbacks=[early_stop_1w, reduce_lr_1w],
    verbose=1
)

print(f"\n✓ 1W Classification NN trained")

# Check training predictions
train_pred_nn_1w = nn_model_1w_class.predict(X_train_1w_class_scaled[:10000], verbose=0).flatten()
print(f"   Training predictions (first 10k): mean={train_pred_nn_1w.mean():.4f}")

# ============================================================================
# 2W Classification Neural Network (Pipeline 2 - NO encodings)
# ============================================================================
print("\n" + "="*80)
print("2. TRAINING 2W CLASSIFICATION (NEURAL NETWORK)")
print("="*80)

print(f"Using Pipeline 2: {len(feature_cols_pipeline2)} features (NO encodings)")

# Scale features
scaler_2w_class = StandardScaler()
X_train_2w_class_scaled = scaler_2w_class.fit_transform(X_train_pipeline2)

print(f"   Train samples: {X_train_2w_class_scaled.shape}")
print(f"   Positive rate: {y_train_2w_class.mean()*100:.2f}%")

# Calculate class weights
pos_weight_2w = (y_train_2w_class == 0).sum() / (y_train_2w_class == 1).sum()
class_weight_2w = {0: 1.0, 1: pos_weight_2w}

print(f"   Class weight for positive class: {pos_weight_2w:.1f}")

# Create model
nn_model_2w_class = create_nn_classification_model(X_train_2w_class_scaled.shape[1])

print("\n Model Architecture:")
nn_model_2w_class.summary()

# Callbacks
early_stop_2w = callbacks.EarlyStopping(
    monitor='loss',
    patience=nn_params['early_stopping_patience'],
    mode='min',
    restore_best_weights=True,
    verbose=1
)

reduce_lr_2w = callbacks.ReduceLROnPlateau(
    monitor='loss',
    factor=0.5,
    patience=5,
    mode='min',
    min_lr=1e-6,
    verbose=1
)

# Train
print("\n Training Neural Network...")
history_2w = nn_model_2w_class.fit(
    X_train_2w_class_scaled, y_train_2w_class,
    epochs=nn_params['epochs'],
    batch_size=nn_params['batch_size'],
    class_weight=class_weight_2w,
    callbacks=[early_stop_2w, reduce_lr_2w],
    verbose=1
)

print(f"\n✓ 2W Classification NN trained")

# Check training predictions
train_pred_nn_2w = nn_model_2w_class.predict(X_train_2w_class_scaled[:10000], verbose=0).flatten()
print(f"   Training predictions (first 10k): mean={train_pred_nn_2w.mean():.4f}")

print("\n" + "="*80)
print("ALL NEURAL NETWORKS TRAINED")
print("="*80)

# ============================================================================
# CELL 8: PREDICT ON TEST & ENSEMBLE
# ============================================================================

print("\n" + "="*80)
print("GENERATING TEST PREDICTIONS")
print("="*80)

# ============================================================================
# 1W Classification - ENSEMBLE (63% LGBM, 37% NN)
# Uses Pipeline 1 (WITH encodings)
# ============================================================================
print("\n" + "="*80)
print("1. 1W CLASSIFICATION PREDICTIONS")
print("="*80)

print(f"Using Pipeline 1: {len(feature_cols_pipeline1)} features (with encodings)")

# LightGBM prediction
test_pred_lgb_1w_class = lgb_model_1w_class.predict(X_test_pipeline1)
print(f"    LightGBM predictions: mean={test_pred_lgb_1w_class.mean():.4f}")

# Neural Network prediction
X_test_1w_class_scaled = scaler_1w_class.transform(X_test_pipeline1)
test_pred_nn_1w_class = nn_model_1w_class.predict(X_test_1w_class_scaled, batch_size=2048, verbose=0).flatten()
print(f"    Neural Network predictions: mean={test_pred_nn_1w_class.mean():.4f}")

# Ensemble (63% LGBM, 37% NN)
test_pred_1w_class = test_pred_lgb_1w_class * 0.63 + test_pred_nn_1w_class * 0.37
test_pred_1w_class = np.clip(test_pred_1w_class, 0, 1)

print(f"   Ensemble predictions: mean={test_pred_1w_class.mean():.4f}")
print(f"   Weights: 63% LGBM + 37% NN")

# ============================================================================
# 2W Classification - ENSEMBLE (79% LGBM, 21% NN)
# Uses Pipeline 2 (NO encodings)
# ============================================================================
print("\n" + "="*80)
print("2. 2W CLASSIFICATION PREDICTIONS")
print("="*80)

print(f"Using Pipeline 2: {len(feature_cols_pipeline2)} features (NO encodings)")

# LightGBM prediction
test_pred_lgb_2w_class = lgb_model_2w_class.predict(X_test_pipeline2)
print(f"   ✓ LightGBM predictions: mean={test_pred_lgb_2w_class.mean():.4f}")

# Neural Network prediction
X_test_2w_class_scaled = scaler_2w_class.transform(X_test_pipeline2)
test_pred_nn_2w_class = nn_model_2w_class.predict(X_test_2w_class_scaled, batch_size=2048, verbose=0).flatten()
print(f"    Neural Network predictions: mean={test_pred_nn_2w_class.mean():.4f}")

# Ensemble (79% LGBM, 21% NN)
test_pred_2w_class = test_pred_lgb_2w_class * 0.79 + test_pred_nn_2w_class * 0.21
test_pred_2w_class = np.clip(test_pred_2w_class, 0, 1)

print(f"   Ensemble predictions: mean={test_pred_2w_class.mean():.4f}")
print(f"   Weights: 79% LGBM + 21% NN")

# ============================================================================
# 1W Regression - XGBOOST ONLY
# Uses Pipeline 1 (WITH encodings)
# ============================================================================
print("\n" + "="*80)
print("3. 1W REGRESSION PREDICTIONS")
print("="*80)

print(f"Using Pipeline 1: {len(feature_cols_pipeline1)} features (with encodings)")

dtest_1w_reg = xgb.DMatrix(X_test_pipeline1)

# Predict in log space
test_pred_1w_reg_log = xgb_model_1w_reg.predict(dtest_1w_reg)

# Transform back from log
test_pred_1w_reg = np.expm1(test_pred_1w_reg_log)
test_pred_1w_reg = np.clip(test_pred_1w_reg, 0, None)


print(f"   ✓ XGBoost predictions: mean={test_pred_1w_reg.mean():.4f}, median={np.median(test_pred_1w_reg):.4f}")
print(f"   ✓ Zeros: {(test_pred_1w_reg == 0).sum():,} ({(test_pred_1w_reg == 0).mean()*100:.1f}%)")

# ============================================================================
# 2W Regression - XGBOOST ONLY (with 8-week decay)
# Uses Pipeline 1 (WITH encodings)
# ============================================================================
print("\n" + "="*80)
print("4. 2W REGRESSION PREDICTIONS")
print("="*80)

print(f"Using Pipeline 1: {len(feature_cols_pipeline1)} features (with encodings)")

dtest_2w_reg = xgb.DMatrix(X_test_pipeline1)

# Predict in log space
test_pred_2w_reg_log = xgb_model_2w_reg.predict(dtest_2w_reg)

# Transform back from log
test_pred_2w_reg = np.expm1(test_pred_2w_reg_log)
test_pred_2w_reg = np.clip(test_pred_2w_reg, 0, None)


print(f"   ✓ XGBoost predictions: mean={test_pred_2w_reg.mean():.4f}, median={np.median(test_pred_2w_reg):.4f}")
print(f"   ✓ Zeros: {(test_pred_2w_reg == 0).sum():,} ({(test_pred_2w_reg == 0).mean()*100:.1f}%)")

# ============================================================================
# COMPARISON WITH TRAINING DATA
# ============================================================================
print("\n" + "="*80)
print("PREDICTION RATE COMPARISON")
print("="*80)

print(f"\n1W Classification:")
print(f"  Training positive rate:  {y_train_1w_class.mean()*100:.2f}%")
print(f"  Test predictions:")
print(f"    LightGBM:  {test_pred_lgb_1w_class.mean()*100:.2f}%")
print(f"    NN:        {test_pred_nn_1w_class.mean()*100:.2f}%")
print(f"    Ensemble:  {test_pred_1w_class.mean()*100:.2f}%")

if test_pred_1w_class.mean() > y_train_1w_class.mean() * 2:
    print(f"  ⚠️ Predictions significantly higher than training")

print(f"\n2W Classification:")
print(f"  Training positive rate:  {y_train_2w_class.mean()*100:.2f}%")
print(f"  Test predictions:")
print(f"    LightGBM:  {test_pred_lgb_2w_class.mean()*100:.2f}%")
print(f"    NN:        {test_pred_nn_2w_class.mean()*100:.2f}%")
print(f"    Ensemble:  {test_pred_2w_class.mean()*100:.2f}%")

if test_pred_2w_class.mean() > y_train_2w_class.mean() * 2:
    print(f"  ⚠️ Predictions significantly higher than training")

print(f"\n1W Regression:")
print(f"  Training mean:  {y_train_1w_reg.mean():.2f}")
print(f"  Test mean:      {test_pred_1w_reg.mean():.2f}")

print(f"\n2W Regression:")
print(f"  Training mean:  {y_train_2w_reg.mean():.2f}")
print(f"  Test mean:      {test_pred_2w_reg.mean():.2f}")

print("\n" + "="*80)
print(" ALL TEST PREDICTIONS GENERATED")
print("="*80)

# Cleanup
del dtest_1w_reg, dtest_2w_reg
gc.collect()

# ============================================================================
# CELL 9: CREATE FINAL SUBMISSION
# ============================================================================

print("\n" + "="*80)
print("CREATING FINAL SUBMISSION")
print("="*80)

# Extract test IDs
customer_ids = test_df['customer_id'].values
product_ids = test_df['product_unit_variant_id'].values
week_starts = pd.to_datetime(test_df['week_start']).dt.strftime('%Y%m%d').values

# Create submission IDs
submission_ids = [
    f"{cust}_{prod}_{week}" 
    for cust, prod, week in zip(customer_ids, product_ids, week_starts)
]

print(f"\n✓ Generated {len(submission_ids):,} submission IDs")

# Create submission dataframe
submission = pd.DataFrame({
    'ID': submission_ids,
    'Target_purchase_next_1w': test_pred_1w_class,
    'Target_qty_next_1w': test_pred_1w_reg,
    'Target_purchase_next_2w': test_pred_2w_class,
    'Target_qty_next_2w': test_pred_2w_reg
})

# Final validation - clip values
submission['Target_purchase_next_1w'] = submission['Target_purchase_next_1w'].clip(0, 1)
submission['Target_purchase_next_2w'] = submission['Target_purchase_next_2w'].clip(0, 1)
submission['Target_qty_next_1w'] = submission['Target_qty_next_1w'].clip(0, None)
submission['Target_qty_next_2w'] = submission['Target_qty_next_2w'].clip(0, None)

print(f"✓ Submission shape: {submission.shape}")

# Display sample
print("\n" + "="*80)
print("SAMPLE PREDICTIONS")
print("="*80)

print("\nFirst 10 rows:")
print(submission.head(10).to_string(index=False))

print("\nLast 10 rows:")
print(submission.tail(10).to_string(index=False))

# Summary statistics
print("\n" + "="*80)
print("SUBMISSION STATISTICS")
print("="*80)

for col in ['Target_purchase_next_1w', 'Target_qty_next_1w', 
            'Target_purchase_next_2w', 'Target_qty_next_2w']:
    print(f"\n{col}:")
    print(f"  Mean:   {submission[col].mean():.6f}")
    print(f"  Median: {submission[col].median():.6f}")
    print(f"  Min:    {submission[col].min():.6f}")
    print(f"  Max:    {submission[col].max():.6f}")
    print(f"  Std:    {submission[col].std():.6f}")
    print(f"  25th:   {submission[col].quantile(0.25):.6f}")
    print(f"  75th:   {submission[col].quantile(0.75):.6f}")
    
    if 'qty' in col:
        zeros = (submission[col] == 0).sum()
        print(f"  Zeros:  {zeros:,} ({zeros/len(submission)*100:.1f}%)")
        non_zeros = (submission[col] > 0).sum()
        print(f"  >0:     {non_zeros:,} ({non_zeros/len(submission)*100:.1f}%)")
    else:
        low = (submission[col] < 0.1).sum()
        mid = ((submission[col] >= 0.1) & (submission[col] < 0.5)).sum()
        high = (submission[col] >= 0.5).sum()
        print(f"  <0.1:   {low:,} ({low/len(submission)*100:.1f}%)")
        print(f"  0.1-0.5: {mid:,} ({mid/len(submission)*100:.1f}%)")
        print(f"  ≥0.5:   {high:,} ({high/len(submission)*100:.1f}%)")

# Validation checks
print("\n" + "="*80)
print("VALIDATION CHECKS")
print("="*80)

checks_passed = True

# Check 1: No NaN values
print("\n1. Checking for NaN values...")
nan_checks = []
for col in submission.columns[1:]:
    nan_count = submission[col].isna().sum()
    if nan_count > 0:
        print(f"   ⚠ {col}: {nan_count} NaN values")
        checks_passed = False
        nan_checks.append(col)

if not nan_checks:
    print("   ✓ No NaN values found")

# Check 2: No infinite values
print("\n2. Checking for infinite values...")
inf_checks = []
for col in submission.columns[1:]:
    inf_count = np.isinf(submission[col]).sum()
    if inf_count > 0:
        print(f"   ⚠ {col}: {inf_count} infinite values")
        checks_passed = False
        inf_checks.append(col)

if not inf_checks:
    print("   ✓ No infinite values found")

# Check 3: Classification predictions in [0, 1]
print("\n3. Checking classification ranges...")
class_range_ok = True

if submission['Target_purchase_next_1w'].min() < 0 or submission['Target_purchase_next_1w'].max() > 1:
    print(f"   ⚠ Target_purchase_next_1w out of range: [{submission['Target_purchase_next_1w'].min():.4f}, {submission['Target_purchase_next_1w'].max():.4f}]")
    checks_passed = False
    class_range_ok = False

if submission['Target_purchase_next_2w'].min() < 0 or submission['Target_purchase_next_2w'].max() > 1:
    print(f"   ⚠ Target_purchase_next_2w out of range: [{submission['Target_purchase_next_2w'].min():.4f}, {submission['Target_purchase_next_2w'].max():.4f}]")
    checks_passed = False
    class_range_ok = False

if class_range_ok:
    print("   ✓ Classification predictions in [0, 1]")

# Check 4: Regression predictions non-negative
print("\n4. Checking regression non-negativity...")
reg_nonneg_ok = True

if submission['Target_qty_next_1w'].min() < 0:
    neg_count = (submission['Target_qty_next_1w'] < 0).sum()
    print(f"   ⚠ Target_qty_next_1w has {neg_count} negative values (min: {submission['Target_qty_next_1w'].min():.4f})")
    checks_passed = False
    reg_nonneg_ok = False

if submission['Target_qty_next_2w'].min() < 0:
    neg_count = (submission['Target_qty_next_2w'] < 0).sum()
    print(f"   ⚠ Target_qty_next_2w has {neg_count} negative values (min: {submission['Target_qty_next_2w'].min():.4f})")
    checks_passed = False
    reg_nonneg_ok = False

if reg_nonneg_ok:
    print("   ✓ Regression predictions non-negative")

# Check 5: ID uniqueness
print("\n5. Checking ID uniqueness...")
unique_ids = submission['ID'].nunique()
total_ids = len(submission)

if unique_ids == total_ids:
    print(f"   ✓ All IDs unique ({unique_ids:,})")
else:
    duplicate_count = total_ids - unique_ids
    print(f"   ⚠ {duplicate_count} duplicate IDs found")
    checks_passed = False

# Check 6: Correct number of rows
print("\n6. Checking row count...")
expected_rows = len(test_df)
actual_rows = len(submission)

if expected_rows == actual_rows:
    print(f"    Correct row count ({actual_rows:,})")
else:
    print(f"   Row count mismatch: expected {expected_rows:,}, got {actual_rows:,}")
    checks_passed = False

# Save submission
filename = 'submission_final_ensemble_lgbm_1w_2w_nn_latest_no_clipping_final.csv'
submission.to_csv(filename, index=False)

print("\n" + "="*80)
print("SUBMISSION SAVED!")
print("="*80)

print(f"\n✓ Filename: {filename}")
print(f"✓ Rows: {len(submission):,}")
print(f"✓ Columns: {list(submission.columns)}")
print(f"✓ File size: {len(submission) * len(submission.columns)} cells")

if checks_passed:
    print(f"\n ALL VALIDATION CHECKS PASSED!")
    print(f"   Submission is ready for upload")
else:
    print(f"\nSOME VALIDATION CHECKS FAILED")
    print(f"   Please review the warnings above")

# Model summary
print("\n" + "="*80)
print("FINAL MODEL SUMMARY")
print("="*80)

print("\n Models Used:")
print("  1W Classification:")
print(f"    • LightGBM (63%) - {len(feature_cols_pipeline1)} features with encodings")
print(f"    • Neural Network (37%) - Same features")
print(f"    • Final: Ensemble of both")

print("\n  2W Classification:")
print(f"    • LightGBM (79%) - {len(feature_cols_pipeline2)} features, NO encodings, 8w decay")
print(f"    • Neural Network (21%) - Same features")
print(f"    • Final: Ensemble of both")

print("\n  1W Regression:")
print(f"    • XGBoost only - {len(feature_cols_pipeline1)} features with encodings")

print("\n  2W Regression:")
print(f"    • XGBoost only - {len(feature_cols_pipeline1)} features with encodings, 8w decay")

print("\n Feature Breakdown:")
print(f"  Pipeline 1 (1W Class, 1W Reg, 2W Reg): {len(feature_cols_pipeline1)} features")
print(f"    = Baseline + 8 category target encodings + 8 new recency/momentum")

print(f"\n  Pipeline 2 (2W Class only): {len(feature_cols_pipeline2)} features")
print(f"    = Baseline + 8 new recency/momentum (NO encodings)")

print("\n Training Data:")
print(f"  1W Classification: {len(y_train_1w_class):,} samples, {y_train_1w_class.mean()*100:.2f}% positive")
print(f"  2W Classification: {len(y_train_2w_class):,} samples, {y_train_2w_class.mean()*100:.2f}% positive")
print(f"  1W Regression:     Mean qty = {y_train_1w_reg.mean():.2f}")
print(f"  2W Regression:     Mean qty = {y_train_2w_reg.mean():.2f}")

print("\n Test Predictions:")
print(f"  1W Classification: {test_pred_1w_class.mean()*100:.2f}% positive (vs {y_train_1w_class.mean()*100:.2f}% train)")
print(f"  2W Classification: {test_pred_2w_class.mean()*100:.2f}% positive (vs {y_train_2w_class.mean()*100:.2f}% train)")
print(f"  1W Regression:     Mean = {test_pred_1w_reg.mean():.2f} (vs {y_train_1w_reg.mean():.2f} train)")
print(f"  2W Regression:     Mean = {test_pred_2w_reg.mean():.2f} (vs {y_train_2w_reg.mean():.2f} train)")

print("\n READY TO SUBMIT!")
print("="*80)

# Final cleanup
del lgb_model_1w_class, lgb_model_2w_class
del xgb_model_1w_reg, xgb_model_2w_reg
del nn_model_1w_class, nn_model_2w_class
gc.collect()

print("\n Memory cleaned")
print("✓ All done!")

