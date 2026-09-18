# Core libraries
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-v0_8-darkgrid')
%matplotlib inline

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', lambda x: '%.3f' % x)



# Load the datasets
train = pd.read_csv('Train.csv')
test = pd.read_csv('Test.csv')
sku_data = pd.read_csv('sku_data.csv')

print("Data loaded successfully!")
print(f"\nTrain shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"SKU data shape: {sku_data.shape}")

print("="*80)
print("TRAIN DATA - BASIC INFORMATION")
print("="*80)

print("\n1. First few rows:")
print(train.head())

print("\n2. Data types and non-null counts:")
print(train.info())

print("\n3. Statistical summary:")
print(train.describe())

print("\n4. Column names:")
print(train.columns.tolist())

print("="*80)
print("TEST DATA - BASIC INFORMATION")
print("="*80)

print("\n1. First few rows:")
print(test.head())

print("\n2. Data types and non-null counts:")
print(test.info())

print("\n3. Statistical summary:")
print(test.describe())

print("\n4. Column names:")
print(test.columns.tolist())

print("="*80)
print("SKU DATA - BASIC INFORMATION")
print("="*80)

print("\n1. First few rows:")
print(sku_data.head())

print("\n2. Data types and non-null counts:")
print(sku_data.info())

print("\n3. Statistical summary:")
print(sku_data.describe())

print("\n4. Column names:")
print(sku_data.columns.tolist())

print("="*80)
print("MISSING VALUES ANALYSIS")
print("="*80)

def analyze_missing(df, name):
    print(f"\n{name}:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing_Count': missing,
        'Missing_Percentage': missing_pct
    })
    missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
    
    if len(missing_df) > 0:
        print(missing_df)
    else:
        print("No missing values!")
    
    return missing_df

train_missing = analyze_missing(train, "TRAIN")
test_missing = analyze_missing(test, "TEST")
sku_missing = analyze_missing(sku_data, "SKU DATA")

  print("="*80)
print("FORMATTING DATE COLUMNS")
print("="*80)

# Convert date columns to datetime
print("\n1. Converting 'week_start' to datetime...")
train['week_start'] = pd.to_datetime(train['week_start'])
test['week_start'] = pd.to_datetime(test['week_start'])

print(f"   Train week_start range: {train['week_start'].min()} to {train['week_start'].max()}")
print(f"   Test week_start range: {test['week_start'].min()} to {test['week_start'].max()}")

# Convert customer_created_at to datetime
print("\n2. Converting 'customer_created_at' to datetime...")
train['customer_created_at'] = pd.to_datetime(train['customer_created_at'])
test['customer_created_at'] = pd.to_datetime(test['customer_created_at'])

print(f"   Train customer_created_at range: {train['customer_created_at'].min()} to {train['customer_created_at'].max()}")
print(f"   Test customer_created_at range: {test['customer_created_at'].min()} to {test['customer_created_at'].max()}")

print("\n✓ Date formatting complete!")
print(f"\nTrain dtypes after conversion:")
print(train[['week_start', 'customer_created_at']].dtypes)

print("="*80)
print("IDENTIFYING COLUMNS TO REMOVE")
print("="*80)

# Target columns (what we need to predict)
target_cols = ['Target_qty_next_1w', 'Target_purchase_next_1w', 
               'Target_qty_next_2w', 'Target_purchase_next_2w']

print(f"\nTarget columns (keep in train):")
for col in target_cols:
    print(f"  - {col}")

# Common columns between train and test (excluding ID)
common_cols = list(set(train.columns) & set(test.columns))
common_cols.remove('ID')  # ID is just identifier, not a feature
print(f"\nCommon columns (train & test): {len(common_cols)}")
for col in sorted(common_cols):
    print(f"  - {col}")

# Columns in train but NOT in test (excluding targets)
train_only = set(train.columns) - set(test.columns) - set(target_cols) - {'ID'}

print(f"\nColumns in TRAIN ONLY (predictors to remove): {len(train_only)}")
for col in sorted(train_only):
    print(f"  - {col}")

print(f"\n  These columns will be REMOVED from train because they don't exist in test!")

print("="*80)
print("REMOVING EXTRA PREDICTOR COLUMNS")
print("="*80)

print(f"\nBefore removal:")
print(f"  Train shape: {train.shape}")

# Get columns to keep
columns_to_keep = common_cols + target_cols + ['ID']
print(f"\nColumns to keep in train: {len(columns_to_keep)}")

# Create cleaned train dataset
train_cleaned = train[columns_to_keep].copy()

print(f"\nAfter removal:")
print(f"  Train shape: {train_cleaned.shape}")
print(f"  Removed: {train.shape[1] - train_cleaned.shape[1]} columns")

# Verify target columns are still there
print(f"\n Target columns still present:")
for col in target_cols:
    if col in train_cleaned.columns:
        print(f"  - {col}: ")
    else:
        print(f"  - {col}: ✗ MISSING!")

# Update train to cleaned version
train = train_cleaned.copy()

print(f"\n Cleaning complete!")
print(f"\nFinal shapes:")
print(f"  Train: {train.shape}")
print(f"  Test: {test.shape}")

print("="*80)
print("FINAL COLUMN ALIGNMENT CHECK")
print("="*80)

# Get feature columns (excluding ID and targets)
train_features = [col for col in train.columns if col not in target_cols and col != 'ID']
test_features = [col for col in test.columns if col != 'ID']

print(f"\nTrain features: {len(train_features)}")
print(f"Test features: {len(test_features)}")

# Check if they match
if set(train_features) == set(test_features):
    print("\nTrain and test have the same features!")
    print(f"\nShared features ({len(train_features)}):")
    for col in sorted(train_features):
        print(f"  - {col}")
else:
    print("\n WARNING: Mismatch detected!")
    
    missing_in_test = set(train_features) - set(test_features)
    if missing_in_test:
        print(f"\n  Missing in test: {missing_in_test}")
    
    missing_in_train = set(test_features) - set(train_features)
    if missing_in_train:
        print(f"  Missing in train: {missing_in_train}")

# Show final column order
print(f"\n\nFinal Train columns:")
print(train.columns.tolist())

print(f"\n\nFinal Test columns:")
print(test.columns.tolist())

print("="*80)
print("TARGET VARIABLES ANALYSIS")
print("="*80)

target_cols = ['Target_purchase_next_1w', 'Target_qty_next_1w', 
               'Target_purchase_next_2w', 'Target_qty_next_2w']

print("\n1. BASIC STATISTICS:")
print("="*80)
for col in target_cols:
    print(f"\n{col}:")
    print(f"  Data type: {train[col].dtype}")
    print(f"  Count: {train[col].count():,}")
    print(f"  Mean: {train[col].mean():.4f}")
    print(f"  Median: {train[col].median():.4f}")
    print(f"  Std: {train[col].std():.4f}")
    print(f"  Min: {train[col].min():.4f}")
    print(f"  Max: {train[col].max():.4f}")
    print(f"  Sum: {train[col].sum():,.0f}")

print("="*80)
print("BINARY TARGETS - CLASS IMBALANCE ANALYSIS")
print("="*80)

binary_targets = ['Target_purchase_next_1w', 'Target_purchase_next_2w']

for col in binary_targets:
    print(f"\n{col}:")
    print("-" * 60)
    
    # Value counts
    value_counts = train[col].value_counts().sort_index()
    print(f"\nValue counts:")
    for val, count in value_counts.items():
        pct = (count / len(train)) * 100
        print(f"  {int(val)}: {count:,} ({pct:.2f}%)")
    
    # Class imbalance ratio
    if 1 in value_counts.index and 0 in value_counts.index:
        imbalance_ratio = value_counts[0] / value_counts[1]
        print(f"\n  Class imbalance ratio (0:1): {imbalance_ratio:.1f}:1")
        print(f"   For every 1 purchase, there are {imbalance_ratio:.1f} non-purchases")

print("="*80)
print("QUANTITY TARGETS - DISTRIBUTION ANALYSIS")
print("="*80)

qty_targets = ['Target_qty_next_1w', 'Target_qty_next_2w']

for col in qty_targets:
    print(f"\n{col}:")
    print("-" * 60)
    
    # Basic stats
    non_zero = train[train[col] > 0][col]
    zero_count = (train[col] == 0).sum()
    
    print(f"\nZero vs Non-zero:")
    print(f"  Zeros: {zero_count:,} ({zero_count/len(train)*100:.2f}%)")
    print(f"  Non-zeros: {len(non_zero):,} ({len(non_zero)/len(train)*100:.2f}%)")
    
    if len(non_zero) > 0:
        print(f"\nNon-zero quantities statistics:")
        print(f"  Mean: {non_zero.mean():.2f}")
        print(f"  Median: {non_zero.median():.2f}")
        print(f"  Std: {non_zero.std():.2f}")
        print(f"  Min: {non_zero.min():.2f}")
        print(f"  25%: {non_zero.quantile(0.25):.2f}")
        print(f"  75%: {non_zero.quantile(0.75):.2f}")
        print(f"  95%: {non_zero.quantile(0.95):.2f}")
        print(f"  99%: {non_zero.quantile(0.99):.2f}")
        print(f"  Max: {non_zero.max():.2f}")

print("="*80)
print("RELATIONSHIP: BINARY vs QUANTITY TARGETS")
print("="*80)

# Check 1-week relationship
print("\n1-WEEK TARGETS:")
print("-" * 60)

purchased_1w = train[train['Target_purchase_next_1w'] == 1]
not_purchased_1w = train[train['Target_purchase_next_1w'] == 0]

print(f"\nWhen Target_purchase_next_1w = 1 (purchased):")
print(f"  Count: {len(purchased_1w):,}")
print(f"  Target_qty_next_1w mean: {purchased_1w['Target_qty_next_1w'].mean():.2f}")
print(f"  Target_qty_next_1w median: {purchased_1w['Target_qty_next_1w'].median():.2f}")
print(f"  Target_qty_next_1w max: {purchased_1w['Target_qty_next_1w'].max():.2f}")

# Check if purchase=1 but qty=0 (data quality check)
anomaly_1w = purchased_1w[purchased_1w['Target_qty_next_1w'] == 0]
print(f"\n    Anomaly check: purchase=1 but qty=0: {len(anomaly_1w):,} cases")

print(f"\nWhen Target_purchase_next_1w = 0 (not purchased):")
print(f"  Count: {len(not_purchased_1w):,}")
print(f"  Target_qty_next_1w mean: {not_purchased_1w['Target_qty_next_1w'].mean():.2f}")
print(f"  Target_qty_next_1w max: {not_purchased_1w['Target_qty_next_1w'].max():.2f}")

# Check if purchase=0 but qty>0 (data quality check)
anomaly_1w_reverse = not_purchased_1w[not_purchased_1w['Target_qty_next_1w'] > 0]
print(f"\n    Anomaly check: purchase=0 but qty>0: {len(anomaly_1w_reverse):,} cases")

# Check 2-week relationship
print("\n\n2-WEEK TARGETS:")
print("-" * 60)

purchased_2w = train[train['Target_purchase_next_2w'] == 1]
not_purchased_2w = train[train['Target_purchase_next_2w'] == 0]

print(f"\nWhen Target_purchase_next_2w = 1 (purchased):")
print(f"  Count: {len(purchased_2w):,}")
print(f"  Target_qty_next_2w mean: {purchased_2w['Target_qty_next_2w'].mean():.2f}")
print(f"  Target_qty_next_2w median: {purchased_2w['Target_qty_next_2w'].median():.2f}")
print(f"  Target_qty_next_2w max: {purchased_2w['Target_qty_next_2w'].max():.2f}")

anomaly_2w = purchased_2w[purchased_2w['Target_qty_next_2w'] == 0]
print(f"\n   Anomaly check: purchase=1 but qty=0: {len(anomaly_2w):,} cases")

print(f"\nWhen Target_purchase_next_2w = 0 (not purchased):")
print(f"  Count: {len(not_purchased_2w):,}")
print(f"  Target_qty_next_2w mean: {not_purchased_2w['Target_qty_next_2w'].mean():.2f}")
print(f"  Target_qty_next_2w max: {not_purchased_2w['Target_qty_next_2w'].max():.2f}")

anomaly_2w_reverse = not_purchased_2w[not_purchased_2w['Target_qty_next_2w'] > 0]
print(f"\n   Anomaly check: purchase=0 but qty>0: {len(anomaly_2w_reverse):,} cases")

print("="*80)
print("TARGET VARIABLES VISUALIZATION")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1-week purchase distribution
ax1 = axes[0, 0]
train['Target_purchase_next_1w'].value_counts().plot(kind='bar', ax=ax1, color=['#e74c3c', '#2ecc71'])
ax1.set_title('Target_purchase_next_1w Distribution', fontsize=12, fontweight='bold')
ax1.set_xlabel('Purchase (0=No, 1=Yes)')
ax1.set_ylabel('Count')
ax1.set_xticklabels(['No Purchase (0)', 'Purchase (1)'], rotation=0)
for p in ax1.patches:
    ax1.annotate(f'{int(p.get_height()):,}', 
                 (p.get_x() + p.get_width()/2., p.get_height()),
                 ha='center', va='bottom')

# 2-week purchase distribution
ax2 = axes[0, 1]
train['Target_purchase_next_2w'].value_counts().plot(kind='bar', ax=ax2, color=['#e74c3c', '#2ecc71'])
ax2.set_title('Target_purchase_next_2w Distribution', fontsize=12, fontweight='bold')
ax2.set_xlabel('Purchase (0=No, 1=Yes)')
ax2.set_ylabel('Count')
ax2.set_xticklabels(['No Purchase (0)', 'Purchase (1)'], rotation=0)
for p in ax2.patches:
    ax2.annotate(f'{int(p.get_height()):,}', 
                 (p.get_x() + p.get_width()/2., p.get_height()),
                 ha='center', va='bottom')

# 1-week quantity distribution (non-zero only)
ax3 = axes[1, 0]
qty_1w_nonzero = train[train['Target_qty_next_1w'] > 0]['Target_qty_next_1w']
ax3.hist(qty_1w_nonzero, bins=50, color='#3498db', edgecolor='black', alpha=0.7)
ax3.set_title('Target_qty_next_1w Distribution (Non-zero only)', fontsize=12, fontweight='bold')
ax3.set_xlabel('Quantity')
ax3.set_ylabel('Frequency')
ax3.axvline(qty_1w_nonzero.median(), color='red', linestyle='--', linewidth=2, label=f'Median: {qty_1w_nonzero.median():.1f}')
ax3.legend()

# 2-week quantity distribution (non-zero only)
ax4 = axes[1, 1]
qty_2w_nonzero = train[train['Target_qty_next_2w'] > 0]['Target_qty_next_2w']
ax4.hist(qty_2w_nonzero, bins=50, color='#9b59b6', edgecolor='black', alpha=0.7)
ax4.set_title('Target_qty_next_2w Distribution (Non-zero only)', fontsize=12, fontweight='bold')
ax4.set_xlabel('Quantity')
ax4.set_ylabel('Frequency')
ax4.axvline(qty_2w_nonzero.median(), color='red', linestyle='--', linewidth=2, label=f'Median: {qty_2w_nonzero.median():.1f}')
ax4.legend()

plt.tight_layout()
plt.show()

print("\n Visualization complete!")

print("="*80)
print("IDENTIFYING NUMERICAL COLUMNS")
print("="*80)

# Get numerical columns (excluding targets and ID)
target_cols = ['Target_qty_next_1w', 'Target_purchase_next_1w', 
               'Target_qty_next_2w', 'Target_purchase_next_2w']

numerical_cols = train.select_dtypes(include=[np.number]).columns.tolist()

# Remove targets and ID-like columns
numerical_cols = [col for col in numerical_cols if col not in target_cols]

print(f"\nNumerical columns found: {len(numerical_cols)}")
for col in numerical_cols:
    print(f"  - {col}")

print(f"\n These will be analyzed")

print("="*80)
print("NUMERICAL COLUMNS - DESCRIPTIVE STATISTICS")
print("="*80)

for col in numerical_cols:
    print(f"\n{'='*60}")
    print(f"{col}")
    print('='*60)
    
    print(f"\nData type: {train[col].dtype}")
    print(f"Count: {train[col].count():,}")
    print(f"Missing: {train[col].isnull().sum():,}")
    print(f"Unique values: {train[col].nunique():,}")
    
    print(f"\nStatistics:")
    print(f"  Mean:   {train[col].mean():.4f}")
    print(f"  Median: {train[col].median():.4f}")
    print(f"  Std:    {train[col].std():.4f}")
    print(f"  Min:    {train[col].min():.4f}")
    print(f"  25%:    {train[col].quantile(0.25):.4f}")
    print(f"  50%:    {train[col].quantile(0.50):.4f}")
    print(f"  75%:    {train[col].quantile(0.75):.4f}")
    print(f"  95%:    {train[col].quantile(0.95):.4f}")
    print(f"  99%:    {train[col].quantile(0.99):.4f}")
    print(f"  Max:    {train[col].max():.4f}")
    
    # Check for zeros
    zero_count = (train[col] == 0).sum()
    zero_pct = (zero_count / len(train)) * 100
    print(f"\nZeros: {zero_count:,} ({zero_pct:.2f}%)")

print("="*80)
print("NUMERICAL COLUMNS - DISTRIBUTION PATTERNS")
print("="*80)

for col in numerical_cols:
    print(f"\n{col}:")
    print("-" * 60)
    
    # Value distribution
    print("\nTop 10 most frequent values:")
    top_values = train[col].value_counts().head(10)
    for val, count in top_values.items():
        pct = (count / len(train)) * 100
        print(f"  {val}: {count:,} ({pct:.2f}%)")
    
    # Skewness and kurtosis
    skewness = train[col].skew()
    kurtosis = train[col].kurtosis()
    
    print(f"\nDistribution shape:")
    print(f"  Skewness: {skewness:.4f}", end="")
    if abs(skewness) < 0.5:
        print("  Fairly symmetric")
    elif skewness > 0:
        print("  Right-skewed (long tail on right)")
    else:
        print("  Left-skewed (long tail on left)")
    
    print(f"  Kurtosis: {kurtosis:.4f}", end="")
    if abs(kurtosis) < 3:
        print(" → Normal-like tails")
    elif kurtosis > 3:
        print("  Heavy tails (more outliers)")
    else:
        print("  Light tails (fewer outliers)")

print("="*80)
print("NUMERICAL FEATURES vs TARGET VARIABLES")
print("="*80)

# Check correlation with targets
target_cols_analysis = ['Target_purchase_next_1w', 'Target_purchase_next_2w']

for target in target_cols_analysis:
    print(f"\n{'='*60}")
    print(f"Correlation with {target}")
    print('='*60)
    
    for col in numerical_cols:
        corr = train[col].corr(train[target])
        print(f"  {col}: {corr:.4f}")
    
    # Find most correlated feature
    correlations = {col: train[col].corr(train[target]) for col in numerical_cols}
    most_corr = max(correlations, key=lambda k: abs(correlations[k]))
    print(f"\n  → Most correlated: {most_corr} ({correlations[most_corr]:.4f})")

print("="*80)
print("IDENTIFYING CATEGORICAL COLUMNS")
print("="*80)

# Get categorical columns (excluding ID and dates)
categorical_cols = train.select_dtypes(include=['object', 'category']).columns.tolist()

# Remove ID and date columns
categorical_cols = [col for col in categorical_cols 
                   if col != 'ID' and 'date' not in col.lower()]

print(f"\nCategorical columns found: {len(categorical_cols)}")
for col in categorical_cols:
    print(f"  - {col}")

print(f"\n These will be analyzed")

print("="*80)
print("CATEGORICAL COLUMNS - BASIC OVERVIEW")
print("="*80)

for col in categorical_cols:
    print(f"\n{'='*60}")
    print(f"{col}")
    print('='*60)
    
    print(f"Data type: {train[col].dtype}")
    print(f"Total values: {len(train):,}")
    print(f"Unique values: {train[col].nunique():,}")
    print(f"Missing values: {train[col].isnull().sum():,}")
    
    print(f"\nValue distribution:")
    value_counts = train[col].value_counts()
    for val, count in value_counts.items():
        pct = (count / len(train)) * 100
        print(f"  {val}: {count:,} ({pct:.2f}%)")

print("="*80)
print("TEMPORAL COVERAGE - DATE RANGES & GAPS")
print("="*80)

# Train temporal coverage
print("\nTRAIN DATA:")
print("-" * 60)
print(f"Date range: {train['week_start'].min()} to {train['week_start'].max()}")
print(f"Number of weeks: {train['week_start'].nunique()}")
print(f"Total days: {(train['week_start'].max() - train['week_start'].min()).days}")

# Test temporal coverage
print("\n\nTEST DATA:")
print("-" * 60)
print(f"Date range: {test['week_start'].min()} to {test['week_start'].max()}")
print(f"Number of weeks: {test['week_start'].nunique()}")
print(f"Total days: {(test['week_start'].max() - test['week_start'].min()).days}")

# Gap between train and test
gap_days = (test['week_start'].min() - train['week_start'].max()).days
print("\n\nGAP BETWEEN TRAIN AND TEST:")
print("-" * 60)
print(f"Days between last train week and first test week: {gap_days} days")
print(f"Weeks: {gap_days // 7} weeks")

# List all weeks
print("\n\nALL WEEKS IN TRAIN:")
all_weeks_train = sorted(train['week_start'].unique())
print(f"Total: {len(all_weeks_train)} weeks")
for week in all_weeks_train:
    count = len(train[train['week_start'] == week])
    print(f"  {week.date()}: {count:,} rows")

print("\n\nALL WEEKS IN TEST:")
all_weeks_test = sorted(test['week_start'].unique())
print(f"Total: {len(all_weeks_test)} weeks")
for week in all_weeks_test:
    count = len(test[test['week_start'] == week])
    print(f"  {week.date()}: {count:,} rows")

print("="*80)
print("CUSTOMER ACCOUNT AGE ANALYSIS")
print("="*80)

# Calculate account age at each week_start
train['account_age_days'] = (train['week_start'] - train['customer_created_at']).dt.days
test['account_age_days'] = (test['week_start'] - test['customer_created_at']).dt.days

print("\nTRAIN - Account Age Statistics:")
print(f"  Mean: {train['account_age_days'].mean():.1f} days ({train['account_age_days'].mean()/365:.2f} years)")
print(f"  Median: {train['account_age_days'].median():.1f} days ({train['account_age_days'].median()/365:.2f} years)")
print(f"  Min: {train['account_age_days'].min():.1f} days")
print(f"  Max: {train['account_age_days'].max():.1f} days ({train['account_age_days'].max()/365:.2f} years)")

print("\nTEST - Account Age Statistics:")
print(f"  Mean: {test['account_age_days'].mean():.1f} days ({test['account_age_days'].mean()/365:.2f} years)")
print(f"  Median: {test['account_age_days'].median():.1f} days ({test['account_age_days'].median()/365:.2f} years)")
print(f"  Min: {test['account_age_days'].min():.1f} days")
print(f"  Max: {test['account_age_days'].max():.1f} days ({test['account_age_days'].max()/365:.2f} years)")

print("="*80)
print("PHASE 1: SETUP - PREPARING DATA FOR FEATURE ENGINEERING")
print("="*80)

# Sort data by customer, product, and week (CRITICAL for time-based features)
print("\n1. Sorting data by time...")
train = train.sort_values(['customer_id', 'product_unit_variant_id', 'week_start']).reset_index(drop=True)
test = test.sort_values(['customer_id', 'product_unit_variant_id', 'week_start']).reset_index(drop=True)

print(f"   Train shape: {train.shape}")
print(f"   Test shape: {test.shape}")

# Combine train and test for feature engineering (we'll split later)
print("\n2. Combining train and test...")
train['is_train'] = 1
test['is_train'] = 0

# Add temporary target columns to test (filled with NaN) so we can combine
for col in ['Target_qty_next_1w', 'Target_purchase_next_1w', 'Target_qty_next_2w', 'Target_purchase_next_2w']:
    if col not in test.columns:
        test[col] = np.nan

# Combine
df_combined = pd.concat([train, test], axis=0, ignore_index=True)
df_combined = df_combined.sort_values(['customer_id', 'product_unit_variant_id', 'week_start']).reset_index(drop=True)

print(f"   Combined shape: {df_combined.shape}")
print(f"   Date range: {df_combined['week_start'].min()} to {df_combined['week_start'].max()}")

# Create a copy for feature engineering
df = df_combined.copy()

print("\n Setup complete!")
print(f"\nCombined dataset ready for feature engineering:")
print(f"  Total rows: {len(df):,}")
print(f"  Train rows: {(df['is_train'] == 1).sum():,}")
print(f"  Test rows: {(df['is_train'] == 0).sum():,}")

print("="*80)
print("CREATING HISTORICAL PURCHASE FEATURES FROM TARGETS")
print("="*80)

# For each customer-product pair, shift the targets to create historical features
print("\nShifting targets to create purchase history...")

# Group by customer and product
groupby_cols = ['customer_id', 'product_unit_variant_id']

# Shift purchase indicators (these become "did they buy last week?" features)
df['purchased_last_1w'] = df.groupby(groupby_cols)['Target_purchase_next_1w'].shift(1)
df['purchased_last_2w'] = df.groupby(groupby_cols)['Target_purchase_next_1w'].shift(2)
df['purchased_last_3w'] = df.groupby(groupby_cols)['Target_purchase_next_1w'].shift(3)
df['purchased_last_4w'] = df.groupby(groupby_cols)['Target_purchase_next_1w'].shift(4)

# Shift quantity purchased
df['qty_purchased_last_1w'] = df.groupby(groupby_cols)['Target_qty_next_1w'].shift(1)
df['qty_purchased_last_2w'] = df.groupby(groupby_cols)['Target_qty_next_1w'].shift(2)
df['qty_purchased_last_3w'] = df.groupby(groupby_cols)['Target_qty_next_1w'].shift(3)
df['qty_purchased_last_4w'] = df.groupby(groupby_cols)['Target_qty_next_1w'].shift(4)

print("\n✓ Historical features created!")

# Check for NaN values (expected in first few weeks)
print(f"\nNaN counts (expected in early weeks):")
print(f"  purchased_last_1w: {df['purchased_last_1w'].isnull().sum():,}")
print(f"  purchased_last_4w: {df['purchased_last_4w'].isnull().sum():,}")

# Preview
print("\nSample data (showing history):")
sample = df[['customer_id', 'product_unit_variant_id', 'week_start', 
             'purchased_last_1w', 'purchased_last_2w', 'purchased_last_3w', 'purchased_last_4w',
             'Target_purchase_next_1w']].head(10)
print(sample.to_string(index=False))

print("="*80)
print("CREATING ROLLING PURCHASE COUNT FEATURES")
print("="*80)

print("\nCalculating rolling purchase counts...")

# Rolling sum of purchases (4, 8, 12 weeks)
# Note: We need to shift by 1 to avoid data leakage (don't include current week)
for window in [4, 8, 12]:
    print(f"  Processing {window}-week window...")
    
    # Count of purchases in last N weeks
    df[f'purchase_count_last_{window}w'] = (
        df.groupby(groupby_cols)['Target_purchase_next_1w']
        .shift(1)
        .rolling(window=window, min_periods=1)
        .sum()
        .values  # Use .values instead of reset_index
    )
    
    # Total quantity purchased in last N weeks
    df[f'total_qty_last_{window}w'] = (
        df.groupby(groupby_cols)['Target_qty_next_1w']
        .shift(1)
        .rolling(window=window, min_periods=1)
        .sum()
        .values  # Use .values instead of reset_index
    )

print("\n Rolling purchase counts created!")

# Check sample
print("\nSample rolling features:")
sample_cols = ['customer_id', 'product_unit_variant_id', 'week_start',
               'purchase_count_last_4w', 'purchase_count_last_8w', 'purchase_count_last_12w']
print(df[sample_cols].head(20).to_string(index=False))

# Statistics
print(f"\nFeature statistics:")
for window in [4, 8, 12]:
    col = f'purchase_count_last_{window}w'
    print(f"\n{col}:")
    print(f"  Mean: {df[col].mean():.2f}")
    print(f"  Max: {df[col].max():.0f}")
    print(f"  NaN: {df[col].isnull().sum():,}")

print("="*80)
print("CREATING PURCHASE RATE FEATURES")
print("="*80)

print("\nCalculating purchase rates (% of weeks with purchase)...")

# Purchase rate = (number of purchases) / (window size)
for window in [4, 8, 12]:
    print(f"  Processing {window}-week window...")
    
    purchase_count_col = f'purchase_count_last_{window}w'
    rate_col = f'purchase_rate_last_{window}w'
    
    # Purchase rate = count / window (will be between 0 and 1)
    df[rate_col] = df[purchase_count_col] / window

print("\n Purchase rate features created!")

# Check sample
print("\nSample purchase rate features:")
sample_cols = ['customer_id', 'product_unit_variant_id', 'week_start',
               'purchase_rate_last_4w', 'purchase_rate_last_8w', 'purchase_rate_last_12w']
print(df[sample_cols].head(20).to_string(index=False))

# Statistics
print(f"\nFeature statistics:")
for window in [4, 8, 12]:
    col = f'purchase_rate_last_{window}w'
    print(f"\n{col}:")
    print(f"  Mean: {df[col].mean():.4f}")
    print(f"  Min: {df[col].min():.4f}")
    print(f"  Max: {df[col].max():.4f}")
    print(f"  NaN: {df[col].isnull().sum():,}")

 print("="*80)
print("CREATING AVERAGE QUANTITY FEATURES")
print("="*80)

print("\nCalculating average quantity per purchase...")

# For each window, calculate average quantity WHEN PURCHASED (not including zeros)
for window in [4, 8, 12]:
    print(f"  Processing {window}-week window...")
    
    qty_col = f'total_qty_last_{window}w'
    count_col = f'purchase_count_last_{window}w'
    avg_col = f'avg_qty_per_purchase_last_{window}w'
    
    # Average = total quantity / number of purchases (avoid division by zero)
    df[avg_col] = df[qty_col] / df[count_col].replace(0, np.nan)
    
    # Fill NaN with 0 (means no purchases in window)
    df[avg_col] = df[avg_col].fillna(0)

print("\n Average quantity features created!")

# Statistics
print(f"\nFeature statistics:")
for window in [4, 8, 12]:
    col = f'avg_qty_per_purchase_last_{window}w'
    print(f"\n{col}:")
    print(f"  Mean: {df[col].mean():.2f}")
    print(f"  Median: {df[col].median():.2f}")
    print(f"  Max: {df[col].max():.2f}")
    print(f"  NaN: {df[col].isnull().sum():,}")

# Sample
print("\nSample average quantity features:")
sample_cols = ['customer_id', 'product_unit_variant_id', 'week_start',
               'purchase_count_last_8w', 'total_qty_last_8w', 'avg_qty_per_purchase_last_8w']
print(df[sample_cols].head(20).to_string(index=False))

print("="*80)
print("PHASE 2: AGGREGATION FEATURES - CUSTOMER ACTIVITY")
print("="*80)

print("\nCalculating customer-level activity across ALL products...")

# For each customer at each week, aggregate their total activity
for window in [4, 8, 12]:
    print(f"  Processing {window}-week window...")
    
    # Total purchases by customer across ALL products (last N weeks)
    df[f'customer_total_purchases_{window}w'] = (
        df.groupby(['customer_id', 'week_start'])['Target_purchase_next_1w']
        .transform('sum')  # Sum across all products for this customer-week
    )
    
    # Shift by window to get PAST purchases (avoid leakage)
    df[f'customer_total_purchases_{window}w'] = (
        df.groupby('customer_id')[f'customer_total_purchases_{window}w']
        .shift(1)
        .rolling(window=window, min_periods=1)
        .sum()
        .values
    )
    
    # Total unique products purchased by customer (last N weeks)
    # We'll use a different approach - count distinct products that were purchased
    temp = df[df['Target_purchase_next_1w'] == 1].groupby(['customer_id', 'week_start'])['product_id'].nunique()
    df[f'customer_unique_products_{window}w'] = df.set_index(['customer_id', 'week_start']).index.map(temp).values
    
    # Shift and rolling
    df[f'customer_unique_products_{window}w'] = (
        df.groupby('customer_id')[f'customer_unique_products_{window}w']
        .shift(1)
        .rolling(window=window, min_periods=1)
        .sum()
        .fillna(0)
        .values
    )

print("\n Customer activity features created!")

# Statistics
print(f"\nFeature statistics:")
for window in [8]:  # Just show 8-week as example
    print(f"\ncustomer_total_purchases_{window}w:")
    print(f"  Mean: {df[f'customer_total_purchases_{window}w'].mean():.2f}")
    print(f"  Max: {df[f'customer_total_purchases_{window}w'].max():.0f}")
    
    print(f"\ncustomer_unique_products_{window}w:")
    print(f"  Mean: {df[f'customer_unique_products_{window}w'].mean():.2f}")
    print(f"  Max: {df[f'customer_unique_products_{window}w'].max():.0f}")

# Sample
print("\nSample customer activity features:")
sample_cols = ['customer_id', 'product_id', 'week_start',
               'customer_total_purchases_8w', 'customer_unique_products_8w']
print(df[sample_cols].head(20).to_string(index=False))

print("="*80)
print("PHASE 2: AGGREGATION FEATURES - PRODUCT POPULARITY")
print("="*80)

print("\nCalculating product-level popularity across ALL customers...")

# For each product at each week, aggregate total demand
for window in [4, 8, 12]:
    print(f"  Processing {window}-week window...")
    
    # Total customers who bought this product (last N weeks)
    df[f'product_total_customers_{window}w'] = (
        df.groupby(['product_unit_variant_id', 'week_start'])['Target_purchase_next_1w']
        .transform('sum')  # Count customers buying this product
    )
    
    # Shift and rolling
    df[f'product_total_customers_{window}w'] = (
        df.groupby('product_unit_variant_id')[f'product_total_customers_{window}w']
        .shift(1)
        .rolling(window=window, min_periods=1)
        .sum()
        .values
    )
    
    # Total quantity sold for this product (last N weeks)
    df[f'product_total_qty_{window}w'] = (
        df.groupby(['product_unit_variant_id', 'week_start'])['Target_qty_next_1w']
        .transform('sum')  # Sum quantity across all customers
    )
    
    # Shift and rolling
    df[f'product_total_qty_{window}w'] = (
        df.groupby('product_unit_variant_id')[f'product_total_qty_{window}w']
        .shift(1)
        .rolling(window=window, min_periods=1)
        .sum()
        .values
    )

print("\n Product popularity features created!")

# Statistics
print(f"\nFeature statistics:")
for window in [8]:  # Just show 8-week as example
    print(f"\nproduct_total_customers_{window}w:")
    print(f"  Mean: {df[f'product_total_customers_{window}w'].mean():.2f}")
    print(f"  Max: {df[f'product_total_customers_{window}w'].max():.0f}")
    
    print(f"\nproduct_total_qty_{window}w:")
    print(f"  Mean: {df[f'product_total_qty_{window}w'].mean():.2f}")
    print(f"  Max: {df[f'product_total_qty_{window}w'].max():.0f}")

# Sample
print("\nSample product popularity features:")
sample_cols = ['product_unit_variant_id', 'week_start',
               'product_total_customers_8w', 'product_total_qty_8w']
print(df[sample_cols].head(20).to_string(index=False))

print("="*80)
print("PHASE 2: AGGREGATION FEATURES - CUSTOMER PREFERENCES")
print("="*80)

print("\nCalculating customer preferences for grades and units...")

# For each customer, what's their purchase distribution across grades?
for window in [8, 12]:
    print(f"  Processing {window}-week window...")
    
    # Purchases by customer × grade
    df[f'customer_grade_purchases_{window}w'] = (
        df.groupby(['customer_id', 'grade_name', 'week_start'])['Target_purchase_next_1w']
        .transform('sum')
    )
    
    # Shift and rolling
    df[f'customer_grade_purchases_{window}w'] = (
        df.groupby(['customer_id', 'grade_name'])[f'customer_grade_purchases_{window}w']
        .shift(1)
        .rolling(window=window, min_periods=1)
        .sum()
        .values
    )
    
    # Purchases by customer × unit
    df[f'customer_unit_purchases_{window}w'] = (
        df.groupby(['customer_id', 'unit_name', 'week_start'])['Target_purchase_next_1w']
        .transform('sum')
    )
    
    # Shift and rolling
    df[f'customer_unit_purchases_{window}w'] = (
        df.groupby(['customer_id', 'unit_name'])[f'customer_unit_purchases_{window}w']
        .shift(1)
        .rolling(window=window, min_periods=1)
        .sum()
        .values
    )

print("\n✓ Customer preference features created!")

# Statistics
print(f"\nFeature statistics:")
for window in [8]:
    print(f"\ncustomer_grade_purchases_{window}w:")
    print(f"  Mean: {df[f'customer_grade_purchases_{window}w'].mean():.2f}")
    print(f"  Max: {df[f'customer_grade_purchases_{window}w'].max():.0f}")
    
    print(f"\ncustomer_unit_purchases_{window}w:")
    print(f"  Mean: {df[f'customer_unit_purchases_{window}w'].mean():.2f}")
    print(f"  Max: {df[f'customer_unit_purchases_{window}w'].max():.0f}")

# Sample
print("\nSample customer preference features:")
sample_cols = ['customer_id', 'grade_name', 'unit_name', 'week_start',
               'customer_grade_purchases_8w', 'customer_unit_purchases_8w']
print(df[sample_cols].head(20).to_string(index=False))

print("="*80)
print("PHASE 2: AGGREGATION FEATURES - CUSTOMER CATEGORY PATTERNS")
print("="*80)

print("\nCalculating purchase patterns by customer category...")

# For each customer_category, what's the typical purchase behavior?
for window in [8, 12]:
    print(f"  Processing {window}-week window...")
    
    # Average purchases by customer_category
    df[f'custcat_avg_purchases_{window}w'] = (
        df.groupby(['customer_category', 'week_start'])['Target_purchase_next_1w']
        .transform('mean')  # Average purchase rate for this category
    )
    
    # Shift and rolling
    df[f'custcat_avg_purchases_{window}w'] = (
        df.groupby('customer_category')[f'custcat_avg_purchases_{window}w']
        .shift(1)
        .rolling(window=window, min_periods=1)
        .mean()
        .values
    )

print("\n Customer category features created!")

# Statistics
print(f"\nFeature statistics:")
for window in [8]:
    print(f"\ncustcat_avg_purchases_{window}w:")
    print(f"  Mean: {df[f'custcat_avg_purchases_{window}w'].mean():.4f}")
    print(f"  Max: {df[f'custcat_avg_purchases_{window}w'].max():.4f}")

print("\nSample customer category features:")
sample_cols = ['customer_category', 'week_start', 'custcat_avg_purchases_8w']
print(df[sample_cols].head(20).to_string(index=False))

print("="*80)
print("PHASE 3: TEMPORAL & SEASONAL FEATURES")
print("="*80)

print("\n1. Creating basic temporal features...")

# Extract time components
df['week_of_year'] = df['week_start'].dt.isocalendar().week
df['month'] = df['week_start'].dt.month
df['quarter'] = df['week_start'].dt.quarter
df['day_of_year'] = df['week_start'].dt.dayofyear

# Is it end of month/quarter?
df['is_month_end'] = (df['week_start'] + pd.Timedelta(days=6)).dt.is_month_end.astype(int)
df['is_quarter_end'] = (df['week_start'] + pd.Timedelta(days=6)).dt.is_quarter_end.astype(int)

# Week of month (1-5)
df['week_of_month'] = ((df['week_start'].dt.day - 1) // 7) + 1

print(" Basic temporal features created!")

# Preview
print("\nSample temporal features:")
sample_cols = ['week_start', 'week_of_year', 'month', 'quarter', 
               'week_of_month', 'is_month_end', 'is_quarter_end']
print(df[sample_cols].head(15).to_string(index=False))

print("="*80)
print("CYCLICAL ENCODING FOR SEASONALITY")
print("="*80)

print("\nEncoding cyclical features (sine/cosine transformations)...")

# Week of year (52 weeks cycle)
df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
df['week_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)

# Month (12 months cycle)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# Quarter (4 quarters cycle)
df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)

print(" Cyclical encoding complete!")

# Preview
print("\nSample cyclical features:")
sample_cols = ['week_start', 'month', 'week_sin', 'week_cos', 'month_sin', 'month_cos']
print(df[sample_cols].head(15).to_string(index=False))

print("="*80)
print("KENYAN FESTIVE PERIODS & PUBLIC HOLIDAYS")
print("="*80)

print("\nCreating Kenyan holiday and festive period features...")

# Extract year for calculations
df['year'] = df['week_start'].dt.year

# Christian Holidays (fixed dates)
df['is_christmas_season'] = ((df['month'] == 12) & (df['week_start'].dt.day >= 15)).astype(int)
df['is_new_year_week'] = ((df['month'] == 1) & (df['week_start'].dt.day <= 7)).astype(int)
df['weeks_to_christmas'] = np.where(
    df['month'] <= 12,
    ((pd.Timestamp(year=df['year'].iloc[0], month=12, day=25) - df['week_start']).dt.days / 7).clip(lower=0),
    52
)

# Easter (approximate - moves each year, typically March/April)
df['is_easter_season'] = (((df['month'] == 3) & (df['week_start'].dt.day >= 15)) | 
                           ((df['month'] == 4) & (df['week_start'].dt.day <= 15))).astype(int)

# Kenyan Public Holidays
df['is_madaraka_week'] = ((df['month'] == 6) & (df['week_start'].dt.day <= 7)).astype(int)  # June 1
df['is_mashujaa_week'] = ((df['month'] == 10) & (df['week_start'].dt.day >= 15) & (df['week_start'].dt.day <= 21)).astype(int)  # Oct 20
df['is_jamhuri_week'] = ((df['month'] == 12) & (df['week_start'].dt.day >= 8) & (df['week_start'].dt.day <= 14)).astype(int)  # Dec 12

# Ramadan and Eid (approximate - Islamic calendar shifts ~11 days earlier each year)
# 2024: Ramadan March 11 - April 9, Eid al-Fitr April 10
# 2025: Ramadan March 1 - March 29, Eid al-Fitr March 30
df['is_ramadan'] = (
    ((df['year'] == 2024) & (df['month'] == 3) & (df['week_start'].dt.day >= 11)) |
    ((df['year'] == 2024) & (df['month'] == 4) & (df['week_start'].dt.day <= 9)) |
    ((df['year'] == 2025) & (df['month'] == 3) & (df['week_start'].dt.day <= 29))
).astype(int)

df['is_eid'] = (
    ((df['year'] == 2024) & (df['month'] == 4) & (df['week_start'].dt.day >= 10) & (df['week_start'].dt.day <= 17)) |
    ((df['year'] == 2025) & (df['month'] == 3) & (df['week_start'].dt.day >= 30)) |
    ((df['year'] == 2025) & (df['month'] == 4) & (df['week_start'].dt.day <= 6))
).astype(int)

# General festive season (November - December)
df['is_festive_season'] = ((df['month'] >= 11)).astype(int)

print(" Kenyan holiday features created!")

# Summary
holiday_features = ['is_christmas_season', 'is_new_year_week', 'is_easter_season', 
                   'is_madaraka_week', 'is_mashujaa_week', 'is_jamhuri_week',
                   'is_ramadan', 'is_eid', 'is_festive_season']

print(f"\nHoliday features distribution:")
for feat in holiday_features:
    count = df[feat].sum()
    pct = (count / len(df)) * 100
    print(f"  {feat}: {count:,} weeks ({pct:.2f}%)")

print("="*80)
print("KENYAN SCHOOL TERM FEATURES")
print("="*80)

print("\nCreating school term features (important for commercial kitchens)...")

# Kenyan school calendar (approximate):
# Term 1: January - March (weeks 1-12)
# Holiday: April (weeks 13-16)
# Term 2: May - July (weeks 17-30)
# Holiday: August (weeks 31-35)
# Term 3: September - November (weeks 36-47)
# Holiday: December (weeks 48-52)

def get_school_term(month):
    """Determine school term based on month"""
    if month in [1, 2, 3]:
        return 1
    elif month == 4:
        return 0  # Holiday
    elif month in [5, 6, 7]:
        return 2
    elif month == 8:
        return 0  # Holiday
    elif month in [9, 10, 11]:
        return 3
    else:  # December
        return 0  # Holiday

df['school_term'] = df['month'].apply(get_school_term)

# Binary flags
df['is_school_term'] = (df['school_term'] > 0).astype(int)
df['is_school_holiday'] = (df['school_term'] == 0).astype(int)

# Specific term indicators
df['is_term_1'] = (df['school_term'] == 1).astype(int)
df['is_term_2'] = (df['school_term'] == 2).astype(int)
df['is_term_3'] = (df['school_term'] == 3).astype(int)

print(" School term features created!")

# Distribution
print(f"\nSchool term distribution:")
print(f"  In term (school open): {df['is_school_term'].sum():,} weeks ({df['is_school_term'].mean()*100:.1f}%)")
print(f"  Holiday (school closed): {df['is_school_holiday'].sum():,} weeks ({df['is_school_holiday'].mean()*100:.1f}%)")
print(f"\n  Term 1: {df['is_term_1'].sum():,} weeks")
print(f"  Term 2: {df['is_term_2'].sum():,} weeks")
print(f"  Term 3: {df['is_term_3'].sum():,} weeks")

# Sample
print("\nSample school term features:")
sample_cols = ['week_start', 'month', 'school_term', 'is_school_term', 'is_school_holiday']
print(df[sample_cols].head(20).to_string(index=False))

print("="*80)
print("KENYAN AGRICULTURAL SEASONS")
print("="*80)

print("\nCreating agricultural season features (affects fresh produce supply)...")

# Kenya has two rainy seasons (planting) and two dry seasons (harvest)
# Long rains: March - May (planting)
# Long dry: June - September (harvest)
# Short rains: October - December (planting)
# Short dry: January - February (harvest)

def get_agricultural_season(month):
    """Determine agricultural season"""
    if month in [3, 4, 5]:
        return 'long_rains'
    elif month in [6, 7, 8, 9]:
        return 'long_dry'
    elif month in [10, 11, 12]:
        return 'short_rains'
    else:  # Jan, Feb
        return 'short_dry'

df['agri_season'] = df['month'].apply(get_agricultural_season)

# Binary indicators
df['is_planting_season'] = df['agri_season'].isin(['long_rains', 'short_rains']).astype(int)
df['is_harvest_season'] = df['agri_season'].isin(['long_dry', 'short_dry']).astype(int)
df['is_long_rains'] = (df['agri_season'] == 'long_rains').astype(int)
df['is_long_dry'] = (df['agri_season'] == 'long_dry').astype(int)
df['is_short_rains'] = (df['agri_season'] == 'short_rains').astype(int)
df['is_short_dry'] = (df['agri_season'] == 'short_dry').astype(int)

print(" Agricultural season features created!")

# Distribution
print(f"\nAgricultural season distribution:")
print(f"  Planting seasons: {df['is_planting_season'].sum():,} weeks ({df['is_planting_season'].mean()*100:.1f}%)")
print(f"  Harvest seasons: {df['is_harvest_season'].sum():,} weeks ({df['is_harvest_season'].mean()*100:.1f}%)")

# Sample
print("\nSample agricultural season features:")
sample_cols = ['week_start', 'month', 'agri_season', 'is_planting_season', 'is_harvest_season']
print(df[sample_cols].head(20).to_string(index=False))

print("="*80)
print("PHASE 3 SUMMARY - TEMPORAL & SEASONAL FEATURES")
print("="*80)

# List all Phase 3 features
phase3_features = [
    # Basic temporal
    'week_of_year', 'month', 'quarter', 'day_of_year', 'week_of_month',
    'is_month_end', 'is_quarter_end',
    
    # Cyclical encoding
    'week_sin', 'week_cos', 'month_sin', 'month_cos', 'quarter_sin', 'quarter_cos',
    
    # Kenyan holidays & festive
    'is_christmas_season', 'is_new_year_week', 'weeks_to_christmas', 'is_easter_season',
    'is_madaraka_week', 'is_mashujaa_week', 'is_jamhuri_week',
    'is_ramadan', 'is_eid', 'is_festive_season',
    
    # School terms
    'school_term', 'is_school_term', 'is_school_holiday',
    'is_term_1', 'is_term_2', 'is_term_3',
    
    # Agricultural seasons
    'is_planting_season', 'is_harvest_season',
    'is_long_rains', 'is_long_dry', 'is_short_rains', 'is_short_dry'
]

print(f"\nTotal Phase 3 features created: {len(phase3_features)}")

# Check for missing values
nan_count_total = 0
for feat in phase3_features:
    nan_count = df[feat].isnull().sum()
    if nan_count > 0:
        nan_count_total += nan_count
        print(f"  {feat}: {nan_count:,} NaNs")

if nan_count_total == 0:
    print("\n No missing values in Phase 3 features!")

# Top correlations with target
print("\n\nTop 10 Phase 3 features by correlation with target:")
train_mask = df['is_train'] == 1
correlations = []

for feat in phase3_features:
    if df[feat].dtype in ['int64', 'float64']:  # Only numeric features
        corr = df[train_mask][[feat, 'Target_purchase_next_1w']].corr().iloc[0, 1]
        correlations.append((feat, corr))

correlations.sort(key=lambda x: abs(x[1]), reverse=True)
for feat, corr in correlations[:10]:
    print(f"  {feat}: {corr:.4f}")

print("\n" + "="*80)
print(" PHASE 3 COMPLETE!")
print("="*80)
print(f"\nDataset now has {df.shape[1]} columns")
print(f"Features added in Phase 3: {len(phase3_features)}")

print("="*80)
print("PHASE 4: ADVANCED PATTERN FEATURES")
print("="*80)

print("\n1. Creating trend features (purchase growth/decline)...")

# Sort to ensure proper order
df = df.sort_values(['customer_id', 'product_unit_variant_id', 'week_start']).reset_index(drop=True)

# Purchase trend: comparing recent vs older history
df['purchase_trend_4w_vs_8w'] = df['purchase_count_last_4w'] - (df['purchase_count_last_8w'] - df['purchase_count_last_4w'])
df['purchase_trend_8w_vs_12w'] = df['purchase_count_last_8w'] - (df['purchase_count_last_12w'] - df['purchase_count_last_8w'])

# Quantity trend
df['qty_trend_4w_vs_8w'] = df['total_qty_last_4w'] - (df['total_qty_last_8w'] - df['total_qty_last_4w'])
df['qty_trend_8w_vs_12w'] = df['total_qty_last_8w'] - (df['total_qty_last_12w'] - df['total_qty_last_8w'])

# Acceleration (trend of trend)
df['purchase_acceleration'] = df['purchase_trend_4w_vs_8w'] - df['purchase_trend_8w_vs_12w']

print(" Trend features created!")

print("\nSample trend features:")
sample_cols = ['customer_id', 'product_unit_variant_id', 'week_start', 
               'purchase_trend_4w_vs_8w', 'qty_trend_4w_vs_8w', 'purchase_acceleration']
print(df[sample_cols].head(20).to_string(index=False))

print("="*80)
print("STREAK FEATURES")
print("="*80)

print("\nCreating streak features (consecutive behavior patterns)...")

# Fill NaN with 0 for streak calculations
purchased_1w = df['purchased_last_1w'].fillna(0).astype(int)
purchased_2w = df['purchased_last_2w'].fillna(0).astype(int)
purchased_3w = df['purchased_last_3w'].fillna(0).astype(int)
purchased_4w = df['purchased_last_4w'].fillna(0).astype(int)

# Current streak of purchases (how many consecutive weeks purchased?)
df['current_purchase_streak'] = (
    purchased_1w + 
    purchased_2w * purchased_1w +
    purchased_3w * purchased_2w * purchased_1w +
    purchased_4w * purchased_3w * purchased_2w * purchased_1w
)

# Current streak of non-purchases
df['current_no_purchase_streak'] = (
    (1 - purchased_1w) + 
    (1 - purchased_2w) * (1 - purchased_1w) +
    (1 - purchased_3w) * (1 - purchased_2w) * (1 - purchased_1w) +
    (1 - purchased_4w) * (1 - purchased_3w) * (1 - purchased_2w) * (1 - purchased_1w)
)

# Has regular pattern (purchases every 2-3 weeks?)
df['has_regular_pattern'] = (
    ((df['purchase_count_last_8w'] >= 3) & (df['purchase_count_last_8w'] <= 5)) |
    ((df['purchase_count_last_12w'] >= 4) & (df['purchase_count_last_12w'] <= 7))
).astype(int)

print(" Streak features created!")

print("\nStreak distribution:")
print(f"  Current purchase streak 0: {(df['current_purchase_streak'] == 0).sum():,}")
print(f"  Current purchase streak 1: {(df['current_purchase_streak'] == 1).sum():,}")
print(f"  Current purchase streak 2: {(df['current_purchase_streak'] == 2).sum():,}")
print(f"  Current purchase streak 3: {(df['current_purchase_streak'] == 3).sum():,}")
print(f"  Current purchase streak 4: {(df['current_purchase_streak'] == 4).sum():,}")
print(f"\n  Has regular pattern: {df['has_regular_pattern'].sum():,} ({df['has_regular_pattern'].mean()*100:.2f}%)")

print("="*80)
print("VOLATILITY FEATURES")
print("="*80)

print("\nCreating volatility features (purchase consistency)...")

# Purchase rate volatility: how consistent is the purchase rate?
df['purchase_rate_volatility'] = (df['purchase_rate_last_4w'] - df['purchase_rate_last_8w']).abs()

# Quantity volatility: standard deviation approximation
# (avg_qty_12w - avg_qty_4w)^2 as proxy for variance
df['qty_volatility'] = (df['avg_qty_per_purchase_last_12w'] - df['avg_qty_per_purchase_last_4w']).abs()

# Switching behavior: did customer change their purchase pattern?
df['is_returning_customer'] = ((df['purchase_count_last_4w'] > 0) & (df['purchase_count_last_12w'] > df['purchase_count_last_4w'])).astype(int)
df['is_new_active'] = ((df['purchase_count_last_4w'] > 0) & (df['purchase_count_last_12w'] == df['purchase_count_last_4w'])).astype(int)
df['is_churning'] = ((df['purchase_count_last_4w'] == 0) & (df['purchase_count_last_12w'] > 0)).astype(int)

print("✓ Volatility features created!")

print("\nCustomer state distribution:")
print(f"  Returning customers: {df['is_returning_customer'].sum():,} ({df['is_returning_customer'].mean()*100:.2f}%)")
print(f"  New active customers: {df['is_new_active'].sum():,} ({df['is_new_active'].mean()*100:.2f}%)")
print(f"  Churning customers: {df['is_churning'].sum():,} ({df['is_churning'].mean()*100:.2f}%)")

print("="*80)
print("PRODUCT LIFECYCLE FEATURES")
print("="*80)

print("\nCreating product lifecycle features...")

# Product age (weeks since first appearance in data)
df['product_age_weeks'] = df.groupby('product_unit_variant_id')['week_start'].rank(method='dense') - 1

# Is this a new product for this customer?
df['is_new_product_for_customer'] = (df['purchase_count_last_12w'].fillna(0) == 0).astype(int)

# Customer's product diversity
df['customer_product_diversity'] = df['customer_unique_products_8w'] / (df['customer_total_purchases_8w'] + 1)

# Product concentration: is this product dominating customer's purchases?
# Simple approach: ratio of this product's purchases to customer's total purchases
df['product_share_of_customer'] = (
    df['purchase_count_last_8w'] / (df['customer_total_purchases_8w'] + 1)
).fillna(0)

print(" Product lifecycle features created!")

print("\nProduct lifecycle summary:")
print(f"  Product age (weeks) - mean: {df['product_age_weeks'].mean():.1f}, max: {df['product_age_weeks'].max():.0f}")
print(f"  New products for customer: {df['is_new_product_for_customer'].sum():,} ({df['is_new_product_for_customer'].mean()*100:.2f}%)")
print(f"  Product diversity - mean: {df['customer_product_diversity'].mean():.3f}")
print(f"  Product share of customer - mean: {df['product_share_of_customer'].mean():.4f}")

print("="*80)
print("RATIO & INTERACTION FEATURES")
print("="*80)

print("\nCreating ratio and interaction features...")

# Customer's share of product demand
df['customer_share_of_product'] = (
    df['purchase_count_last_8w'] / (df['product_total_customers_8w'] + 1)
)

# Product's share of customer spend (purchase-based)
# Already have: product_share_of_customer

# Customer activity ratio: recent vs historical
df['customer_activity_ratio'] = (
    df['customer_total_purchases_4w'] / (df['customer_total_purchases_8w'] - df['customer_total_purchases_4w'] + 1)
)

# Product popularity ratio: recent vs historical
df['product_popularity_ratio'] = (
    df['product_total_customers_4w'] / (df['product_total_customers_8w'] - df['product_total_customers_4w'] + 1)
)

# Cross features: customer category × school term
df['custcat_x_school_term'] = df['customer_category'].astype(str) + '_' + df['school_term'].astype(str)

# Customer status × festive season
df['custstatus_x_festive'] = df['customer_status'].astype(str) + '_' + df['is_festive_season'].astype(str)

# Grade × agricultural season
df['grade_x_agri_season'] = df['grade_name'].astype(str) + '_' + df['agri_season'].astype(str)

print(" Ratio & interaction features created!")

print("\nRatio features summary:")
print(f"  Customer share of product - mean: {df['customer_share_of_product'].mean():.4f}")
print(f"  Customer activity ratio - mean: {df['customer_activity_ratio'].mean():.3f}")
print(f"  Product popularity ratio - mean: {df['product_popularity_ratio'].mean():.3f}")

print(f"\nInteraction features:")
print(f"  custcat_x_school_term - {df['custcat_x_school_term'].nunique()} unique values")
print(f"  custstatus_x_festive - {df['custstatus_x_festive'].nunique()} unique values")
print(f"  grade_x_agri_season - {df['grade_x_agri_season'].nunique()} unique values")

print("="*80)
print("DATA PREPROCESSING FOR MODEL TRAINING")
print("="*80)

print("\nStep 1: Separating train and test datasets...")

# Separate based on is_train flag
df_train = df[df['is_train'] == 1].copy()
df_test = df[df['is_train'] == 0].copy()

print(f"Train shape: {df_train.shape}")
print(f" Test shape: {df_test.shape}")

# Drop the is_train flag
df_train.drop('is_train', axis=1, inplace=True)
df_test.drop('is_train', axis=1, inplace=True)

print("\n" + "="*80)
print("ASSESSMENT: Missing Values and Data Types")
print("="*80)

# Check missing values in train
print("\nMissing values in TRAIN set:")
missing_train = df_train.isnull().sum()
missing_train = missing_train[missing_train > 0].sort_values(ascending=False)
if len(missing_train) > 0:
    print(f"\nFeatures with missing values: {len(missing_train)}")
    for col, count in missing_train.head(20).items():
        pct = (count / len(df_train)) * 100
        print(f"  {col}: {count:,} ({pct:.2f}%)")
else:
    print("  No missing values!")

# Check missing values in test
print("\nMissing values in TEST set:")
missing_test = df_test.isnull().sum()
missing_test = missing_test[missing_test > 0].sort_values(ascending=False)
if len(missing_test) > 0:
    print(f"\nFeatures with missing values: {len(missing_test)}")
    for col, count in missing_test.head(20).items():
        pct = (count / len(df_test)) * 100
        print(f"  {col}: {count:,} ({pct:.2f}%)")
else:
    print("  No missing values!")

# Data types summary
print("\n" + "="*80)
print("Data Types Summary:")
print("="*80)
print(df_train.dtypes.value_counts())

print("="*80)
print("IDENTIFYING FEATURE GROUPS")
print("="*80)

# Target columns (only in train)
target_cols = ['Target_purchase_next_1w', 'Target_qty_next_1w', 
               'Target_purchase_next_2w', 'Target_qty_next_2w']

# ID columns (not for training, but keep for tracking)
id_cols = ['customer_id', 'product_unit_variant_id', 'product_id', 
           'product_grade_variant_id']

# Datetime columns
datetime_cols = ['week_start', 'customer_created_at']

# Categorical columns (object/string type)
categorical_cols = df_train.select_dtypes(include=['object']).columns.tolist()
# Remove datetime cols if accidentally included
categorical_cols = [col for col in categorical_cols if col not in datetime_cols]

# Numeric columns (excluding targets and IDs)
numeric_cols = df_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
numeric_cols = [col for col in numeric_cols if col not in target_cols + id_cols]

print(f"Target columns: {len(target_cols)}")
print(f"  {target_cols}")

print(f"\nID columns: {len(id_cols)}")
print(f"  {id_cols}")

print(f"\nDatetime columns: {len(datetime_cols)}")
print(f"  {datetime_cols}")

print(f"\nCategorical columns: {len(categorical_cols)}")
print(f"  {categorical_cols}")

print(f"\nNumeric columns: {len(numeric_cols)}")
print(f"  First 20: {numeric_cols[:20]}")

# Verify total
total_features = len(categorical_cols) + len(numeric_cols)
print(f"\n Total feature columns: {total_features}")

print("="*80)
print("HANDLING MISSING VALUES - NUMERIC FEATURES")
print("="*80)

print("\nFilling missing values in numeric features...")

# Strategy:
# - Lag features: Fill with 0 (no purchase history)
# - Count features: Fill with 0 (no activity)
# - Rate features: Fill with 0 (no purchase rate)
# - Average features: Fill with 0 (no purchases to average)
# - Other features: Fill with -999 as a flag value

# Define fillna strategies
lag_features = [col for col in numeric_cols if 'last_' in col or 'purchased_' in col or 'qty_purchased_' in col]
count_features = [col for col in numeric_cols if 'count_' in col or 'total_' in col or 'unique_' in col or 'customers_' in col]
rate_features = [col for col in numeric_cols if 'rate_' in col or 'ratio_' in col or 'share_' in col or 'diversity' in col]
avg_features = [col for col in numeric_cols if 'avg_' in col]
trend_features = [col for col in numeric_cols if 'trend_' in col or 'acceleration' in col]
streak_features = [col for col in numeric_cols if 'streak' in col]
volatility_features = [col for col in numeric_cols if 'volatility' in col]

# Fill with 0 for most engineered features
fill_zero_cols = list(set(lag_features + count_features + rate_features + avg_features + 
                          trend_features + streak_features + volatility_features))

# Fill with 0
for col in fill_zero_cols:
    if col in df_train.columns:
        df_train[col].fillna(0, inplace=True)
        df_test[col].fillna(0, inplace=True)

print(f" Filled {len(fill_zero_cols)} numeric features with 0")

# Check remaining missing values
remaining_missing_train = df_train[numeric_cols].isnull().sum()
remaining_missing_train = remaining_missing_train[remaining_missing_train > 0]

remaining_missing_test = df_test[numeric_cols].isnull().sum()
remaining_missing_test = remaining_missing_test[remaining_missing_test > 0]

if len(remaining_missing_train) > 0:
    print(f"\nRemaining missing values in TRAIN (numeric):")
    for col, count in remaining_missing_train.items():
        print(f"  {col}: {count:,}")
        # Fill remaining with -999
        df_train[col].fillna(-999, inplace=True)
        df_test[col].fillna(-999, inplace=True)
    print("   Filled remaining with -999")
else:
    print("\n No remaining missing values in numeric features!")

print("\n Numeric features preprocessing complete!")

print("="*80)
print("HANDLING MISSING VALUES - CATEGORICAL FEATURES")
print("="*80)

print("\nChecking categorical features for missing values...")

missing_cat_train = df_train[categorical_cols].isnull().sum()
missing_cat_train = missing_cat_train[missing_cat_train > 0]

missing_cat_test = df_test[categorical_cols].isnull().sum()
missing_cat_test = missing_cat_test[missing_cat_test > 0]

if len(missing_cat_train) > 0 or len(missing_cat_test) > 0:
    print(f"\nMissing categorical values found!")
    
    if len(missing_cat_train) > 0:
        print(f"\nTRAIN:")
        for col, count in missing_cat_train.items():
            print(f"  {col}: {count:,}")
    
    if len(missing_cat_test) > 0:
        print(f"\nTEST:")
        for col, count in missing_cat_test.items():
            print(f"  {col}: {count:,}")
    
    # Fill with 'MISSING' category
    for col in categorical_cols:
        df_train[col].fillna('MISSING', inplace=True)
        df_test[col].fillna('MISSING', inplace=True)
    
    print("\n Filled categorical missing values with 'MISSING'")
else:
    print(" No missing values in categorical features!")

print("\n Categorical features preprocessing complete!")

print("="*80)
print("ENCODING CATEGORICAL FEATURES - TARGET ENCODING")
print("="*80)

print("\nUsing Target Encoding for categorical features...")
print("(Better than one-hot encoding for high cardinality and tree-based models)")

from sklearn.preprocessing import LabelEncoder

# We'll use target encoding with smoothing to prevent overfitting
# For each categorical feature, calculate mean target per category

def target_encode_with_smoothing(train_df, test_df, cat_col, target_col, alpha=10):
    """
    Target encode a categorical column with smoothing
    alpha: smoothing parameter (higher = more regularization)
    """
    # Global mean
    global_mean = train_df[target_col].mean()
    
    # Calculate statistics per category
    agg = train_df.groupby(cat_col)[target_col].agg(['mean', 'count'])
    
    # Smoothed mean = (count * category_mean + alpha * global_mean) / (count + alpha)
    smoothed_mean = (agg['count'] * agg['mean'] + alpha * global_mean) / (agg['count'] + alpha)
    
    # Create encoding dictionary
    encoding_dict = smoothed_mean.to_dict()
    
    # Apply to train and test
    train_encoded = train_df[cat_col].map(encoding_dict).fillna(global_mean)
    test_encoded = test_df[cat_col].map(encoding_dict).fillna(global_mean)
    
    return train_encoded, test_encoded

# Target encode using Target_purchase_next_1w (binary target)
print("\nTarget encoding categorical features...")

alpha = 10  # Smoothing parameter

for col in categorical_cols:
    print(f"  Encoding {col}...")
    
    # Create new column name
    new_col_name = f"{col}_target_encoded"
    
    # Target encode
    df_train[new_col_name], df_test[new_col_name] = target_encode_with_smoothing(
        df_train, df_test, col, 'Target_purchase_next_1w', alpha=alpha
    )

print(f"\n Target encoded {len(categorical_cols)} categorical features!")

# Keep original categorical columns for now (might use label encoding too)
print("\nAlso applying Label Encoding as backup...")

for col in categorical_cols:
    le = LabelEncoder()
    
    # Fit on combined train+test to ensure same encoding
    combined_values = pd.concat([df_train[col], df_test[col]], axis=0)
    le.fit(combined_values)
    
    # Transform
    new_col_name = f"{col}_label_encoded"
    df_train[new_col_name] = le.transform(df_train[col])
    df_test[new_col_name] = le.transform(df_test[col])

print(f" Label encoded {len(categorical_cols)} categorical features!")



print("="*80)
print("FEATURE ENGINEERING SUMMARY")
print("="*80)

print(f"\nTrain shape: {df_train.shape}")
print(f"Test shape: {df_test.shape}")

# Count feature types
target_encoded_cols = [col for col in df_train.columns if '_target_encoded' in col]
label_encoded_cols = [col for col in df_train.columns if '_label_encoded' in col]

print(f"\nFeature breakdown:")
print(f"  Numeric features: {len(numeric_cols)}")
print(f"  Target encoded features: {len(target_encoded_cols)}")
print(f"  Label encoded features: {len(label_encoded_cols)}")
print(f"  Original categorical: {len(categorical_cols)}")
print(f"  ID columns: {len(id_cols)}")
print(f"  Datetime columns: {len(datetime_cols)}")
print(f"  Target columns (train only): {len(target_cols)}")

# Final missing value check
print("\n" + "="*80)
print("FINAL MISSING VALUE CHECK")
print("="*80)

missing_train_final = df_train.isnull().sum().sum()
missing_test_final = df_test.isnull().sum().sum()

print(f"\nTrain missing values: {missing_train_final:,}")
print(f"Test missing values: {missing_test_final:,}")

if missing_train_final == 0 and missing_test_final == 0:
    print("\n NO MISSING VALUES - READY FOR MODELING! ✓✓✓")
else:
    print("\n WARNING: Still have missing values!")
    if missing_train_final > 0:
        print("\nTrain columns with NaN:")
        print(df_train.isnull().sum()[df_train.isnull().sum() > 0])
    if missing_test_final > 0:
        print("\nTest columns with NaN:")
        print(df_test.isnull().sum()[df_test.isnull().sum() > 0])

# Check for infinite values
inf_train = np.isinf(df_train.select_dtypes(include=[np.number])).sum().sum()
inf_test = np.isinf(df_test.select_dtypes(include=[np.number])).sum().sum()

print(f"\nTrain infinite values: {inf_train:,}")
print(f"Test infinite values: {inf_test:,}")

if inf_train == 0 and inf_test == 0:
    print(" No infinite values!")
else:
    print(" WARNING: Infinite values detected!")

print("="*80)
print("PREPARING FINAL FEATURE SETS")
print("="*80)

# Get all columns from train (excluding targets and week_start)
all_train_cols = df_train.columns.tolist()

# Features = everything except targets and week_start (keep week_start for tracking but not modeling)
feature_cols = [col for col in all_train_cols if col not in target_cols]

# Separate ID columns (might want to exclude from modeling)
id_cols_present = [col for col in id_cols if col in feature_cols]

# Modeling features = all features (we can decide to exclude IDs during training)
modeling_features_with_ids = feature_cols.copy()
modeling_features_no_ids = [col for col in feature_cols if col not in id_cols and col != 'week_start']

print(f" Total columns in train: {len(all_train_cols)}")
print(f"  - Target columns: {len(target_cols)}")
print(f"  - Feature columns (with IDs): {len(modeling_features_with_ids)}")
print(f"  - Feature columns (without IDs and week_start): {len(modeling_features_no_ids)}")

print(f"\n Total columns in test: {len(df_test.columns)}")

# Verify train and test have same feature columns
test_cols = df_test.columns.tolist()
train_feature_cols = [col for col in feature_cols if col not in target_cols]

missing_in_test = set(train_feature_cols) - set(test_cols)
extra_in_test = set(test_cols) - set(train_feature_cols)

if len(missing_in_test) == 0 and len(extra_in_test) == 0:
    print("\n Train and test have IDENTICAL feature columns!")
else:
    if len(missing_in_test) > 0:
        print(f"\n Missing in test: {missing_in_test}")
    if len(extra_in_test) > 0:
        print(f" Extra in test: {extra_in_test}")

print("\n Feature sets prepared!")

print("="*80)
print("SAVING PROCESSED DATASETS")
print("="*80)

print("\nSaving processed train and test datasets...")

# Save full datasets (with all columns)
train_output_path = 'train_processed.csv'
test_output_path = 'test_processed.csv'

df_train.to_csv(train_output_path, index=False)
print(f" Saved: {train_output_path}")
print(f"  Shape: {df_train.shape}")
print(f"  Columns: {list(df_train.columns[:10])}... (showing first 10)")

df_test.to_csv(test_output_path, index=False)
print(f" Saved: {test_output_path}")
print(f"  Shape: {df_test.shape}")
print(f"  Columns: {list(df_test.columns[:10])}... (showing first 10)")

