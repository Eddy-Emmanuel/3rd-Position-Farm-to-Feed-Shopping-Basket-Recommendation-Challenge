# ============================================================================
# CELL 1: IMPORTS
# ============================================================================

import pandas as pd
import numpy as np
import lightgbm as lgb
import gc
import warnings
warnings.filterwarnings('ignore')

print(" Libraries imported")

# ============================================================================
# CELL 28: LOAD RAW TRANSACTION DATA
# ============================================================================

print("\n" + "="*80)
print("LOADING RAW TRANSACTION DATA")
print("="*80)

# Load the raw data
train_raw = pd.read_csv('Train.csv')
test_raw = pd.read_csv('Test.csv')

print(f"\n Raw train shape: {train_raw.shape}")
print(f" Raw test shape: {test_raw.shape}")

print(f"\nRaw train columns ({len(train_raw.columns)}):")
print(train_raw.columns.tolist())

print(f"\nFirst 5 rows of raw data:")
print(train_raw.head().to_string())

print(f"\nData types:")
print(train_raw.dtypes)

# Check for date columns
print("\n" + "="*60)
print("IDENTIFYING DATE COLUMNS")
print("="*60)

date_candidates = [col for col in train_raw.columns if any(x in col.lower() for x in ['date', 'time', 'day', 'week'])]
print(f"\nPotential date columns: {date_candidates}")

# Sample values from key columns
print("\n" + "="*60)
print("SAMPLE VALUES FROM KEY COLUMNS")
print("="*60)

for col in train_raw.columns[:15]:
    print(f"\n{col}:")
    print(f"  Sample values: {train_raw[col].head(3).tolist()}")
    print(f"  Unique count: {train_raw[col].nunique():,}")
    print(f"  Data type: {train_raw[col].dtype}")

# ============================================================================
# CELL 29: CREATE NEW FEATURES - RECENCY & MOMENTUM
# ============================================================================

print("\n" + "="*80)
print("FEATURE ENGINEERING - HIGH PRIORITY FEATURES")
print("="*80)

# Convert dates
train_raw['week_start'] = pd.to_datetime(train_raw['week_start'])
train_raw['customer_created_at'] = pd.to_datetime(train_raw['customer_created_at'])

test_raw['week_start'] = pd.to_datetime(test_raw['week_start'])
test_raw['customer_created_at'] = pd.to_datetime(test_raw['customer_created_at'])

print("\n✓ Dates converted to datetime")

# Combine train and test for consistent feature engineering
# Add a flag to identify which is which
train_raw['is_train'] = 1
test_raw['is_train'] = 0

# Align columns - test doesn't have targets
target_cols = ['Target_qty_next_1w', 'Target_purchase_next_1w', 'Target_qty_next_2w', 'Target_purchase_next_2w']
for col in target_cols:
    if col not in test_raw.columns:
        test_raw[col] = np.nan

# Also add current week info to test (all zeros since test is prediction period)
test_week_cols = ['qty_this_week', 'num_orders_week', 'spend_this_week', 'purchased_this_week', 'selling_price']
for col in test_week_cols:
    if col not in test_raw.columns:
        test_raw[col] = 0

print(f"\n Train columns: {len(train_raw.columns)}")
print(f" Test columns: {len(test_raw.columns)}")

# Combine
all_data = pd.concat([train_raw, test_raw], axis=0, ignore_index=True)
all_data = all_data.sort_values(['customer_id', 'product_unit_variant_id', 'week_start']).reset_index(drop=True)

print(f"\n✓ Combined data shape: {all_data.shape}")
print(f"  Train rows: {all_data['is_train'].sum():,}")
print(f"  Test rows: {(all_data['is_train'] == 0).sum():,}")

print("\n" + "="*80)
print("CREATING NEW FEATURES")
print("="*80)

# Feature 1: Days since last purchase (ANY product)
print("\n1. Creating: days_since_last_purchase_any_product...")

all_data_sorted = all_data.sort_values(['customer_id', 'week_start'])

# Get last purchase date for each customer
customer_last_purchase = []
for customer_id, group in all_data_sorted.groupby('customer_id'):
    group = group.copy()
    group['last_purchase_date'] = None
    
    last_purchase = None
    for idx, row in group.iterrows():
        if last_purchase is not None:
            days_diff = (row['week_start'] - last_purchase).days
        else:
            days_diff = np.nan
        
        customer_last_purchase.append({
            'index': idx,
            'days_since_last_purchase_any': days_diff
        })
        
        if row['purchased_this_week'] == 1:
            last_purchase = row['week_start']

days_since_df = pd.DataFrame(customer_last_purchase)
all_data = all_data.merge(days_since_df, left_index=True, right_on='index', how='left')
all_data.drop('index', axis=1, inplace=True)

print(f"    Created: days_since_last_purchase_any")
print(f"     Non-null: {all_data['days_since_last_purchase_any'].notna().sum():,}")
print(f"     Mean: {all_data['days_since_last_purchase_any'].mean():.2f} days")

# Feature 2: Days since last purchase of THIS specific product
print("\n2. Creating: days_since_last_purchase_this_product...")

product_last_purchase = []
for (customer_id, product_id), group in all_data.groupby(['customer_id', 'product_unit_variant_id']):
    group = group.sort_values('week_start').copy()
    
    last_purchase = None
    for idx, row in group.iterrows():
        if last_purchase is not None:
            days_diff = (row['week_start'] - last_purchase).days
        else:
            days_diff = np.nan
        
        product_last_purchase.append({
            'index': idx,
            'days_since_last_purchase_this_product': days_diff
        })
        
        if row['purchased_this_week'] == 1:
            last_purchase = row['week_start']

days_since_product_df = pd.DataFrame(product_last_purchase)
all_data = all_data.merge(days_since_product_df, left_index=True, right_on='index', how='left')
all_data.drop('index', axis=1, inplace=True)

print(f"    Created: days_since_last_purchase_this_product")
print(f"     Non-null: {all_data['days_since_last_purchase_this_product'].notna().sum():,}")
print(f"     Mean: {all_data['days_since_last_purchase_this_product'].mean():.2f} days")

print("\n Recency features complete!")
print(f"\nNew data shape: {all_data.shape}")

# ============================================================================
# CELL 30 (LEAK-FREE): CREATE MOMENTUM & REGULARITY FEATURES
# ============================================================================

print("\n" + "="*80)
print("CREATING MOMENTUM & REGULARITY FEATURES (LEAK-FREE)")
print("="*80)

# Sort data once
all_data = all_data.sort_values(['customer_id', 'product_unit_variant_id', 'week_start']).reset_index(drop=True)

# Feature 3: Consecutive weeks purchased (vectorized)
print("\n3. Creating: consecutive_weeks_purchased...")

all_data['consecutive_weeks_purchased'] = all_data.groupby(['customer_id', 'product_unit_variant_id'])['purchased_this_week'].apply(
    lambda x: x.groupby((x != x.shift()).cumsum()).cumsum()
).reset_index(level=[0,1], drop=True)

# Set to 0 where not purchased
all_data.loc[all_data['purchased_this_week'] == 0, 'consecutive_weeks_purchased'] = 0

print(f"    Created: consecutive_weeks_purchased")
print(f"     Max: {all_data['consecutive_weeks_purchased'].max():.0f} weeks")
print(f"     Mean: {all_data['consecutive_weeks_purchased'].mean():.2f} weeks")

# Feature 4: Consecutive weeks NOT purchased (vectorized)
print("\n4. Creating: consecutive_weeks_not_purchased...")

all_data['consecutive_weeks_not_purchased'] = all_data.groupby(['customer_id', 'product_unit_variant_id'])['purchased_this_week'].apply(
    lambda x: ((1-x).groupby((x != x.shift()).cumsum()).cumsum())
).reset_index(level=[0,1], drop=True)

# Set to 0 where purchased
all_data.loc[all_data['purchased_this_week'] == 1, 'consecutive_weeks_not_purchased'] = 0

print(f"   Created: consecutive_weeks_not_purchased")
print(f"     Max: {all_data['consecutive_weeks_not_purchased'].max():.0f} weeks")
print(f"     Mean: {all_data['consecutive_weeks_not_purchased'].mean():.2f} weeks")

# Feature 5: Purchase momentum (rolling weighted average)
print("\n5. Creating: purchase_momentum_4w...")

def weighted_purchase_momentum(group):
    # Calculate weighted average of last 4 purchases (more recent = higher weight)
    result = group['purchased_this_week'].rolling(window=4, min_periods=3).apply(
        lambda x: np.average(x, weights=[1,2,3,4][-len(x):]) if len(x) >= 3 else np.nan,
        raw=True
    )
    return result.shift(1)  # Shift to avoid leakage

all_data['purchase_momentum_4w'] = all_data.groupby(['customer_id', 'product_unit_variant_id']).apply(
    weighted_purchase_momentum
).reset_index(level=[0,1], drop=True)

print(f"    Created: purchase_momentum_4w")
print(f"     Non-null: {all_data['purchase_momentum_4w'].notna().sum():,}")
print(f"     Mean: {all_data['purchase_momentum_4w'].mean():.4f}")

# Feature 6: Quantity momentum
print("\n6. Creating: quantity_momentum_4w...")

def weighted_qty_momentum(group):
    result = group['qty_this_week'].rolling(window=4, min_periods=3).apply(
        lambda x: np.average(x, weights=[1,2,3,4][-len(x):]) if len(x) >= 3 else np.nan,
        raw=True
    )
    return result.shift(1)  # Shift to avoid leakage

all_data['quantity_momentum_4w'] = all_data.groupby(['customer_id', 'product_unit_variant_id']).apply(
    weighted_qty_momentum
).reset_index(level=[0,1], drop=True)

print(f"   ✓ Created: quantity_momentum_4w")
print(f"     Non-null: {all_data['quantity_momentum_4w'].notna().sum():,}")
print(f"     Mean: {all_data['quantity_momentum_4w'].mean():.4f}")

# Feature 7: Is regular customer (purchased 3+ times in last 4 weeks)
print("\n7. Creating: is_regular_customer_this_product...")

def is_regular(group):
    result = group['purchased_this_week'].rolling(window=4, min_periods=4).sum()
    return (result.shift(1) >= 3).astype(int)  # Shift to avoid leakage

all_data['is_regular_customer_this_product'] = all_data.groupby(['customer_id', 'product_unit_variant_id']).apply(
    is_regular
).reset_index(level=[0,1], drop=True)

print(f"   ✓ Created: is_regular_customer_this_product")
print(f"     Regular customers: {all_data['is_regular_customer_this_product'].sum():,} ({all_data['is_regular_customer_this_product'].mean()*100:.2f}%)")

# Feature 8: Average days between purchases (OPTIMIZED - vectorized approach)
print("\n8. Creating: avg_days_between_purchases...")

def calc_avg_days_between_vectorized(group):
    # Get purchase dates only
    purchase_mask = group['purchased_this_week'] == 1
    purchase_dates = group.loc[purchase_mask, 'week_start']
    
    if len(purchase_dates) >= 2:
        # Calculate differences between consecutive purchases
        diffs = purchase_dates.diff().dt.days
        avg_diff = diffs.expanding().mean()  # Expanding mean (each row knows only past)
        
        # Map back to all rows in group
        result = pd.Series(index=group.index, dtype=float)
        result[purchase_mask] = avg_diff
        result = result.fillna(method='ffill')  # Forward fill to non-purchase weeks
        
        return result
    else:
        return pd.Series(np.nan, index=group.index)

all_data['avg_days_between_purchases'] = all_data.groupby(['customer_id', 'product_unit_variant_id']).apply(
    calc_avg_days_between_vectorized
).reset_index(level=[0,1], drop=True)

print(f"   Created: avg_days_between_purchases")
print(f"     Non-null: {all_data['avg_days_between_purchases'].notna().sum():,}")
print(f"     Mean: {all_data['avg_days_between_purchases'].mean():.2f} days")

print("\n All momentum & regularity features complete (LEAK-FREE)!")
print(f"\nNew data shape: {all_data.shape}")
print(f"Total features: {all_data.shape[1]}")

gc.collect()

# ============================================================================
# CELL 31 (OPTIMIZED & LEAK-FREE): PRODUCT DIVERSITY & BASKET FEATURES
# ============================================================================

print("\n" + "="*80)
print("CREATING PRODUCT DIVERSITY & BASKET FEATURES (OPTIMIZED & LEAK-FREE)")
print("="*80)

# Feature 9: SKIPPED - Herfindahl index too computationally expensive
print("\n9. Skipping: customer_product_concentration (too slow for marginal benefit)")

# Feature 10: Percentage of customer spending - USING EXPANDING SUM
print("\n10. Creating: pct_customer_spending_this_product...")

# Create expanding cumulative sums (shifted to avoid leakage)
all_data['cumsum_spend_customer'] = all_data.groupby('customer_id')['spend_this_week'].cumsum().shift(1)
all_data['cumsum_spend_customer_product'] = all_data.groupby(['customer_id', 'product_unit_variant_id'])['spend_this_week'].cumsum().shift(1)

all_data['pct_customer_spending_this_product'] = np.where(
    all_data['cumsum_spend_customer'] > 0,
    (all_data['cumsum_spend_customer_product'] / all_data['cumsum_spend_customer']) * 100,
    0
)

# Drop temporary columns
all_data.drop(['cumsum_spend_customer', 'cumsum_spend_customer_product'], axis=1, inplace=True)

print(f"    Created: pct_customer_spending_this_product")
print(f"     Mean: {all_data['pct_customer_spending_this_product'].mean():.2f}%")

# Feature 11: Product rank in customer basket - USING EXPANDING SUM
print("\n11. Creating: product_rank_in_customer_basket...")

# Create expanding cumulative quantities (shifted to avoid leakage)
all_data['cumsum_qty_customer_product'] = all_data.groupby(['customer_id', 'product_unit_variant_id'])['qty_this_week'].cumsum().shift(1).fillna(0)

# Rank products by cumulative quantity within each customer-week
all_data['product_rank_in_customer_basket'] = all_data.groupby(['customer_id', 'week_start'])['cumsum_qty_customer_product'].rank(
    method='dense', ascending=False
).fillna(999)

# Drop temporary column
all_data.drop('cumsum_qty_customer_product', axis=1, inplace=True)

print(f"    Created: product_rank_in_customer_basket")
print(f"     Mean rank: {all_data['product_rank_in_customer_basket'].mean():.2f}")

# Feature 12: Customer average basket size - SIMPLIFIED VECTORIZED
print("\n12. Creating: customer_avg_basket_size...")

# Count unique products per customer per week
products_per_week = all_data[all_data['purchased_this_week'] == 1].groupby(['customer_id', 'week_start'])['product_unit_variant_id'].nunique().reset_index()
products_per_week.columns = ['customer_id', 'week_start', 'basket_size']

# Calculate expanding mean of basket size per customer
products_per_week = products_per_week.sort_values(['customer_id', 'week_start'])
products_per_week['avg_basket_size'] = products_per_week.groupby('customer_id')['basket_size'].expanding().mean().reset_index(level=0, drop=True)

# Shift to avoid leakage (use previous weeks' average)
products_per_week['avg_basket_size'] = products_per_week.groupby('customer_id')['avg_basket_size'].shift(1)

# Merge back to main data
all_data = all_data.merge(
    products_per_week[['customer_id', 'week_start', 'avg_basket_size']],
    on=['customer_id', 'week_start'],
    how='left'
)

# Forward fill within each customer (carry forward last known average)
all_data['customer_avg_basket_size'] = all_data.groupby('customer_id')['avg_basket_size'].fillna(method='ffill').fillna(1)

# Drop temporary column
all_data.drop('avg_basket_size', axis=1, inplace=True)

print(f"    Created: customer_avg_basket_size")
print(f"     Mean: {all_data['customer_avg_basket_size'].mean():.2f} products/week")

# Feature 13: Customer total quantity purchased (rolling 8 weeks)
print("\n13. Creating: customer_total_qty_8w...")

def rolling_customer_qty(group):
    return group['qty_this_week'].rolling(window=8, min_periods=1).sum().shift(1)

all_data['customer_total_qty_8w'] = all_data.groupby('customer_id').apply(
    rolling_customer_qty
).reset_index(level=0, drop=True)

print(f"    Created: customer_total_qty_8w")
print(f"     Mean: {all_data['customer_total_qty_8w'].mean():.2f}")

# Feature 14: Product popularity (total customers buying in last 4 weeks)
print("\n14. Creating: product_popularity_4w...")

def product_popularity(group):
    result = []
    for week in sorted(group['week_start'].unique()):
        last_4w = group[
            (group['week_start'] < week) & 
            (group['week_start'] >= week - pd.Timedelta(days=28))
        ]
        unique_customers = last_4w[last_4w['purchased_this_week'] == 1]['customer_id'].nunique()
        week_data = group[group['week_start'] == week].index
        result.extend([unique_customers] * len(week_data))
    
    return pd.Series(result, index=group.index)

all_data['product_popularity_4w'] = all_data.groupby('product_unit_variant_id').apply(
    product_popularity
).reset_index(level=0, drop=True)

print(f"    Created: product_popularity_4w")
print(f"     Mean: {all_data['product_popularity_4w'].mean():.2f} customers")

print("\n Product diversity & basket features complete (OPTIMIZED & LEAK-FREE)!")
print(f"\nNew data shape: {all_data.shape}")

gc.collect()

# ============================================================================
# CELL 32 (OPTIMIZED & LEAK-FREE): EXPONENTIAL DECAY & PRICE FEATURES
# ============================================================================

print("\n" + "="*80)
print("CREATING EXPONENTIAL DECAY & PRICE FEATURES (OPTIMIZED & LEAK-FREE)")
print("="*80)

# Feature 15: Exponentially weighted purchase rate - ALREADY CORRECT
print("\n15. Creating: exp_weighted_purchase_rate...")

def exp_weighted_purchase(group):
    result = group['purchased_this_week'].ewm(alpha=0.3, adjust=False).mean().shift(1)
    return result

all_data['exp_weighted_purchase_rate'] = all_data.groupby(['customer_id', 'product_unit_variant_id']).apply(
    exp_weighted_purchase
).reset_index(level=[0,1], drop=True)

print(f"    Created: exp_weighted_purchase_rate")
print(f"     Mean: {all_data['exp_weighted_purchase_rate'].mean():.4f}")

# Feature 16: Exponentially weighted average quantity - ALREADY CORRECT
print("\n16. Creating: exp_weighted_avg_qty...")

def exp_weighted_qty(group):
    result = group['qty_this_week'].ewm(alpha=0.3, adjust=False).mean().shift(1)
    return result

all_data['exp_weighted_avg_qty'] = all_data.groupby(['customer_id', 'product_unit_variant_id']).apply(
    exp_weighted_qty
).reset_index(level=[0,1], drop=True)

print(f"    Created: exp_weighted_avg_qty")
print(f"     Mean: {all_data['exp_weighted_avg_qty'].mean():.4f}")

# Feature 17: Price change indicator - ALREADY CORRECT
print("\n17. Creating: price_vs_avg...")

def price_deviation(group):
    avg_price = group['selling_price'].expanding().mean()
    current_price = group['selling_price']
    deviation = (current_price - avg_price) / (avg_price + 1)
    return deviation

all_data['price_vs_avg'] = all_data.groupby(['customer_id', 'product_unit_variant_id']).apply(
    price_deviation
).reset_index(level=[0,1], drop=True)

print(f"    Created: price_vs_avg")
print(f"     Mean: {all_data['price_vs_avg'].mean():.4f}")

# Feature 18: Customer vs market quantity ratio (OPTIMIZED - vectorized)
print("\n18. Creating: customer_vs_market_qty_ratio...")

# Calculate expanding average for each product (market baseline)
# Only for rows where qty > 0
market_avg = all_data[all_data['qty_this_week'] > 0].groupby('product_unit_variant_id')['qty_this_week'].expanding().mean().reset_index(level=0, drop=True).shift(1)

# Calculate expanding average for each customer-product combination
customer_avg = all_data[all_data['qty_this_week'] > 0].groupby(['customer_id', 'product_unit_variant_id'])['qty_this_week'].expanding().mean().reset_index(level=[0,1], drop=True).shift(1)

# Align indices
all_data.loc[all_data['qty_this_week'] > 0, 'market_avg_temp'] = market_avg
all_data.loc[all_data['qty_this_week'] > 0, 'customer_avg_temp'] = customer_avg

# Forward fill within groups
all_data['market_avg_temp'] = all_data.groupby('product_unit_variant_id')['market_avg_temp'].fillna(method='ffill')
all_data['customer_avg_temp'] = all_data.groupby(['customer_id', 'product_unit_variant_id'])['customer_avg_temp'].fillna(method='ffill')

# Calculate ratio
all_data['customer_vs_market_qty_ratio'] = np.where(
    (all_data['market_avg_temp'] > 0) & (all_data['customer_avg_temp'].notna()),
    all_data['customer_avg_temp'] / all_data['market_avg_temp'],
    1.0
)

# Drop temporary columns
all_data.drop(['market_avg_temp', 'customer_avg_temp'], axis=1, inplace=True)

print(f"    Created: customer_vs_market_qty_ratio")
print(f"     Mean: {all_data['customer_vs_market_qty_ratio'].mean():.4f}")

# Feature 19: Weeks since first purchase (OPTIMIZED - vectorized)
print("\n19. Creating: weeks_since_first_purchase...")

def calc_weeks_since_first_vectorized(group):
    # Find first purchase date in the group
    first_purchase_mask = group['purchased_this_week'] == 1
    
    if first_purchase_mask.any():
        first_purchase = group.loc[first_purchase_mask, 'week_start'].min()
        # Calculate weeks difference for all rows
        weeks_diff = (group['week_start'] - first_purchase).dt.days / 7
        return weeks_diff
    else:
        return pd.Series(0, index=group.index)

all_data['weeks_since_first_purchase'] = all_data.groupby(['customer_id', 'product_unit_variant_id'], group_keys=False).apply(
    calc_weeks_since_first_vectorized
)

print(f"   ✓ Created: weeks_since_first_purchase")
print(f"     Mean: {all_data['weeks_since_first_purchase'].mean():.2f} weeks")

# Feature 20: Purchase volatility - ALREADY CORRECT
print("\n20. Creating: purchase_volatility_8w...")

def calc_volatility(group):
    return group['purchased_this_week'].rolling(window=8, min_periods=4).std().shift(1)

all_data['purchase_volatility_8w'] = all_data.groupby(['customer_id', 'product_unit_variant_id']).apply(
    calc_volatility
).reset_index(level=[0,1], drop=True)

print(f"    Created: purchase_volatility_8w")
print(f"     Mean: {all_data['purchase_volatility_8w'].mean():.4f}")

print("\n All exponential decay & price features complete (OPTIMIZED & LEAK-FREE)!")
print(f"\nNew data shape: {all_data.shape}")
print(f"Total new features: {all_data.shape[1] - 21}")

gc.collect()

# ============================================================================
# CELL 33 (OPTIMIZED & LEAK-FREE): ADDITIONAL HIGH-VALUE FEATURES
# ============================================================================

print("\n" + "="*80)
print("CREATING ADDITIONAL HIGH-VALUE FEATURES (OPTIMIZED & LEAK-FREE)")
print("="*80)

# Feature 21: Average orders per week (rolling 4w)
print("\n21. Creating: avg_orders_per_week_4w...")

def rolling_avg_orders(group):
    return group['num_orders_week'].rolling(window=4, min_periods=2).mean().shift(1)

all_data['avg_orders_per_week_4w'] = all_data.groupby(['customer_id', 'product_unit_variant_id']).apply(
    rolling_avg_orders
).reset_index(level=[0,1], drop=True)

print(f"    Created: avg_orders_per_week_4w")
print(f"     Mean: {all_data['avg_orders_per_week_4w'].mean():.4f}")

# Feature 22: Order frequency momentum
print("\n22. Creating: order_frequency_momentum...")

def order_frequency_momentum(group):
    result = group['num_orders_week'].rolling(window=4, min_periods=3).apply(
        lambda x: np.average(x, weights=[1,2,3,4][-len(x):]) if len(x) >= 3 else np.nan,
        raw=True
    )
    return result.shift(1)

all_data['order_frequency_momentum'] = all_data.groupby(['customer_id', 'product_unit_variant_id']).apply(
    order_frequency_momentum
).reset_index(level=[0,1], drop=True)

print(f"    Created: order_frequency_momentum")
print(f"     Mean: {all_data['order_frequency_momentum'].mean():.4f}")

# Feature 23: Is bulk buyer (OPTIMIZED - vectorized)
print("\n23. Creating: is_bulk_buyer...")

# Calculate expanding mean of orders per week (only when purchased)
all_data['temp_orders_when_purchased'] = np.where(
    all_data['purchased_this_week'] == 1,
    all_data['num_orders_week'],
    np.nan
)

all_data['expanding_avg_orders'] = all_data.groupby(['customer_id', 'product_unit_variant_id'])['temp_orders_when_purchased'].expanding().mean().reset_index(level=[0,1], drop=True).shift(1)

all_data['is_bulk_buyer'] = (all_data['expanding_avg_orders'] > 1.5).astype(int).fillna(0)

# Drop temporary columns
all_data.drop(['temp_orders_when_purchased', 'expanding_avg_orders'], axis=1, inplace=True)

print(f"    Created: is_bulk_buyer")
print(f"     Bulk buyers: {all_data['is_bulk_buyer'].sum():,} ({all_data['is_bulk_buyer'].mean()*100:.2f}%)")

# Feature 24: Spend momentum (4 week weighted)
print("\n24. Creating: spend_momentum_4w...")

def spend_momentum(group):
    result = group['spend_this_week'].rolling(window=4, min_periods=3).apply(
        lambda x: np.average(x, weights=[1,2,3,4][-len(x):]) if len(x) >= 3 else np.nan,
        raw=True
    )
    return result.shift(1)

all_data['spend_momentum_4w'] = all_data.groupby(['customer_id', 'product_unit_variant_id']).apply(
    spend_momentum
).reset_index(level=[0,1], drop=True)

print(f"    Created: spend_momentum_4w")
print(f"     Mean: {all_data['spend_momentum_4w'].mean():.4f}")

# Feature 25: Spend volatility (std of spend in last 8 weeks)
print("\n25. Creating: spend_volatility_8w...")

def spend_volatility(group):
    return group['spend_this_week'].rolling(window=8, min_periods=4).std().shift(1)

all_data['spend_volatility_8w'] = all_data.groupby(['customer_id', 'product_unit_variant_id']).apply(
    spend_volatility
).reset_index(level=[0,1], drop=True)

print(f"    Created: spend_volatility_8w")
print(f"     Mean: {all_data['spend_volatility_8w'].mean():.4f}")

# Feature 26: Std of days between purchases (OPTIMIZED - vectorized)
print("\n26. Creating: std_days_between_purchases...")

def calc_std_days_between_vectorized(group):
    # Get purchase dates only
    purchase_mask = group['purchased_this_week'] == 1
    purchase_dates = group.loc[purchase_mask, 'week_start']
    
    if len(purchase_dates) >= 3:
        # Calculate differences between consecutive purchases
        diffs = purchase_dates.diff().dt.days
        std_diff = diffs.expanding().std()  # Expanding std (each row knows only past)
        
        # Map back to all rows in group
        result = pd.Series(index=group.index, dtype=float)
        result[purchase_mask] = std_diff
        result = result.fillna(method='ffill')  # Forward fill to non-purchase weeks
        
        return result
    else:
        return pd.Series(np.nan, index=group.index)

all_data['std_days_between_purchases'] = all_data.groupby(['customer_id', 'product_unit_variant_id'], group_keys=False).apply(
    calc_std_days_between_vectorized
)

print(f"    Created: std_days_between_purchases")
print(f"     Mean: {all_data['std_days_between_purchases'].mean():.2f} days")

# Feature 27: Purchase interval stability (coefficient of variation)
print("\n27. Creating: purchase_interval_stability...")

# CV = std / mean (lower = more stable)
all_data['purchase_interval_stability'] = np.where(
    all_data['avg_days_between_purchases'] > 0,
    all_data['std_days_between_purchases'] / all_data['avg_days_between_purchases'],
    np.nan
)

print(f"    Created: purchase_interval_stability")
print(f"     Mean: {all_data['purchase_interval_stability'].mean():.4f}")
print(f"     (Lower = more regular purchasing pattern)")

# Feature 28: Product growth rate (OPTIMIZED - simplified)
print("\n28. Creating: product_growth_rate_4w...")

# Calculate rolling count of unique customers in last 4 weeks vs previous 4 weeks
def product_growth_simplified(group):
    # Count unique customers in rolling 4-week windows
    result = []
    for week in sorted(group['week_start'].unique()):
        recent_4w = group[
            (group['week_start'] < week) &
            (group['week_start'] >= week - pd.Timedelta(days=28)) &
            (group['purchased_this_week'] == 1)
        ]['customer_id'].nunique()
        
        previous_4w = group[
            (group['week_start'] < week - pd.Timedelta(days=28)) &
            (group['week_start'] >= week - pd.Timedelta(days=56)) &
            (group['purchased_this_week'] == 1)
        ]['customer_id'].nunique()
        
        if previous_4w > 0:
            growth = (recent_4w - previous_4w) / previous_4w
        else:
            growth = 0
        
        week_data = group[group['week_start'] == week].index
        result.extend([growth] * len(week_data))
    
    return pd.Series(result, index=group.index)

all_data['product_growth_rate_4w'] = all_data.groupby('product_unit_variant_id').apply(
    product_growth_simplified
).reset_index(level=0, drop=True)

print(f"    Created: product_growth_rate_4w")
print(f"     Mean: {all_data['product_growth_rate_4w'].mean():.4f}")

# Feature 29: Product reorder rate (% of customers who bought again in 8w)
print("\n29. Creating: product_reorder_rate_8w...")

def product_reorder_rate(group):
    result = []
    for week in sorted(group['week_start'].unique()):
        last_8w = group[
            (group['week_start'] < week) &
            (group['week_start'] >= week - pd.Timedelta(days=56)) &
            (group['purchased_this_week'] == 1)
        ]
        
        if len(last_8w) > 0:
            customer_purchase_counts = last_8w.groupby('customer_id').size()
            customers_who_reordered = (customer_purchase_counts > 1).sum()
            total_customers = len(customer_purchase_counts)
            
            reorder_rate = customers_who_reordered / total_customers if total_customers > 0 else 0
        else:
            reorder_rate = 0
        
        week_data = group[group['week_start'] == week].index
        result.extend([reorder_rate] * len(week_data))
    
    return pd.Series(result, index=group.index)

all_data['product_reorder_rate_8w'] = all_data.groupby('product_unit_variant_id').apply(
    product_reorder_rate
).reset_index(level=0, drop=True)

print(f"    Created: product_reorder_rate_8w")
print(f"     Mean: {all_data['product_reorder_rate_8w'].mean():.4f}")

gc.collect()

# Feature 30: Customer category diversity (FASTER ALTERNATIVE)
print("\n30. Creating: customer_category_diversity...")

# Create a flag for first occurrence of each product per customer
all_data['is_new_product'] = ~all_data.duplicated(subset=['customer_id', 'product_id'])

# Cumulative sum of new products per customer (shifted to avoid leakage)
all_data['customer_category_diversity'] = all_data.groupby('customer_id')['is_new_product'].cumsum().shift(1).fillna(0).astype(int)

# Drop temporary column
all_data.drop('is_new_product', axis=1, inplace=True)

print(f"    Created: customer_category_diversity")
print(f"     Mean: {all_data['customer_category_diversity'].mean():.2f} categories")

# ============================================================================
# CELL 34: PREPARE ENHANCED DATASET FOR MODELING
# ============================================================================

print("\n" + "="*80)
print("PREPARING ENHANCED DATASET FOR MODELING")
print("="*80)

# Split back into train and test
train_enhanced = all_data[all_data['is_train'] == 1].copy()
test_enhanced = all_data[all_data['is_train'] == 0].copy()

print(f"\nTrain enhanced shape: {train_enhanced.shape}")
print(f"Test enhanced shape: {test_enhanced.shape}")

# List of ALL new features we created
new_features = [
    # From Cell 29 (Recency features)
    'days_since_last_purchase_any',
    'days_since_last_purchase_this_product',
    
    # From Cell 30 (Momentum & Regularity)
    'consecutive_weeks_purchased',
    'consecutive_weeks_not_purchased',
    'purchase_momentum_4w',
    'quantity_momentum_4w',
    'is_regular_customer_this_product',
    'avg_days_between_purchases',
    
    # From Cell 31 (Product Diversity & Basket) - skipped customer_product_concentration
    'pct_customer_spending_this_product',
    'product_rank_in_customer_basket',
    'customer_avg_basket_size',
    'customer_total_qty_8w',
    'product_popularity_4w',
    
    # From Cell 32 (Exponential Decay & Price)
    'exp_weighted_purchase_rate',
    'exp_weighted_avg_qty',
    'price_vs_avg',
    'customer_vs_market_qty_ratio',
    'weeks_since_first_purchase',
    'purchase_volatility_8w',
    
    # From Cell 33 (Additional High-Value Features)
    'avg_orders_per_week_4w',
    'order_frequency_momentum',
    'is_bulk_buyer',
    'spend_momentum_4w',
    'spend_volatility_8w',
    'std_days_between_purchases',
    'purchase_interval_stability',
    'product_growth_rate_4w',
    'product_reorder_rate_8w',
    'customer_category_diversity'
]

print(f"\n Created {len(new_features)} new features:")
for i, feat in enumerate(new_features, 1):
    print(f"  {i:2d}. {feat}")

# Check which features actually exist in the data
existing_features = [f for f in new_features if f in train_enhanced.columns]
missing_features = [f for f in new_features if f not in train_enhanced.columns]

if missing_features:
    print(f"\n Warning: {len(missing_features)} features not found:")
    for feat in missing_features:
        print(f"  - {feat}")

print(f"\n {len(existing_features)} features exist in data")

# Check missing values in new features
print("\n" + "="*60)
print("MISSING VALUE ANALYSIS FOR NEW FEATURES")
print("="*60)

missing_stats = []
for feat in existing_features:
    missing_count = train_enhanced[feat].isna().sum()
    missing_pct = (missing_count / len(train_enhanced)) * 100
    missing_stats.append({
        'Feature': feat,
        'Missing_Count': missing_count,
        'Missing_Pct': missing_pct
    })

missing_df = pd.DataFrame(missing_stats).sort_values('Missing_Pct', ascending=False)
print("\nTop 15 features by missing percentage:")
print(missing_df.head(15).to_string(index=False))

# Fill missing values with appropriate defaults
print("\n" + "="*60)
print("FILLING MISSING VALUES")
print("="*60)

fill_values = {
    # Recency features
    'days_since_last_purchase_any': 999,
    'days_since_last_purchase_this_product': 999,
    
    # Momentum features
    'consecutive_weeks_purchased': 0,
    'consecutive_weeks_not_purchased': 0,
    'purchase_momentum_4w': 0,
    'quantity_momentum_4w': 0,
    'is_regular_customer_this_product': 0,
    'avg_days_between_purchases': 30,
    'std_days_between_purchases': 15,
    'purchase_interval_stability': 0.5,
    
    # Basket features
    'pct_customer_spending_this_product': 0,
    'product_rank_in_customer_basket': 999,
    'customer_avg_basket_size': 1,
    'customer_total_qty_8w': 0,
    'product_popularity_4w': 1,
    
    # Exponential & Price features
    'exp_weighted_purchase_rate': 0,
    'exp_weighted_avg_qty': 0,
    'price_vs_avg': 0,
    'customer_vs_market_qty_ratio': 1.0,
    'weeks_since_first_purchase': 0,
    'purchase_volatility_8w': 0,
    
    # Additional features
    'avg_orders_per_week_4w': 0,
    'order_frequency_momentum': 0,
    'is_bulk_buyer': 0,
    'spend_momentum_4w': 0,
    'spend_volatility_8w': 0,
    'product_growth_rate_4w': 0,
    'product_reorder_rate_8w': 0.5,
    'customer_category_diversity': 1
}

filled_count = 0
for feat, fill_val in fill_values.items():
    if feat in train_enhanced.columns:
        before_train = train_enhanced[feat].isna().sum()
        before_test = test_enhanced[feat].isna().sum()
        
        train_enhanced[feat].fillna(fill_val, inplace=True)
        test_enhanced[feat].fillna(fill_val, inplace=True)
        
        if before_train > 0 or before_test > 0:
            print(f"   Filled {feat} with {fill_val} (Train: {before_train:,}, Test: {before_test:,})")
            filled_count += 1

print(f"\n Filled {filled_count} features with missing values!")

# Save enhanced datasets
print("\n" + "="*60)
print("SAVING ENHANCED DATASETS")
print("="*60)

train_enhanced.to_csv('train_enhanced_features_v2.csv', index=False)
test_enhanced.to_csv('test_enhanced_features_v2.csv', index=False)

print(f"\n Saved: train_enhanced_features_v2.csv ({train_enhanced.shape})")
print(f"Saved: test_enhanced_features_v2.csv ({test_enhanced.shape})")


# ============================================================================
# CELL 35: MERGE NEW FEATURES WITH PROCESSED DATA
# ============================================================================

print("\n" + "="*80)
print("MERGING NEW FEATURES WITH PROCESSED DATA")
print("="*80)

# Load the original processed data
print("\nLoading original processed data...")
train_processed = pd.read_csv('/kaggle/input/processed-dataset/train_processed.csv')
test_processed = pd.read_csv('/kaggle/input/processed-dataset/test_processed.csv')

print(f" Train processed shape: {train_processed.shape}")
print(f" Test processed shape: {test_processed.shape}")

# Load the enhanced features we just created
print("\nLoading enhanced features...")
train_enhanced = pd.read_csv('train_enhanced_features_v2.csv')
test_enhanced = pd.read_csv('test_enhanced_features_v2.csv')

print(f" Train enhanced shape: {train_enhanced.shape}")
print(f" Test enhanced shape: {test_enhanced.shape}")

# Convert dates for merging
train_processed['week_start'] = pd.to_datetime(train_processed['week_start'])
test_processed['week_start'] = pd.to_datetime(test_processed['week_start'])
train_enhanced['week_start'] = pd.to_datetime(train_enhanced['week_start'])
test_enhanced['week_start'] = pd.to_datetime(test_enhanced['week_start'])

# Merge keys
merge_keys = ['customer_id', 'product_unit_variant_id', 'week_start']

print(f"\nMerging on: {merge_keys}")

# List of new features to add (from enhanced data)
new_features = [
    'days_since_last_purchase_any',
    'days_since_last_purchase_this_product',
    'consecutive_weeks_purchased',
    'consecutive_weeks_not_purchased',
    'purchase_momentum_4w',
    'quantity_momentum_4w',
    'is_regular_customer_this_product',
    'avg_days_between_purchases',
    'pct_customer_spending_this_product',
    'product_rank_in_customer_basket',
    'customer_avg_basket_size',
    'customer_total_qty_8w',
    'product_popularity_4w',
    'exp_weighted_purchase_rate',
    'exp_weighted_avg_qty',
    'price_vs_avg',
    'customer_vs_market_qty_ratio',
    'weeks_since_first_purchase',
    'purchase_volatility_8w',
    'avg_orders_per_week_4w',
    'order_frequency_momentum',
    'is_bulk_buyer',
    'spend_momentum_4w',
    'spend_volatility_8w',
    'std_days_between_purchases',
    'purchase_interval_stability',
    'product_growth_rate_4w',
    'product_reorder_rate_8w',
    'customer_category_diversity'
]

# Filter to only features that exist in enhanced data
features_to_add = [f for f in new_features if f in train_enhanced.columns]

print(f"\nAdding {len(features_to_add)} new features to processed data...")

# Merge train
print("\nMerging train data...")
train_final = train_processed.merge(
    train_enhanced[merge_keys + features_to_add],
    on=merge_keys,
    how='left'
)

print(f" Train final shape: {train_final.shape}")
print(f"  Original: {train_processed.shape[1]} columns")
print(f"  Added: {len(features_to_add)} new features")
print(f"  Total: {train_final.shape[1]} columns")

# Merge test
print("\nMerging test data...")
test_final = test_processed.merge(
    test_enhanced[merge_keys + features_to_add],
    on=merge_keys,
    how='left'
)

print(f" Test final shape: {test_final.shape}")
print(f"  Original: {test_processed.shape[1]} columns")
print(f"  Added: {len(features_to_add)} new features")
print(f"  Total: {test_final.shape[1]} columns")

# Check for missing values after merge
print("\n" + "="*60)
print("CHECKING MERGE QUALITY")
print("="*60)

print("\nNew features missing values in train:")
missing_count = 0
for feat in features_to_add[:15]:  # Show first 15
    missing = train_final[feat].isna().sum()
    if missing > 0:
        print(f"  {feat}: {missing:,} ({missing/len(train_final)*100:.2f}%)")
        missing_count += 1

if missing_count == 0:
    print("   No missing values in first 15 features!")
else:
    print(f"\n  Found {missing_count} features with missing values")

# Save final merged datasets
print("\n" + "="*60)
print("SAVING FINAL MERGED DATASETS")
print("="*60)

train_filename = 'train_final_with_new_features.csv'
test_filename = 'test_final_with_new_features.csv'

train_final.to_csv(train_filename, index=False)
test_final.to_csv(test_filename, index=False)

