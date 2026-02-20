# ml-feature-store

Lightweight feature store with point-in-time correctness for machine learning systems.

## Overview

Feature stores solve a critical gap in ML systems: consistent feature engineering across training and serving. Teams either build features twice (training pipeline and serving pipeline) creating skew, or use expensive commercial platforms (Tecton, Feast) requiring infrastructure overhead.

This library provides essential feature store functionality—point-in-time correct feature retrieval—without the operational complexity. Register feature tables with timestamps, query historical features at specific dates, prevent data leakage.

## Problem Statement

Most ML teams face feature consistency challenges:

1. **Training/serving skew**: Features computed differently in training vs production. Model trained on engineered features, but serving computes features differently. Performance degrades.

2. **Data leakage**: Using future data in training. Example: training on 2024 customer data using 2025 feature values. Model learns patterns that don't exist, fails in production.

3. **Feature duplication**: Same features computed in multiple pipelines. If logic changes, all pipelines must be updated or skew occurs.

4. **No reproducibility**: Can't recreate historical feature sets for model debugging or retraining.

Expensive commercial solutions require Kubernetes, complex orchestration, substantial infrastructure investment.

## Solution

Minimal, lightweight feature store:
- Register feature DataFrames with timestamp columns
- Query historical features at specific points in time
- Prevents look-ahead bias in backtesting
- No infrastructure required (works locally or on shared storage)

## Architecture

```
Feature Registration:
[customer_features.csv (with timestamp column)]
    ↓
store.register("customer_features", df)

Point-in-time Query:
store.get("customer_features", 
          entity_ids=[1, 2, 3],
          timestamp=datetime(2024, 1, 15))
    ↓
Returns features as they existed on Jan 15
(no future data)
```

## Installation

```bash
pip install ml-feature-store
```

## Usage

### Basic Feature Store

```python
import pandas as pd
from datetime import datetime
from ml_feature_store import FeatureStore

# Create feature store
store = FeatureStore()

# Register feature table with timestamp
customer_features = pd.DataFrame({
    'entity_id': [1, 2, 3, 1, 2, 3],
    'timestamp': pd.to_datetime([
        '2024-01-01', '2024-01-01', '2024-01-01',
        '2024-02-01', '2024-02-01', '2024-02-01'
    ]),
    'recency_days': [5, 10, 15, 3, 8, 12],
    'frequency': [10, 25, 5, 12, 28, 6]
})

store.register('customer_features', customer_features)

# Query historical features (point-in-time correct)
features_jan_15 = store.get('customer_features',
                            entity_ids=[1, 2],
                            timestamp=datetime(2024, 1, 15))

# Returns:
# entity_id | recency_days | frequency
# 1         | 5            | 10
# 2         | 10           | 25

# Query at later date
features_feb_15 = store.get('customer_features',
                            entity_ids=[1, 2],
                            timestamp=datetime(2024, 2, 15))

# Returns (updated values):
# entity_id | recency_days | frequency
# 1         | 3            | 12
# 2         | 8            | 28
```

### Backtesting with Point-in-time Correctness

```python
# Critical for realistic backtesting: use historical feature values
for date in pd.date_range('2024-01-01', '2024-12-01', freq='MS'):
    # Get features as they existed on this date
    features = store.get('customer_features',
                         entity_ids=model_portfolio,
                         timestamp=date)
    
    # Train/score model with historical features
    predictions = model.predict(features)
    
    # No data leakage: model can't see future feature values
```

## Design Decisions

| Decision | Rationale | Trade-off |
|----------|-----------|-----------|
| Timestamp-based retrieval | Prevents look-ahead bias, matches reality | Requires timestamp columns in data |
| DataFrame storage | Works with pandas, no additional dependencies | Limited to Python ML workflows |
| Filter on timestamp | Simple, interpretable behavior | Doesn't handle late-arriving features |

## Performance Characteristics

Query latency (varies with feature table size):

| Feature Table Size | Query Time (10 entities) | Notes |
|-------------------|-------------------------|-------|
| 100K rows | <1ms | In-memory DataFrame |
| 1M rows | 5-10ms | Small enough for memory |
| 10M rows | 100-500ms | Consider partitioning |
| 100M+ rows | Use DuckDB or Parquet | Not recommended for pandas |

For large feature tables, partition by date and load relevant partitions only.

## Real-World Applications

**Credit Risk Scoring**: Use historical customer behavior features at loan origination date. Prevents future-looking predictions.

**Recommendation Systems**: Query user preference features at request time, not with future data.

**Fraud Detection**: Use historical transaction patterns at transaction time for scoring.

**Churn Prediction**: Features computed at prediction date, not with future customer outcomes.

## Limitations

- Pandas-based (scales to ~100M rows)
- Single-machine (no distributed query)
- No real-time feature computation (batch only)
- No schema validation or lineage tracking

## Scaling to Larger Feature Sets

For larger deployments, use DuckDB or Parquet backend:

```python
# With DuckDB
import duckdb
conn = duckdb.connect(':memory:')
conn.execute('CREATE TABLE features AS SELECT * FROM read_parquet(...)')
```

## Future Enhancements

- DuckDB backend for larger datasets
- Parquet file support
- Incremental feature updates
- Feature lineage tracking
- Real-time feature serving

## Contributing

Contributions welcome for backends (DuckDB, Spark) and time-series features.

## License

MIT License.

## References

- Point-in-time correctness in feature stores (Tecton, Feast documentation)
- Data leakage prevention (Kaufman et al.)
