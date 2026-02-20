# Interview Prep: ml-feature-store

## Elevator Pitch

Lightweight feature store with point-in-time correctness for small-to-mid ML teams. Register feature tables (DataFrames with timestamps), query historical features at specific dates. Prevents training/serving skew and data leakage. Alternative to Tecton/Feast for teams that don't need enterprise infrastructure.

## Key Insight

Point-in-time correctness is the real value prop, not the database. Most teams don't even know they have training/serving skew because they never check historical consistency.

## Core Pattern

```python
store.register("user_features", df)  # Has timestamp column
features = store.get("user_features", entity_ids=[1,2,3], 
                     timestamp=datetime(2024, 1, 15))
# Returns features as they existed on Jan 15, no future data
```

## Interview Questions

- "Why point-in-time?" → Prevents look-ahead bias in backtesting
- "vs Tecton?" → Tecton is $$$. This is DIY-friendly.
- "Scaling?" → Use DuckDB or Parquet for larger datasets
- "Backfill?" → Historical data can be pre-computed and cached

