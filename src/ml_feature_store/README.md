# ml_feature_store Module

Point-in-time correct feature retrieval.

## Core Concept

Traditional feature stores query current state. This queries historical state:
- "What were this user's features on Jan 15, 2024?"
- Returns features as they existed then (no future data)

## Implementation

- **store.py**: Main FeatureStore API
  - register(name, df): Associate DataFrame with table name
  - get(table, entity_ids, timestamp): Retrieve historical features
  - Uses pandas merge with asof matching for point-in-time lookups

- **validation.py**: Schema enforcement
  - Checks timestamp column exists
  - Validates entity_id uniqueness within timestamps
  - Handles missing entities gracefully

## Correctness Guarantee

The asof merge ensures: for each entity at timestamp T, return the most recent row <= T. This prevents lookahead (returning future data).
