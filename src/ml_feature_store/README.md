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
  - Temporal filtering: returns only rows where timestamp <= query_timestamp
  - Latest-per-entity: groups by entity_id and selects the most recent row within the time window

## Correctness Guarantee

Temporal filtering and latest-per-entity selection ensure: for each entity at timestamp T, return the most recent row where timestamp <= T. This prevents lookahead (returning future data).
