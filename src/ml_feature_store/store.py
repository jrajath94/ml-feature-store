import pandas as pd
from datetime import datetime
from typing import Dict, List

class FeatureStore:
    def __init__(self):
        self.features: Dict[str, pd.DataFrame] = {}
    
    def register(self, name: str, df: pd.DataFrame) -> None:
        """Register feature table."""
        self.features[name] = df.copy()
    
    def get(self, name: str, entity_ids: List, timestamp: datetime) -> pd.DataFrame:
        """Get features at point-in-time."""
        df = self.features[name]
        if 'timestamp' in df.columns:
            df = df[df['timestamp'] <= timestamp]
        return df[df['entity_id'].isin(entity_ids)]
