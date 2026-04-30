import polars as pl
from deltalake import write_deltalake
import os

def create_feature_table():
    """Создание feature table для ML"""
    
    silver_path = "data/silver/flights"
    gold_path = "data/gold/feature_table"
    
    print("Reading silver layer...")
    df = pl.scan_delta(silver_path)
    
    print("Creating features...")
    features = (df
        .select([
            "ARR_DELAY",  # target
            "hour", "day_of_week", "season",
            "OP_CARRIER", "ORIGIN", "DEST", "DISTANCE"
        ])
        .collect()
    )
    
    print(f"Feature table shape: {features.shape}")
    
    # Записываем
    os.makedirs(gold_path, exist_ok=True)
    write_deltalake(gold_path, features.to_pandas(), mode="overwrite")
    
    print(f"✅ Feature table created at {gold_path}")

if __name__ == "__main__":
    create_feature_table()
