import polars as pl
from deltalake import write_deltalake
import os

def create_aggregates():
    """Создание аналитических витрин"""
    
    silver_path = "data/silver/flights"
    gold_path = "data/gold"
    
    print("Reading silver layer...")
    df = pl.scan_delta(silver_path)
    
    # Агрегат 1: Средние задержки по аэропорту назначения и часу
    print("Creating aggregate: delays_by_dest_hour...")
    agg1 = (df
        .group_by(["DEST", "hour"])
        .agg([
            pl.col("ARR_DELAY").mean().alias("avg_arr_delay"),
            pl.col("ARR_DELAY").count().alias("flight_count")
        ])
        .sort("DEST", "hour")
        .collect()
    )
    
    # Агрегат 2: По авиакомпании и сезону
    print("Creating aggregate: delays_by_carrier_season...")
    agg2 = (df
        .group_by(["OP_CARRIER", "season"])
        .agg([
            pl.col("ARR_DELAY").mean().alias("avg_arr_delay"),
            pl.col("ARR_DELAY").count().alias("flight_count")
        ])
        .sort("OP_CARRIER", "season")
        .collect()
    )
    
    # Записываем
    os.makedirs(gold_path, exist_ok=True)
    write_deltalake(f"{gold_path}/delays_by_dest_hour", agg1.to_pandas(), mode="overwrite")
    write_deltalake(f"{gold_path}/delays_by_carrier_season", agg2.to_pandas(), mode="overwrite")
    
    print(f"✅ Aggregates created at {gold_path}")

if __name__ == "__main__":
    create_aggregates()
