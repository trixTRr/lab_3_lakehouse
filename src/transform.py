import polars as pl
from deltalake import write_deltalake, DeltaTable
import os

def transform_silver():
    # Очистка, трансформация и запись в silver
    
    bronze_path = "data/bronze/flights"
    silver_path = "data/silver/flights"
    
    if not os.path.exists(bronze_path):
        print("Bronze layer not found. Run ingest.py first.")
        return
    
    print("Reading bronze layer...")
    df = pl.read_delta(bronze_path)
    
    print(f"Initial rows: {df.height:,}")
    
    # Фильтрация
    df = df.filter(
        (pl.col("Cancelled") == 0) &
        (pl.col("ArrDelay").is_not_null()) &
        (pl.col("ArrDelay").abs() < 720)
    )
    
    print(f"After filtering: {df.height:,} rows")
    
    # Создаём новые признаки
    df = df.with_columns([
        pl.col("FlightDate").alias("flight_date"),
        pl.col("Year").alias("year"),
        pl.col("Month").alias("month"),
        (pl.col("CRSDepTime") // 100).alias("hour"),
        pl.col("DayOfWeek").alias("day_of_week"),
        pl.when(pl.col("Month").is_between(3, 5)).then(pl.lit(0))
         .when(pl.col("Month").is_between(6, 8)).then(pl.lit(1))
         .when(pl.col("Month").is_between(9, 11)).then(pl.lit(2))
         .otherwise(pl.lit(3)).alias("season"),
        (pl.col("Origin") + "_" + pl.col("Dest")).alias("route"),
    ])
    
    # Выбираем нужные колонки
    silver_df = df.select([
        "flight_date",
        "year",
        "month",
        "hour",
        "day_of_week",
        "season",
        pl.col("Marketing_Airline_Network").alias("OP_CARRIER"),
        pl.col("Origin").alias("ORIGIN"),
        pl.col("Dest").alias("DEST"),
        "route",
        pl.col("Distance").alias("DISTANCE"),
        pl.col("DepDelay").alias("DEP_DELAY"),
        pl.col("ArrDelay").alias("ARR_DELAY"),
    ])
    
    print(f"\nFinal shape: {silver_df.shape}")
    print(f"Years: {sorted(silver_df['year'].unique().to_list())}")
    print(f"Months: {sorted(silver_df['month'].unique().to_list())}")
    
    # Удаляем старую таблицу, если есть
    if os.path.exists(silver_path):
        import shutil
        shutil.rmtree(silver_path)
        print(f"Removed old silver table")
    
    # Записываем в Delta
    os.makedirs(silver_path, exist_ok=True)
    write_deltalake(silver_path, silver_df.to_pandas(), mode="overwrite")
    
    # Оптимизация
    print("Optimizing...")
    dt = DeltaTable(silver_path)
    dt.optimize.compact()
    dt.optimize.z_order(["year", "month"])
    
    print(f"\nSilver layer created at {silver_path}")
    print(f"   Total flights: {silver_df.height:,}")
    print(f"   Delta version: {dt.version()}")

if __name__ == "__main__":
    transform_silver()