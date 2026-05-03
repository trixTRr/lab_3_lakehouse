import polars as pl
from deltalake import write_deltalake, DeltaTable
import os

def transform_silver():
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
    
    # Создаём признаки
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
    
    # Выбираем колонки
    silver_df = df.select([
        pl.col("FlightDate").alias("flight_date"),
        pl.col("Year").alias("year"),
        pl.col("Month").alias("month"),
        (pl.col("CRSDepTime") // 100).alias("hour"),
        pl.col("DayOfWeek").alias("day_of_week"),
        pl.col("season").alias("season"),
        pl.col("Marketing_Airline_Network").alias("OP_CARRIER"),
        pl.col("Origin").alias("ORIGIN"),
        pl.col("Dest").alias("DEST"),
        pl.col("route").alias("route"),
        pl.col("Distance").alias("DISTANCE"),
        pl.col("DepDelay").alias("DEP_DELAY"),
        pl.col("ArrDelay").alias("ARR_DELAY"),
    ])
    
    # Приводим flight_date к строковому типу для единообразия
    silver_df = silver_df.with_columns(
        pl.col("flight_date").cast(pl.Utf8).alias("flight_date")
    )
    
    # Создаём уникальный идентификатор
    silver_df = silver_df.with_columns(
        (pl.col("flight_date") + "_" + 
         pl.col("OP_CARRIER") + "_" + 
         pl.col("ORIGIN") + "_" + 
         pl.col("DEST")).alias("flight_id")
    )
    
    # Нормализация
    numeric_features = ["hour", "day_of_week", "DISTANCE", "DEP_DELAY"]
    for col in numeric_features:
        if col in silver_df.columns:
            col_min = silver_df[col].min()
            col_max = silver_df[col].max()
            if col_max - col_min > 0:
                silver_df = silver_df.with_columns(
                    ((pl.col(col) - col_min) / (col_max - col_min)).alias(f"{col}_norm")
                )
            else:
                silver_df = silver_df.with_columns(
                    pl.lit(0.5).alias(f"{col}_norm")
                )
    
    # Удаляем дубликаты по flight_id
    silver_df = silver_df.unique(subset=["flight_id"], keep="last")
    
    print(f"Unique flights: {silver_df.height:,}")
    
    # Определяем финальный набор колонок (единая схема)
    final_columns = [
        "flight_id", "flight_date", "year", "month", "hour", "day_of_week",
        "season", "OP_CARRIER", "ORIGIN", "DEST", "route", "DISTANCE",
        "DEP_DELAY", "ARR_DELAY"
    ]
    
    # Добавляем нормализованные колонки, если они есть
    norm_cols = [f"{col}_norm" for col in numeric_features]
    final_columns.extend([col for col in norm_cols if col in silver_df.columns])
    
    # Убеждаемся, что все колонки есть
    for col in final_columns:
        if col not in silver_df.columns:
            silver_df = silver_df.with_columns(pl.lit(None).alias(col))
    
    # Приводим к финальной схеме
    silver_df = silver_df.select(final_columns)
    
    # Создаём директорию
    os.makedirs(silver_path, exist_ok=True)
    
    # Проверяем существование таблицы
    if os.path.exists(f"{silver_path}/_delta_log"):
        print("\nUpdating existing silver table...")
        
        # Загружаем существующую таблицу
        existing_df = pl.read_delta(silver_path)
        
        print(f"Existing rows: {existing_df.height:,}")
        print(f"Existing columns: {existing_df.columns[:5]}...")
        
        # Приводим существующую таблицу к той же схеме
        # Конвертируем flight_date в строку, если нужно
        if "flight_date" in existing_df.columns:
            if existing_df["flight_date"].dtype != pl.Utf8:
                existing_df = existing_df.with_columns(
                    pl.col("flight_date").cast(pl.Utf8).alias("flight_date")
                )
        
        # Добавляем недостающие колонки
        for col in final_columns:
            if col not in existing_df.columns:
                existing_df = existing_df.with_columns(pl.lit(None).alias(col))
        
        # Выбираем только нужные колонки в правильном порядке
        existing_df = existing_df.select(final_columns)
        
        print(f"Existing rows after schema alignment: {existing_df.height:,}")
        print(f"New rows: {silver_df.height:,}")
        
        # Находим ID для обновления
        existing_ids = set(existing_df['flight_id'].unique().to_list())
        new_ids = set(silver_df['flight_id'].unique().to_list())
        
        update_ids = existing_ids.intersection(new_ids)
        insert_ids = new_ids - existing_ids
        
        print(f"To update: {len(update_ids):,} records")
        print(f"To insert: {len(insert_ids):,} records")
        
        # Объединяем данные
        if len(update_ids) > 0:
            # Удаляем обновляемые записи из существующей таблицы
            keep_df = existing_df.filter(~pl.col("flight_id").is_in(list(update_ids)))
            # Берем новые версии обновляемых записей
            update_records = silver_df.filter(pl.col("flight_id").is_in(list(update_ids)))
            
            if len(insert_ids) > 0:
                insert_records = silver_df.filter(pl.col("flight_id").is_in(list(insert_ids)))
                final_df = pl.concat([keep_df, update_records, insert_records])
            else:
                final_df = pl.concat([keep_df, update_records])
        else:
            # Только новые записи
            final_df = pl.concat([existing_df, silver_df])
        
        # Удаляем возможные дубликаты
        final_df = final_df.unique(subset=["flight_id"], keep="last")
        
        print(f"Final rows after merge: {final_df.height:,}")
        
        # Перезаписываем таблицу
        import shutil
        shutil.rmtree(silver_path)
        write_deltalake(silver_path, final_df.to_pandas(), mode="overwrite")
        
        # Оптимизация
        dt = DeltaTable(silver_path)
        dt.optimize.compact()
        
        print(f"\nSilver layer updated!")
        print(f"   Total rows: {final_df.height:,}")
        print(f"   Delta version: {dt.version()}")
        
    else:
        print("\nCreating new silver table...")
        write_deltalake(silver_path, silver_df.to_pandas(), mode="overwrite")
        
        # Оптимизация
        dt = DeltaTable(silver_path)
        dt.optimize.compact()
        dt.optimize.z_order(["year", "month"])
        
        print(f"\nSilver layer created!")
        print(f"   Total rows: {silver_df.height:,}")
        print(f"   Delta version: {dt.version()}")
    
    # Выводим статистику
    final_check = pl.read_delta(silver_path)
    print(f"\nFinal statistics:")
    print(f"   Total unique flights: {final_check['flight_id'].unique().len():,}")
    print(f"   Total rows: {final_check.height:,}")
    print(f"   Total columns: {final_check.width:,}")
    
    if final_check['flight_id'].unique().len() == final_check.height:
        print(f"   No duplicates detected")
    else:
        print(f"   WARNING: Duplicates still present!")
        
        # Показываем пример дубликатов
        duplicates = final_check.group_by("flight_id").agg(pl.count().alias("cnt")).filter(pl.col("cnt") > 1)
        print(f"   Found {duplicates.height} duplicate flight_ids")
        if duplicates.height > 0:
            print(f"   Example: {duplicates.head(3)}")

if __name__ == "__main__":
    transform_silver()