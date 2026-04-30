import polars as pl
from deltalake import write_deltalake
import os
from datetime import datetime

def ingest_bronze():
    # Загрузка CSV в Delta таблицу (локально)
    
    raw_path = "data/raw"
    bronze_path = "data/bronze/flights"
    
    if not os.path.exists(raw_path):
        print(f"Папка '{raw_path}' не существует!")
        return
    
    os.makedirs(bronze_path, exist_ok=True)
    
    csv_files = [f for f in os.listdir(raw_path) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"В папке '{raw_path}' нет CSV файлов!")
        return
    
    print(f"Найдено CSV файлов: {len(csv_files)}")
    
    for csv_file in csv_files:
        print(f"\nОбработка {csv_file}...")
        
        # Читаем CSV с указанием типов для важных колонок
        try:
            df = pl.read_csv(
                os.path.join(raw_path, csv_file),
                null_values=["NA", "", "NULL", "null"],
                try_parse_dates=True,
                infer_schema_length=10000  # Увеличиваем для лучшего определения типов
            )
            print(f"   Прочитано строк: {df.height}")
            print(f"   Исходных колонок: {len(df.columns)}")
        except Exception as e:
            print(f"   Ошибка чтения: {e}")
            continue
        
        # 1. Удаляем столбцы, которые полностью состоят из NULL
        before_cols = len(df.columns)
        df = df.drop([col for col in df.columns if df[col].null_count() == df.height])
        after_cols = len(df.columns)
        
        if before_cols > after_cols:
            print(f"   Удалено пустых колонок: {before_cols - after_cols}")
        
        # 2. Преобразуем типы для критических колонок
        # Список колонок, которые должны быть числовыми
        numeric_cols = ["ARR_DELAY", "DEP_DELAY", "DISTANCE", "CANCELLED", "DIVERTED"]
        for col in numeric_cols:
            if col in df.columns:
                df = df.with_columns(pl.col(col).cast(pl.Float64).fill_null(0))
        
        # 3. Преобразуем строковые колонки
        string_cols = ["OP_CARRIER", "ORIGIN", "DEST", "TAIL_NUM", "FL_NUM"]
        for col in string_cols:
            if col in df.columns:
                df = df.with_columns(pl.col(col).cast(pl.Utf8).fill_null("UNKNOWN"))
        
        # 4. Преобразуем дату
        if "FL_DATE" in df.columns:
            df = df.with_columns(pl.col("FL_DATE").cast(pl.Utf8))
        
        # Добавляем технические колонки
        df = df.with_columns([
            pl.lit(datetime.now().isoformat()).alias("_ingestion_timestamp"),
            pl.lit(csv_file).alias("_source_file")
        ])
        
        # Проверяем финальные типы
        print(f"   Финальных колонок: {len(df.columns)}")
        print(f"   Типы данных:")
        for col in df.columns[:5]:  # Показываем первые 5 колонок
            print(f"      - {col}: {df[col].dtype}")
        
        # Записываем в Delta
        mode = "overwrite" if os.path.exists(f"{bronze_path}/_delta_log") else "overwrite"
        try:
            write_deltalake(bronze_path, df.to_pandas(), mode=mode)
            print(f"   Записано в Delta (mode={mode})")
        except Exception as e:
            print(f"   Ошибка записи: {e}")
            # Показываем проблемные колонки
            print(f"   Проверка типов колонок:")
            for col in df.columns:
                print(f"      {col}: {df[col].dtype}")
            continue
    
    # Информация о результате
    if os.path.exists(f"{bronze_path}/_delta_log"):
        print(f"\nBronze layer создан!")
        print(f"   Путь: {bronze_path}")
    else:
        print(f"\nНе удалось создать Bronze слой")

if __name__ == "__main__":
    ingest_bronze()