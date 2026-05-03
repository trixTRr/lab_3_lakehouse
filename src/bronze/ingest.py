import polars as pl
from deltalake import write_deltalake, DeltaTable
import os
from datetime import datetime

def ingest_bronze():
    # Загрузка CSV в Delta таблицу с партиционированием по дате
    
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
        
        # Читаем CSV
        try:
            df = pl.read_csv(
                os.path.join(raw_path, csv_file),
                null_values=["NA", "", "NULL", "null"],
                try_parse_dates=True,
                infer_schema_length=10000
            )
            print(f"   Прочитано строк: {df.height}")
            print(f"   Исходных колонок: {len(df.columns)}")
        except Exception as e:
            print(f"   Ошибка чтения: {e}")
            continue
        
        # Удаляем полностью пустые колонки
        before_cols = len(df.columns)
        df = df.drop([col for col in df.columns if df[col].null_count() == df.height])
        after_cols = len(df.columns)
        
        if before_cols > after_cols:
            print(f"   Удалено пустых колонок: {before_cols - after_cols}")
        
        # Преобразуем типы для числовых колонок
        numeric_cols = ["ARR_DELAY", "DEP_DELAY", "DISTANCE", "CANCELLED", "DIVERTED", 
                        "ArrDelay", "DepDelay", "Distance", "Cancelled", "Diverted"]
        for col in numeric_cols:
            if col in df.columns:
                df = df.with_columns(pl.col(col).cast(pl.Float64).fill_null(0))
        
        # Преобразуем строковые колонки
        string_cols = ["OP_CARRIER", "ORIGIN", "DEST", "TAIL_NUM", "FL_NUM",
                       "Marketing_Airline_Network", "Origin", "Dest"]
        for col in string_cols:
            if col in df.columns:
                df = df.with_columns(pl.col(col).cast(pl.Utf8).fill_null("UNKNOWN"))
        
        # Определяем колонку с датой
        date_col = None
        if "FL_DATE" in df.columns:
            date_col = "FL_DATE"
        elif "FlightDate" in df.columns:
            date_col = "FlightDate"
        
        if date_col is None:
            print(f"   Нет колонки с датой! Загружаем целиком")
            df = df.with_columns([
                pl.lit(datetime.now().isoformat()).alias("_ingestion_timestamp"),
                pl.lit(csv_file).alias("_source_file")
            ])
            write_deltalake(bronze_path, df.to_pandas(), mode="overwrite")
            continue
        
        # Преобразуем дату в нужный формат
        if df[date_col].dtype != pl.Date:
            df = df.with_columns(pl.col(date_col).str.strptime(pl.Date, "%Y-%m-%d"))
        
        # Проверяем, есть ли уже колонки Year и Month
        has_year = "Year" in df.columns
        has_month = "Month" in df.columns
        
        # Определяем колонки для партиционирования
        if has_year and has_month:
            print(f"   Используем существующие колонки 'Year' и 'Month'")
            
            # Ищем колонку с днем месяца (разные варианты написания)
            day_col = None
            for col in ["DayOfMonth", "DayofMonth", "Day_Of_Month", "DAY_OF_MONTH"]:
                if col in df.columns:
                    day_col = col
                    break
            
            if day_col is None:
                # Если нет - добавляем
                df = df.with_columns(pl.col(date_col).dt.day().alias("DayOfMonth"))
                day_col = "DayOfMonth"
                print(f"   Добавлена колонка 'DayOfMonth'")
            else:
                print(f"   Используем существующую колонку '{day_col}'")
            
            partition_cols = ["Year", "Month", day_col]
            
        else:
            # Если нет Year/Month - создаём новые с другими именами
            print(f"   Создаём колонки 'year_num', 'month_num', 'day_num'")
            
            if "year_num" not in df.columns:
                df = df.with_columns(pl.col(date_col).dt.year().alias("year_num"))
            if "month_num" not in df.columns:
                df = df.with_columns(pl.col(date_col).dt.month().alias("month_num"))
            if "day_num" not in df.columns:
                df = df.with_columns(pl.col(date_col).dt.day().alias("day_num"))
            
            partition_cols = ["year_num", "month_num", "day_num"]
        
        # Добавляем служебные колонки (если их ещё нет)
        if "_ingestion_timestamp" not in df.columns:
            df = df.with_columns(pl.lit(datetime.now().isoformat()).alias("_ingestion_timestamp"))
        if "_source_file" not in df.columns:
            df = df.with_columns(pl.lit(csv_file).alias("_source_file"))
        
        # Сортируем по дате
        df = df.sort(date_col)
        
        unique_days = df[date_col].unique().len()
        print(f"   Уникальных дней: {unique_days}")
        print(f"   Партиционирование по: {partition_cols}")
        
        # Записываем все данные одной операцией
        try:
            # Удаляем старую таблицу
            if os.path.exists(bronze_path):
                import shutil
                shutil.rmtree(bronze_path)
                print(f"   Удалена старая таблица")
            
            # Записываем с партиционированием
            write_deltalake(
                bronze_path, 
                df.to_pandas(), 
                mode="overwrite",
                partition_by=partition_cols
            )
            print(f"   Данные записаны с партиционированием по {partition_cols}")
                
        except Exception as e:
            print(f"   Ошибка записи: {e}")
            continue
    
    # Информация о результате
    if os.path.exists(f"{bronze_path}/_delta_log"):
        dt = DeltaTable(bronze_path)
        print(f"\nBronze layer создан!")
        print(f"   Путь: {bronze_path}")
        print(f"   Версия: {dt.version()}")
        
        print(f"\n   История версий (последние 5):")
        for h in dt.history()[:5]:
            print(f"      Version {h['version']}: {h['operation']} - {h['timestamp']}")
    else:
        print(f"\nНе удалось создать Bronze слой")

if __name__ == "__main__":
    ingest_bronze()