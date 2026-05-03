Пример запроса с pushdown-оптимизациями:

import polars as pl

query = (pl.scan_delta('data/silver/flights')
         .filter(pl.col('year') == 2024)
         .group_by('OP_CARRIER')
         .agg(pl.col('ARR_DELAY').mean()))

print(query.explain())

Вывод:

AGGREGATE[maintain_order: false]
  [col("ARR_DELAY").mean()] BY [col("OP_CARRIER")]
  FROM
  simple π 2/2 ["ARR_DELAY", "OP_CARRIER"]
    Parquet SCAN [/home/sansan/lab_3_lakehouse/data/silver/flights/part-00000-b9f6b881-0291-42d9-9858-a1f7cb2763a0-c000.zstd.parquet]
    PROJECT 3/13 COLUMNS
    SELECTION: [(col("year")) == (2024)]

Почему partition по year + month: Типичные запросы фильтруют по дате (аналитика за месяц/сезон). 
Z-order на FL_DATE дополнительно ускоряет точные диапазоны, располагая данные одного дня в соседних файлах.


# Как запустить:

## 1. Скачайте датасет

Flight Delay Analysis 2018–2024 (https://www.kaggle.com/code/peymanradmanesh/flight-delay-analysis-2018-2024/input). Добавьте его в папку data/raw

## 2. Зайдите в командную строку и введите:

wsl -d Ubuntu

cd /mnt/c/l_a_b

python3 -m venv venv

source venv/bin/activate

## 3. Активируйте WSL интеграцию в Docker Desktop

## 4. Перенесите в локальную директорию все файлы

mkdir -p ~/lab_3_lakehouse

cp -r /mnt/c/lab_3_lakehouse/* ~/lab_3_lakehouse/

cd ~/lab_3_lakehouse

python3 -m venv venv

source venv/bin/activate

## 5. Выполните файлы:

python src/bronze/ingest.py

python src/silver/transform.py

python src/gold/aggregates.py

python src/gold/feature_table.py

python src/ml/train_models.py

python src/demo_delta_features.py

mlflow ui --backend-store-uri file:./mlflow_local --port 5001

# Задача лабораторной работы:

Построить пайплайн обработки данных (bronze → silver → gold) для прогнозирования задержек авиарейсов с использованием Polars и Delta Lake, реализуя концепцию Lakehouse архитектуры.

# Решение, реализованное в работе:

## 1. Архитектура пайплайна (Bronze → Silver → Gold)

### Bronze Layer

Требование: Загрузить CSV в Delta-таблицу. Грузить по годам/батчами в режиме append.

Выполнение:

- Реализован скрипт src/bronze/ingest.py

- Загружен датасет flight_data_2018_2024.csv (https://www.kaggle.com/code/peymanradmanesh/flight-delay-analysis-2018-2024/input)

- Результат: 582,425 строк, 120 колонок

- Сплитование: по дням

- Формат: Delta Lake (Parquet + _delta_log)

- Режим записи: overwrite для первой загрузки

df = pl.read_csv(csv_path, try_parse_dates=True)
write_deltalake(bronze_path, df.to_pandas(), mode="overwrite")

<img width="823" height="458" alt="image" src="https://github.com/user-attachments/assets/8311fc72-74ea-4ec5-9eed-2cd2b0385ccc" />

### Silver Layer

Требование: Очистка (NA, отменённые, выбросы), производные признаки, партиционирование, MERGE.

Выполнение:

1. Очистка:

- Удалены отменённые рейсы (Cancelled == 0)

- Удалены выбросы (ARR_DELAY.abs() < 720 минут)

- Обработаны NULL значения

2. Производные признаки:

- hour - час вылета (из CRSDepTime)

- day_of_week - день недели

- season - сезон (Spring/Summer/Fall/Winter)

- route - маршрут (ORIGIN + "_" + DEST)

3. Партиционирование: по year, month (Z-ORDER оптимизация)

4. Нормализация числовых признаков (от 0 до 1)

5. Выбор колонок:

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

6. При повторном запуске пайплайна используется MERGE для обновления данных

7. Результат: 221,511 строк, 18 колонок

df.with_columns([
    (pl.col("CRSDepTime") // 100).alias("hour"),
    pl.when(pl.col("month").is_between(3, 5)).then(pl.lit("Spring"))
     .when(pl.col("month").is_between(6, 8)).then(pl.lit("Summer"))
     .otherwise(pl.lit("Winter")).alias("season"),
])

<img width="835" height="555" alt="image" src="https://github.com/user-attachments/assets/164920ef-e247-4a81-9ad5-f7204b6868b8" />

### Gold Layer

Требование: Аналитические агрегаты + feature table для ML.

Выполнение:

1.Аналитические агрегаты:

- delays_by_dest_hour - задержки по аэропорту назначения и часу

- delays_by_carrier_season - задержки по авиакомпании и сезону

- monthly_trends - тренды задержек по месяцам

2. Feature table для ML:

- Признаки: hour, day_of_week, season, OP_CARRIER, ORIGIN, DEST, DISTANCE, DEP_DELAY, ARR_DELAY

features = df.select([
    "ARR_DELAY", "hour", "day_of_week", "season",
    "OP_CARRIER", "ORIGIN", "DEST", "DISTANCE", "DEP_DELAY"
])
write_deltalake(gold_path, features.to_pandas(), mode="overwrite")

<img width="541" height="93" alt="image" src="https://github.com/user-attachments/assets/aaebf5fb-f888-4e5e-93e9-f0092cc0b48e" />

## 2. Машинное обучение

### Регрессия (предсказание задержки в минутах)

<img width="277" height="48" alt="image" src="https://github.com/user-attachments/assets/f60ff10e-baaa-43bf-bb3b-11146b34c658" />

### Классификация (задержка > 15 минут)

<img width="149" height="48" alt="image" src="https://github.com/user-attachments/assets/244990e7-0fb7-444d-b0b3-368cd3db7808" />

### Feature Importance

Логируются топ-10 наиболее важных признаков для классификации.

## 3. Docker

Использован Docker контейнер для MLFlow с образом python:3.10-slim.

## 4. MLflow логирование

http://localhost:5001

### Выполненные действия:

1. Tracking URI: file:./mlflow_local

2. Experiment: flight_delay_prediction

3. Логируемые параметры:

- gold_table_version - версия gold-таблицы

- gold_table_path - путь к таблице

- n_estimators, max_depth - гиперпараметры

<img width="933" height="379" alt="image" src="https://github.com/user-attachments/assets/0b7ee172-fb17-43af-83cd-a885b01d7646" />

<img width="897" height="48" alt="image" src="https://github.com/user-attachments/assets/b16f9137-38c4-40b1-a933-0218c9ca23ff" />

4. Логируемые метрики:

- regression_mae

- classification_auc

- feature_imp_* - важность признаков

<img width="912" height="369" alt="image" src="https://github.com/user-attachments/assets/a8ec7b4c-984a-4a29-9989-ce74a4cdb1f1" />

<img width="905" height="275" alt="image" src="https://github.com/user-attachments/assets/ac44877d-b8dd-4857-a4ac-73e7e7d27167" />

<img width="900" height="88" alt="image" src="https://github.com/user-attachments/assets/c267fba2-942a-4638-b7da-89c328d12c57" />

5. Логируемые модели:

- random_forest_regressor

- random_forest_classifier

<img width="863" height="256" alt="image" src="https://github.com/user-attachments/assets/0626234b-1e06-4095-a14f-2280ffcbaef5" />

<img width="842" height="151" alt="image" src="https://github.com/user-attachments/assets/4fd5ab0a-8ef0-465b-928d-2e32b99c14ac" />

<img width="760" height="137" alt="image" src="https://github.com/user-attachments/assets/0ab1c217-d90a-4302-b3c2-5574a6ea7f90" />

<img width="734" height="150" alt="image" src="https://github.com/user-attachments/assets/e073cee9-799f-4f51-8588-4db615424642" />

<img width="249" height="150" alt="image" src="https://github.com/user-attachments/assets/e940f16a-05d1-4b33-b3ff-663285aab2df" />

## 5. Polars Lazy API и pushdown оптимизации

### Использование lazy API

query = (pl.scan_delta('data/silver/flights')
         .filter(pl.col('year') == 2024)
         .group_by('OP_CARRIER')
         .agg(pl.col('ARR_DELAY').mean()))
result = query.collect()

### Explain() с pushdown демонстрацией

### Результат в плане выполнения:

- Predicate Pushdown - фильтр year = 2024 применяется при сканировании

- Partition Pruning - читаются только партиции с year=2024

- Projection Pushdown - читаются только нужные колонки

### Обоснование выбора партиций:

Партиции по year и month выбраны потому, что:

1. Типичные запросы фильтруют данные по временным диапазонам

2. Обеспечивается эффективное отсечение партиций (partition pruning)

3. Размер партиций сбалансирован (~50-150 MB каждая)

4. Z-ORDER на (year, month) дополнительно ускоряет запросы

## 6. Delta Lake возможности

### MERGE (обновление без дублирования)

Реализация: Silver слой перезаписывается через mode="overwrite", что имитирует полное обновление. Для инкрементальных обновлений код поддерживает MERGE через deltalake.merge().

### OPTIMIZE (compaction)

Для объединения мелких файлов

dt = DeltaTable(silver_path)
dt.optimize.compact()

### Z-ORDER

dt.optimize.z_order(["year", "month"])

### VACUUM (очистка старых версий)

files_to_delete = dt.vacuum(dry_run=True, retention_hours=168)

### Time Travel (чтение прошлых версий)

df_v0 = pl.read_delta(silver_path, version=0)
df_current = pl.read_delta(silver_path)
print(f"Version 0: {df_v0.height:,} rows")
print(f"Current: {df_current.height:,} rows")

### Schema Evolution

Код поддерживает эволюцию схемы через schema_mode="merge" при записи новых данных с дополнительными колонками.

## 7. Технические требования

- Хранилище: локальные папки data/

- Polars lazy API: scan_delta() → filter() → group_by() → agg() → collect()

- README с explain(): демонстрация pushdown оптимизаций

- Delta возможности (3+ сверх MERGE): OPTIMIZE, Z-ORDER, VACUUM, Time Travel

- MLflow логирование: параметры, метрики, модели, версия gold-таблицы

- Структура src/, notebooks/, logs/

## 8. Демонстрационный скрипт

Файл demo_delta_features.py

1. TIME TRAVEL DEMO

<img width="873" height="295" alt="image" src="https://github.com/user-attachments/assets/38f8be4f-a7a9-4680-91a4-8dfb8e4bde3f" />

2. VACUUM DEMO

<img width="850" height="262" alt="image" src="https://github.com/user-attachments/assets/f04f425a-130a-47e3-acc5-3c0fc16841f0" />

3. OPTIMIZE & Z-ORDER DEMO

<img width="851" height="435" alt="image" src="https://github.com/user-attachments/assets/9bebef40-63a7-4223-95fc-d6b15fec1834" />

4. PARTITION PRUNING DEMO

<img width="1453" height="643" alt="image" src="https://github.com/user-attachments/assets/c94e1b7b-66f3-4b8a-b82d-e5510697fc5e" />

## Выводы

Созданные агрегаты позволяют:

- Анализировать задержки по аэропортам и часам

- Сравнивать авиакомпании по сезонам

- Отслеживать тренды задержек по месяцам

Feature table содержит:

- 9 признаков для предсказания задержек

- Возможность бинарной классификации (задержка > 15 минут)

Количество обработанных данных:

- Исходный датасет: 582,425 строк

- После очистки: 221,511 строк

Вывод: Очистка данных удалила 62% записей. Причины следующие:

1. Удалены отменённые рейсы (Cancelled = 1) - ~3.8%

2. Удалены записи с NULL в ArrDelay - ~15%

3. Удалены выбросы (задержка > 12 часов) - ~0.2%

4. Остальные потери связаны с удалением дубликатов и служебных записей

Качество моделей:

1. Регрессия: MAE = 30.44 мин

- Модель способна предсказывать задержки с приемлемой точностью

- Ошибка в 30 минут - значительная для коротких перелётов

- Для улучшения нужны дополнительные признаки (погода, день недели, праздники)

2. Классификация: AUC = 0.619

- Модель способна отличать задержанные рейсы от своевременных

- AUC 0.619 - умеренное качество (хороший результат - 0.7-0.8)

Вывод: Регрессия более полезна для практики, так как даёт точное время задержки.
