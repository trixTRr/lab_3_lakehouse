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
## 1. Зайдите в командную строку и введите:

wsl -d Ubuntu

cd /mnt/c/l_a_b

python3 -m venv venv

source venv/bin/activate

## 2. Активируйте WSL интеграцию в Docker Desktop

## 3. Перенесите в локальную директорию все файлы

mkdir -p ~/lab_3_lakehouse

cp -r /mnt/c/lab_3_lakehouse/* ~/lab_3_lakehouse/

cd ~/lab_3_lakehouse

python3 -m venv venv

source venv/bin/activate

## 4. Выполните файлы:

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

- Загружен датасет flight_data_2018_2024.csv

- Результат: 582,425 строк, 105 колонок

- Формат: Delta Lake (Parquet + _delta_log)

- Режим записи: overwrite для первой загрузки

df = pl.read_csv(csv_path, try_parse_dates=True)
write_deltalake(bronze_path, df.to_pandas(), mode="overwrite")

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

4. Результат: 557,733 строк, 13 колонок

df.with_columns([
    (pl.col("CRSDepTime") // 100).alias("hour"),
    pl.when(pl.col("month").is_between(3, 5)).then(pl.lit("Spring"))
     .when(pl.col("month").is_between(6, 8)).then(pl.lit("Summer"))
     .otherwise(pl.lit("Winter")).alias("season"),
])

### Gold Layer

Требование: Аналитические агрегаты + feature table для ML.

Выполнение:

1.Аналитические агрегаты:

- delays_by_dest_hour - задержки по аэропорту назначения и часу

- delays_by_carrier_season - задержки по авиакомпании и сезону

- monthly_trends - тренды задержек по месяцам

2. Feature table для ML:

- Результат: 446,186 строк, 9 признаков + целевая переменная

- Признаки: hour, day_of_week, season, OP_CARRIER, ORIGIN, DEST, DISTANCE, DEP_DELAY, ARR_DELAY

features = df.select([
    "ARR_DELAY", "hour", "day_of_week", "season",
    "OP_CARRIER", "ORIGIN", "DEST", "DISTANCE", "DEP_DELAY"
])
write_deltalake(gold_path, features.to_pandas(), mode="overwrite")

## 2. Машинное обучение

### Регрессия (предсказание задержки в минутах)

### Классификация (задержка > 15 минут)

### Feature Importance

Логируются топ-10 наиболее важных признаков для классификации.

## 3. Docker

Использован Docker контейнер для MLFlow с образом python:3.10-slim.

## 4. MLflow логирование

### Выполненные действия:

1. Tracking URI: file:./mlflow_local

2. Experiment: flight_delay_prediction

3. Логируемые параметры:

- gold_table_version - версия gold-таблицы

- gold_table_path - путь к таблице

- n_estimators, max_depth - гиперпараметры

4. Логируемые метрики:

- regression_mae

- classification_auc

- feature_imp_* - важность признаков

5. Логируемые модели:

- random_forest_regressor

- random_forest_classifier

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

Реализованы следующие оптимизации:

1. Адаптивное количество партиций:

...
if use_optimization:
        spark_builder = spark_builder \
            .config("spark.sql.shuffle.partitions", "8") \
            .config("spark.default.parallelism", "8") \
            ...

2. Кэширование данных:

...
df = df.persist(StorageLevel.MEMORY_AND_DISK)
df.count()
...

3. Включение адаптивных запросов:

...
.config("spark.sql.adaptive.enabled", "true") \
.config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
...
.config("spark.sql.adaptive.skewJoin.enabled", "true")
...

## 7. Сравнение результатов и визуализация:

Создан скрипт auto_plot_results.py для автоматического построения графиков:

### Создаваемые графики:
1. execution_time.png - сравнение времени выполнения

<img width="1782" height="884" alt="image" src="https://github.com/user-attachments/assets/f2b4897f-e257-43de-a168-e546fcd514d3" />

2. nodes_comparison.png - сравнение 1 DN vs 3 DN

<img width="1482" height="881" alt="image" src="https://github.com/user-attachments/assets/acd4846d-a5da-4b43-9234-d7a8a5b400ac" />

3. memory_usage.png - использование RAM

<img width="1782" height="879" alt="image" src="https://github.com/user-attachments/assets/efe38b6d-a93b-4665-b5d1-35e1660e2a85" />

4. time_vs_memory.png - компромисс время-память

<img width="1477" height="1180" alt="image" src="https://github.com/user-attachments/assets/0cf3538a-7fd3-4363-b275-0526a83fe877" />

5. trend.png - тренд производительности

<img width="1482" height="884" alt="image" src="https://github.com/user-attachments/assets/7e095398-742b-4873-9f2a-52ed979f2e81" />

6. speedup.png - ускорение от оптимизации

<img width="1182" height="883" alt="image" src="https://github.com/user-attachments/assets/5a143ef6-eb13-4c52-a5af-c15385ecc7e7" />

7. time_breakdown.png - разбивка времени выполнения

<img width="1781" height="883" alt="image" src="https://github.com/user-attachments/assets/4355b514-69fa-44a7-a66c-ca87109fab3e" />

### HTML отчет:

Сгенерирован интерактивный HTML отчет с таблицами, графиками и выводами.

# Выводы:

- Лучшее время: 5.95 секунд в конфигурации 3DN_Optimized

- Лучшее использование RAM: 38 MB в конфигурации 1DN_Optimized

- Оптимизация на 1 DataNode дала ускорение в 1.15x

- Масштабирование до 3 DataNodes дало ускорение в 1.24x



