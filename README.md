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

- Результат: 582,425 строк, 105 колонок

- Сплитование: по дням

- Формат: Delta Lake (Parquet + _delta_log)

- Режим записи: overwrite для первой загрузки

df = pl.read_csv(csv_path, try_parse_dates=True)
write_deltalake(bronze_path, df.to_pandas(), mode="overwrite")

<img width="614" height="413" alt="image" src="https://github.com/user-attachments/assets/01bb02e3-3604-482d-a044-79f6c13a8a4b" />

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

5. Результат: 557,733 строк, 13 колонок

df.with_columns([
    (pl.col("CRSDepTime") // 100).alias("hour"),
    pl.when(pl.col("month").is_between(3, 5)).then(pl.lit("Spring"))
     .when(pl.col("month").is_between(6, 8)).then(pl.lit("Summer"))
     .otherwise(pl.lit("Winter")).alias("season"),
])

<img width="491" height="307" alt="image" src="https://github.com/user-attachments/assets/69c65e55-4962-435a-a4bb-4f923fe77d8f" />

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

<img width="568" height="48" alt="image" src="https://github.com/user-attachments/assets/adb62a91-28fa-4f2b-af5d-65b0088946b5" />

## 2. Машинное обучение

### Регрессия (предсказание задержки в минутах)

<img width="220" height="48" alt="image" src="https://github.com/user-attachments/assets/3a5e14db-d82c-4ab4-817d-ad0ec6e07f7e" />

### Классификация (задержка > 15 минут)

<img width="169" height="48" alt="image" src="https://github.com/user-attachments/assets/ab45e4a2-4c30-4569-a330-d526f7ab3df9" />

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

<img width="890" height="271" alt="image" src="https://github.com/user-attachments/assets/5d64644b-5c5a-4287-8b4d-fa12ef43754b" />

<img width="874" height="48" alt="image" src="https://github.com/user-attachments/assets/01eadebb-a2b1-44de-a98d-e82834181c0d" />

4. Логируемые метрики:

- regression_mae

- classification_auc

- feature_imp_* - важность признаков

<img width="879" height="222" alt="image" src="https://github.com/user-attachments/assets/2a7caade-cabe-466c-a2db-eb62db77f349" />

<img width="885" height="247" alt="image" src="https://github.com/user-attachments/assets/480a5418-f3db-473d-ad8a-a3d7672d7193" />

<img width="887" height="85" alt="image" src="https://github.com/user-attachments/assets/8e9d57e0-a8bf-4916-a814-ffd9c16e0d02" />

5. Логируемые модели:

- random_forest_regressor

- random_forest_classifier

<img width="921" height="259" alt="image" src="https://github.com/user-attachments/assets/ef66dee9-86fc-414a-a0ef-20e057bd0121" />

<img width="896" height="137" alt="image" src="https://github.com/user-attachments/assets/30b33b4e-34a3-4356-acc9-7bf9cc2538f2" />

<img width="813" height="134" alt="image" src="https://github.com/user-attachments/assets/875c0fb3-1e73-4977-9050-c92db118a84f" />

<img width="719" height="142" alt="image" src="https://github.com/user-attachments/assets/46967c8e-dde0-4102-a664-2d761bd80e01" />

<img width="184" height="143" alt="image" src="https://github.com/user-attachments/assets/a7b836ac-6231-461b-8bd2-4e5252db1556" />

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

<img width="783" height="434" alt="image" src="https://github.com/user-attachments/assets/52e95c85-8d02-4a10-90d6-65e769826c60" />

<img width="727" height="50" alt="image" src="https://github.com/user-attachments/assets/4c8bd37a-5fff-43e1-8cca-1b614851015c" />

2. VACUUM DEMO

<img width="783" height="256" alt="image" src="https://github.com/user-attachments/assets/4aa2a7ad-0855-4c68-96ff-f87d6a19615f" />

3. OPTIMIZE & Z-ORDER DEMO

<img width="791" height="415" alt="image" src="https://github.com/user-attachments/assets/d75e3381-42bd-4475-8e52-b5f3b89f3efe" />

4. PARTITION PRUNING DEMO

<img width="1338" height="598" alt="image" src="https://github.com/user-attachments/assets/173dc2a7-a530-45d0-ac7c-f7643b941ae5" />

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

- После очистки: 557,733 строк

- Feature table: 446,186 строк

Вывод: Очистка данных удалила всего 4.2% записей, что говорит о хорошем качестве исходного датасета.

Качество моделей:

1. Регрессия: MAE = 28.73 мин

- Модель способна предсказывать задержки с приемлемой точностью

- Ошибка в 29 минут - значительная для коротких перелётов

- Для улучшения нужны дополнительные признаки (погода, день недели, праздники)

2. Классификация: AUC = 0.626

- Модель способна отличать задержанные рейсы от своевременных

- AUC 0.626 - умеренное качество (хороший результат - 0.7-0.8)

Вывод: Регрессия более полезна для практики, так как даёт точное время задержки.
