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