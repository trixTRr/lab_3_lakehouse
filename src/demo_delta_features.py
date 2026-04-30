import polars as pl
from deltalake import DeltaTable
import os

def print_section(title):
    print("\n" + "=" * 70)
    print(f"{title}")
    print("=" * 70)

def demo_time_travel():
    # Демонстрация Time Travel
    print_section("TIME TRAVEL DEMO")
    
    silver_path = "data/silver/flights"
    
    if not os.path.exists(silver_path):
        print(f"Table not found: {silver_path}")
        return
    
    dt = DeltaTable(silver_path)
    
    # Текущая версия
    current_version = dt.version()
    print(f"\nCurrent version: {current_version}")
    
    # История версий
    print("\nVersion history:")
    history = dt.history()
    for h in history[:5]:
        print(f"   Version {h['version']}: {h['operation']} at {h['timestamp']}")
    
    # Time Travel к версии 0
    if current_version > 0:
        print(f"\nReading version 0 (Time Travel)...")
        df_v0 = pl.read_delta(silver_path, version=0)
        df_current = pl.read_delta(silver_path)
        print(f"   Version 0 rows: {df_v0.height:,}")
        print(f"   Current rows: {df_current.height:,}")
        
        if df_current.height != df_v0.height:
            print(f"   Difference: {df_current.height - df_v0.height:+,} rows")
        else:
            print(f"   Row count unchanged (optimization operation)")
        
        # Проверяем, какие годы есть в данных
        print(f"\n   Year distribution in current version:")
        year_counts = df_current.group_by("year").count().sort("year")
        for row in year_counts.iter_rows():
            print(f"      {row[0]}: {row[1]:,} flights")
    else:
        print("   Only one version available")

def demo_vacuum():
    # Демонстрация VACUUM
    print_section("VACUUM DEMO")
    
    silver_path = "data/silver/flights"
    
    if not os.path.exists(silver_path):
        print(f"Table not found: {silver_path}")
        return
    
    dt = DeltaTable(silver_path)
    
    print(f"\nTable: {silver_path}")
    print(f"Current version: {dt.version()}")
    
    # Получаем информацию о размере таблицы через файловую систему
    try:
        import subprocess
        result = subprocess.run(
            ["du", "-sh", silver_path], 
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"Table size: {result.stdout.split()[0]}")
    except:
        pass
    
    # Vacuum dry run с retention 168 часов (7 дней)
    print("\nRunning VACUUM (dry run) with 168 hours retention...")
    try:
        files_to_delete = dt.vacuum(dry_run=True, retention_hours=168)
        if files_to_delete and len(files_to_delete) > 0:
            print(f"   Would delete: {len(files_to_delete)} files")
            print(f"   Files older than 7 days would be removed")
        else:
            print("   No files to delete (all within retention period)")
    except Exception as e:
        print(f"   Vacuum demo: {e}")
        print("   (This is normal if no old versions exist)")

def demo_optimize():
    # Демонстрация OPTIMIZE и Z-ORDER
    print_section("OPTIMIZE & Z-ORDER DEMO")
    
    silver_path = "data/silver/flights"
    
    if not os.path.exists(silver_path):
        print(f"Table not found: {silver_path}")
        return
    
    dt = DeltaTable(silver_path)
    
    print(f"\nBEFORE OPTIMIZE:")
    print(f"   Version: {dt.version()}")
    
    # OPTIMIZE (compaction)
    print("\nRunning OPTIMIZE (compaction)...")
    try:
        dt.optimize.compact()
        print("   Compaction completed")
    except Exception as e:
        print(f"   Compaction: {e}")
    
    # Z-ORDER
    print("\nRunning Z-ORDER on (year, month)...")
    try:
        dt.optimize.z_order(["year", "month"])
        print("   Z-ORDER completed")
    except Exception as e:
        print(f"   Z-ORDER: {e}")
    
    print(f"\nAFTER OPTIMIZE:")
    print(f"   Version: {dt.version()}")
    print("\nOptimization complete")

def demo_partition_pruning():
    # Демонстрация partition pruning через explain()
    print_section("PARTITION PRUNING DEMO")
    
    silver_path = "data/silver/flights"
    
    if not os.path.exists(silver_path):
        print(f"Table not found: {silver_path}")
        return
    
    print("\nQuery: filter(year == 2024) → group_by(OP_CARRIER) → mean(ARR_DELAY)")
    
    query = (pl.scan_delta(silver_path)
             .filter(pl.col("year") == 2024)
             .group_by("OP_CARRIER")
             .agg(pl.col("ARR_DELAY").mean()))
    
    plan = query.explain()
    print(plan)
    
    # Выполняем запрос и показываем результат
    print("\nActual Results (top 10 carriers by avg delay in 2024):")
    try:
        result = query.collect().sort("ARR_DELAY", descending=True).head(10)
        for row in result.iter_rows():
            print(f"   {row[0]}: {row[1]:.1f} minutes")
    except Exception as e:
        print(f"   Could not execute query: {e}")

def main():
    print("   DELTA LAKE FEATURES DEMONSTRATION")
    
    # Проверка существования таблицы
    if not os.path.exists("data/silver/flights"):
        print("\nSilver table not found!")
        print("   Please run: python src/silver/transform_final.py")
        return
    
    demo_time_travel()
    demo_vacuum()
    demo_optimize()
    demo_partition_pruning()

if __name__ == "__main__":
    main()