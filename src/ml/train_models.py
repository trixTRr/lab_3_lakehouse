import mlflow
import mlflow.sklearn
import pandas as pd
import polars as pl
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, roc_auc_score
import os
from deltalake import DeltaTable

def get_table_version(table_path):
    # Получить текущую версию Delta-таблицы
    try:
        dt = DeltaTable(table_path)
        return dt.version()
    except Exception as e:
        return None

def train_models():
    feature_path = "data/gold/feature_table"
    
    if not os.path.exists(feature_path):
        print("Feature table not found")
        return
    
    # Получаем версию gold-таблицы
    gold_version = get_table_version(feature_path)
    
    print("Loading feature table...")
    df = pl.read_delta(feature_path).to_pandas()
    
    print(f"Total rows: {len(df):,}")
    
    # Создаём целевую переменную для классификации
    df["is_delayed"] = (df["ARR_DELAY"] > 15).astype(int)
    
    # Разделяем признаки и цели
    X = df.drop(columns=["ARR_DELAY", "is_delayed"])
    y_reg = df["ARR_DELAY"]
    y_clf = df["is_delayed"]
    
    # One-hot encoding для категориальных колонок
    categorical_cols = ["OP_CARRIER", "ORIGIN", "DEST"]
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    print(f"Training shape: {X.shape}")
    print(f"Positive class rate: {y_clf.mean():.3f}")
    
    X_train, X_test, y_train_reg, y_test_reg = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )
    _, _, y_train_clf, y_test_clf = train_test_split(
        X, y_clf, test_size=0.2, random_state=42
    )
    
    mlflow.set_tracking_uri("file:./mlflow_local")
    mlflow.set_experiment("flight_delay_prediction")
    
    with mlflow.start_run(run_name="flight_delay_models_final") as run:
        # Логируем версию gold-таблицы
        if gold_version is not None:
            mlflow.log_param("gold_table_version", gold_version)
            mlflow.log_param("gold_table_path", feature_path)
        
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth_reg", 15)
        mlflow.log_param("max_depth_clf", 10)
        mlflow.log_param("sample_size", len(df))
        
        # Регрессия
        print("\nTraining regression model...")
        rf_reg = RandomForestRegressor(
            n_estimators=100, 
            max_depth=15, 
            random_state=42,
            n_jobs=-1
        )
        rf_reg.fit(X_train, y_train_reg)
        y_pred_reg = rf_reg.predict(X_test)
        mae = mean_absolute_error(y_test_reg, y_pred_reg)
        
        mlflow.log_metric("regression_mae", mae)
        mlflow.sklearn.log_model(rf_reg, "random_forest_regressor")
        print(f"  MAE: {mae:.2f} minutes")
        
        # Классификация
        print("\nTraining classification model...")
        rf_clf = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10, 
            random_state=42,
            n_jobs=-1
        )
        rf_clf.fit(X_train, y_train_clf)
        y_pred_clf = rf_clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test_clf, y_pred_clf)
        
        mlflow.log_metric("classification_auc", auc)
        mlflow.sklearn.log_model(rf_clf, "random_forest_classifier")
        
        # Feature importance (топ-10)
        importances = rf_clf.feature_importances_
        top_indices = importances.argsort()[-10:][::-1]
        for i in top_indices[:10]:
            feature_name = X.columns[i][:40]  # ограничиваем длину имени
            mlflow.log_metric(f"feature_imp_{feature_name}", importances[i])
        
        print(f"\nModels logged to MLflow")
        print(f"   Gold table version: {gold_version}")
        print(f"   Run ID: {run.info.run_id}")
        print(f"   MAE: {mae:.2f} minutes")
        print(f"   AUC: {auc:.3f}")
        print("\n   View with: mlflow ui --backend-store-uri file:./mlflow_local")

if __name__ == "__main__":
    train_models()
