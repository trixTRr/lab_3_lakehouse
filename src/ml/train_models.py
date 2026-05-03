import mlflow
import mlflow.sklearn
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, roc_auc_score
import os
from deltalake import DeltaTable

def get_table_version(table_path):
    try:
        dt = DeltaTable(table_path)
        return dt.version()
    except:
        return None

def train_models():
    feature_path = "data/gold/feature_table"
    
    if not os.path.exists(feature_path):
        print("Feature table not found")
        return
    
    gold_version = get_table_version(feature_path)
    
    print("Loading feature table...")
    df = pl.read_delta(feature_path).to_pandas()
    
    if len(df) > 300000:
        df = df.sample(n=300000, random_state=42)
        print(f"Sampled to {len(df):,} rows")
    
    # Создаём целевую переменную
    df["is_delayed"] = (df["ARR_DELAY"] > 15).astype(int)
    
    X = df.drop(columns=["ARR_DELAY", "is_delayed"])
    y_reg = df["ARR_DELAY"]
    y_clf = df["is_delayed"]
    
    categorical_cols = ["OP_CARRIER", "ORIGIN", "DEST"]
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    X_train, X_test, y_train_reg, y_test_reg = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )
    _, _, y_train_clf, y_test_clf = train_test_split(
        X, y_clf, test_size=0.2, random_state=42
    )
    
    mlflow.set_tracking_uri("file:./mlflow_local")
    mlflow.set_experiment("flight_delay_prediction")
    
    with mlflow.start_run(run_name="flight_delay_models_with_predictions") as run:

        # Регрессия
        print("Training regression model...")
        rf_reg = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
        rf_reg.fit(X_train, y_train_reg)
        y_pred_reg = rf_reg.predict(X_test)
        mae = mean_absolute_error(y_test_reg, y_pred_reg)
        
        mlflow.log_metric("regression_mae", mae)
        mlflow.sklearn.log_model(rf_reg, "random_forest_regressor")
        
        # Классификация
        print("Training classification model...")
        rf_clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf_clf.fit(X_train, y_train_clf)
        y_pred_clf = rf_clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test_clf, y_pred_clf)
        
        mlflow.log_metric("classification_auc", auc)
        mlflow.sklearn.log_model(rf_clf, "random_forest_classifier")
        
        # Сохраняем прогнозы

        # Логируем параметры
        if gold_version is not None:
            mlflow.log_param("gold_table_version", gold_version)
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth_reg", 15)
        mlflow.log_param("max_depth_clf", 10)
        mlflow.log_param("sample_size", len(df))
        
        # Feature importance (топ-10)
        importances = rf_clf.feature_importances_
        top_indices = importances.argsort()[-10:][::-1]
        for i in top_indices[:10]:
            feature_name = X.columns[i][:40]  # ограничиваем длину имени
            mlflow.log_metric(f"feature_imp_{feature_name}", importances[i])
        # 1. Сохраняем прогнозы в CSV
        predictions_df = pd.DataFrame({
            'actual_delay': y_test_reg.values,
            'predicted_delay': y_pred_reg,
            'delay_probability': y_pred_clf,
            'is_delayed_actual': y_test_clf.values,
            'is_delayed_predicted': (y_pred_clf > 0.5).astype(int),
            'error': abs(y_test_reg.values - y_pred_reg)
        })
        
        # Сохраняем CSV
        predictions_csv = "predictions.csv"
        predictions_df.to_csv(predictions_csv, index=False)
        mlflow.log_artifact(predictions_csv)
        print(f"   Прогнозы сохранены как артефакт: {predictions_csv}")
        
        # 2. Сохраняем график: реальные vs предсказанные
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test_reg, y_pred_reg, alpha=0.5, s=10)
        plt.plot([y_test_reg.min(), y_test_reg.max()], 
                 [y_test_reg.min(), y_test_reg.max()], 
                 'r--', lw=2, label='Идеальное предсказание')
        plt.xlabel('Реальная задержка (минуты)')
        plt.ylabel('Предсказанная задержка (минуты)')
        plt.title('Регрессия: реальные vs предсказанные значения')
        plt.legend()
        plt.tight_layout()
        
        plot_path = "predictions_plot.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        print(f"   График сохранён как артефакт: {plot_path}")
        plt.close()
        
        # 3. Сохраняем гистограмму ошибок
        plt.figure(figsize=(10, 6))
        plt.hist(predictions_df['error'], bins=50, edgecolor='black')
        plt.xlabel('Ошибка предсказания (минуты)')
        plt.ylabel('Количество рейсов')
        plt.title(f'Распределение ошибок (MAE = {mae:.2f} мин)')
        plt.tight_layout()
        
        error_plot = "error_distribution.png"
        plt.savefig(error_plot)
        mlflow.log_artifact(error_plot)
        print(f"   Гистограмма ошибок сохранена: {error_plot}")
        plt.close()
        
        # 4. Сохраняем топ-10 самых ошибочных прогнозов
        worst_predictions = predictions_df.nlargest(10, 'error')
        worst_predictions.to_csv("worst_predictions.csv", index=False)
        mlflow.log_artifact("worst_predictions.csv")
        
        print(f"\nMLflow run completed!")
        print(f"   Run ID: {run.info.run_id}")
        print(f"   MAE: {mae:.2f}, AUC: {auc:.3f}")
        print(f"   Artifacts: predictions.csv, predictions_plot.png, error_distribution.png")

if __name__ == "__main__":
    train_models()