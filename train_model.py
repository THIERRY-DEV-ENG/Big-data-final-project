# train_model.py
# Train two models to predict CPI (inflation) from other economic indicators

import pandas as pd
import sqlalchemy
import os
import json
import logging
import joblib
from dotenv import load_dotenv

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

FEATURES = ["unemployment", "fed_funds", "consumer_sentiment"]
TARGET = "cpi"


def load_data():
    """Load cleaned data from PostgreSQL."""
    url = (
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    engine = sqlalchemy.create_engine(url)
    df = pd.read_sql("SELECT * FROM cleaned_economic_data ORDER BY date", engine)
    logger.info(f"Loaded {len(df)} rows from database")
    return df


def prepare_features(df):
    """Drop rows missing consumer_sentiment - only for training, not the stored data."""
    df_ml = df[FEATURES + [TARGET]].dropna()
    logger.info(f"Rows available for training after dropping NaN: {len(df_ml)} (of {len(df)} total)")
    return df_ml[FEATURES], df_ml[TARGET]


def train_and_evaluate():
    df = load_data()
    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.info(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")

    results = {}

    # Model 1: Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    pred_test = lr.predict(X_test)
    pred_train = lr.predict(X_train)
    results["linear_regression"] = {
        "model": lr,
        "mae": mean_absolute_error(y_test, pred_test),
        "rmse": root_mean_squared_error(y_test, pred_test),
        "r2_test": r2_score(y_test, pred_test),
        "r2_train": r2_score(y_train, pred_train),
    }

    # Model 2: Random Forest
    rf = RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42)
    rf.fit(X_train, y_train)
    pred_test = rf.predict(X_test)
    pred_train = rf.predict(X_train)
    results["random_forest"] = {
        "model": rf,
        "mae": mean_absolute_error(y_test, pred_test),
        "rmse": root_mean_squared_error(y_test, pred_test),
        "r2_test": r2_score(y_test, pred_test),
        "r2_train": r2_score(y_train, pred_train),
    }

    # Print comparison
    print("\n" + "=" * 55)
    print("MODEL COMPARISON: Predicting CPI (inflation)")
    print("=" * 55)
    for name, r in results.items():
        gap = r["r2_train"] - r["r2_test"]
        print(f"\n{name.upper()}")
        print(f"  MAE:  {r['mae']:.3f}")
        print(f"  RMSE: {r['rmse']:.3f}")
        print(f"  R2 (test):  {r['r2_test']:.3f}")
        print(f"  R2 (train): {r['r2_train']:.3f}")
        print(f"  Overfit gap: {gap:.3f} {'WARNING - overfitting' if gap > 0.1 else 'OK'}")

    winner = "random_forest" if results["random_forest"]["r2_test"] > results["linear_regression"]["r2_test"] else "linear_regression"
    print(f"\nBetter model: {winner} (higher test R2)")
    print("=" * 55)

    # Check which feature Random Forest is relying on most - investigate the perfect R2
    importances = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("\nRandom Forest feature importances:")
    print(importances)
    print("=" * 55)
    # Overfitting demonstration: compare different max_depth values
    print("\nOverfitting check across different max_depth values:")
    for depth in [2, 4, 6, None]:
        rf_test = RandomForestRegressor(n_estimators=100, max_depth=depth, random_state=42)
        rf_test.fit(X_train, y_train)
        r2_tr = r2_score(y_train, rf_test.predict(X_train))
        r2_te = r2_score(y_test, rf_test.predict(X_test))
        print(f"  max_depth={depth}: train R2={r2_tr:.3f}  test R2={r2_te:.3f}  gap={r2_tr - r2_te:.3f}")

    # Save both models + metrics
    os.makedirs("models", exist_ok=True)
    for name, r in results.items():
        joblib.dump(r["model"], f"models/{name}.pkl")
        with open(f"models/{name}_metrics.json", "w") as f:
            json.dump({
                "mae": r["mae"], "rmse": r["rmse"],
                "r2_test": r["r2_test"], "r2_train": r["r2_train"],
            }, f, indent=2)
    logger.info("Models and metrics saved to models/")

    # Save predictions for Tableau (one required chart: predictions vs actual)
    df_preds = X_test.copy()
    df_preds["actual_cpi"] = y_test.values
    df_preds["predicted_cpi_rf"] = results["random_forest"]["model"].predict(X_test)
    df_preds["predicted_cpi_lr"] = results["linear_regression"]["model"].predict(X_test)
    os.makedirs("data", exist_ok=True)
    df_preds.to_csv("data/predictions.csv", index=False)
    logger.info("Predictions saved to data/predictions.csv")

    return results


if __name__ == "__main__":
    train_and_evaluate()