"""
save_model.py - Abalone Rings Regression
Run: python save_model.py
Output: model.joblib, feature_columns.joblib
"""

import joblib
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import numpy as np

FEATURE_COLS = [
    'Shell weight',
    'Height',
    'Diameter',
    'Volume',
    'Length',
    'Whole weight',
    'Whole weight.2',
    'Meat_weight',
    'Whole weight.1',
    'Sex',
]
TARGET = 'Rings'


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Sex encoding
    sex_map = {'M': 0, 'F': 1, 'I': 2}
    df['Sex'] = df['Sex'].map(sex_map)
    # Engineered features
    df['Volume']      = df['Length'] * df['Diameter'] * df['Height']
    df['Meat_weight'] = df['Whole weight'] - df['Shell weight']
    df['Shell_ratio'] = df['Shell weight'] / (df['Whole weight'] + 1)
    df['Density']     = df['Whole weight'] / (df['Volume'] + 1)
    return df


def main():
    print("Veri yükleniyor...")
    train = pd.read_csv("train.csv")
    print(f"Train boyutu: {train.shape}")

    df = feature_engineering(train)

    X = df[FEATURE_COLS].astype(float)
    y = df[TARGET].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("LightGBM eğitiliyor...")
    model = LGBMRegressor(n_jobs=-1, random_state=42, verbose=-1)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae  = mean_absolute_error(y_test, preds)
    r2   = r2_score(y_test, preds)

    print(f"\nRMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"R²   : {r2:.4f}")

    joblib.dump(model, "model.joblib")
    joblib.dump(FEATURE_COLS, "feature_columns.joblib")
    print("\n✅ model.joblib ve feature_columns.joblib kaydedildi.")


if __name__ == "__main__":
    main()