import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

def preprocess_data(input_path, output_path):
    df = pd.read_csv(input_path)

    # Отделяем признаки и переменную-таргет
    X = df.drop("target", axis=1)
    y = df["target"]

    # Масштабируем признаки
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Обратно скаладыаем признаки и тарегет
    df_processed = pd.DataFrame(X_scaled, columns=X.columns)
    df_processed["target"] = y

    # Сохраняем
    df_processed.to_csv(output_path, index=False)

def main():
    preprocess_data("train/data.csv", "train/data_preprocessed.csv")
    preprocess_data("test/data.csv", "test/data_preprocessed.csv")

if __name__ == "__main__":
    main()
    print("Data preprocessing done.")
