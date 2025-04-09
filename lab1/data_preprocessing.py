import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

def preprocess_data():
    # загрузка данных
    train_df = pd.read_csv("train/train_data.csv")
    test_df = pd.read_csv("test/test_data.csv")

    # разделим прзнаки и целевую переменную
    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"]

    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]

    # масштабируем признаки
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # сохраним обратно
    pd.DataFrame(X_train_scaled, columns=X_train.columns).assign(target=y_train).to_csv("train/preprocessed_train.csv", index=False)
    pd.DataFrame(X_test_scaled, columns=X_test.columns).assign(target=y_test).to_csv("test/preprocessed_test.csv", index=False)

    print("Данные успешно предобработаны и сохранены.")

if __name__ == "__main__":
    preprocess_data()
