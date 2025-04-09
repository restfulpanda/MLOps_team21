import os
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

def create_regression_data():
    # Сгенерируем данные 8 признаков, 1 целевая переменная
    X, y = make_regression(
        n_samples=1000,
        n_features=8,
        noise=15.0,          #немного шума
        random_state=42
    )

    # Разделим данные
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # создадим датафреймы
    train_df = pd.DataFrame(X_train, columns=[f"feature{i}" for i in range(1, 9)])
    train_df["target"] = y_train

    test_df = pd.DataFrame(X_test, columns=[f"feature{i}" for i in range(1, 9)])
    test_df["target"] = y_test

    # создадим папки
    os.makedirs("train", exist_ok=True)
    os.makedirs("test", exist_ok=True)

    # сохраним
    train_df.to_csv("train/train_data.csv", index=False)
    test_df.to_csv("test/test_data.csv", index=False)

    print("Данные успешно созданы и сохранены")

if __name__ == "__main__":
    create_regression_data()
