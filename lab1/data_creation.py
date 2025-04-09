import numpy as np
import pandas as pd
import os

def create_data():
    # Сгенерируем случайные данные
    X = np.random.randn(100, 3)  
    y = np.random.randint(0, 2, size=100)  # Два класса: 0 или 1

    df = pd.DataFrame(X, columns=["feature1", "feature2", "feature3"])
    df["target"] = y

    # Разделим их
    df_train = df.iloc[:80]
    df_test = df.iloc[80:]

    # Создадим для них папки
    if not os.path.exists("train"):
        os.makedirs("train")
    if not os.path.exists("test"):
        os.makedirs("test")

    # Сохраним
    df_train.to_csv("train/data.csv", index=False)
    df_test.to_csv("test/data.csv", index=False)

if __name__ == "__main__":
    create_data()
    print("Data creation done.")
