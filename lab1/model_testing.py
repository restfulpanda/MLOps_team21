import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
import os

def test_model():
    # загружаем модель
    with open("model/model.pkl", "rb") as f:
        model = pickle.load(f)

    # считываем данные
    df_test = pd.read_csv("test/data_preprocessed.csv")
    X_test = df_test.drop("target", axis=1)
    y_test = df_test["target"]

    # предсказываем
    y_pred = model.predict(X_test)

    # рассчитываем метрику
    acc = accuracy_score(y_test, y_pred)

    print(f"Test accuracy: {acc:.3f}")

def main():
    test_model()

if __name__ == "__main__":
    main()
    print("Model testing done.")
