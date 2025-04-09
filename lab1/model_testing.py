import pandas as pd
import pickle
from sklearn.metrics import r2_score

def test_model():
    # щагрузка модели
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    # загрузка тестовых данных
    df = pd.read_csv("test/preprocessed_test.csv")
    X_test = df.drop(columns=["target"])
    y_test = df["target"]

    # предсказания
    y_pred = model.predict(X_test)

    # метрика
    r2 = r2_score(y_test, y_pred)

    print(f"Model test accuracy is: {r2:.4f}")

if __name__ == "__main__":
    test_model()
