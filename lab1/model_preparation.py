import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

def train_model():
    # загрузка предобработанных данных
    df = pd.read_csv("train/preprocessed_train.csv")
    X = df.drop(columns=["target"])
    y = df["target"]

    # обучение модели
    model = LinearRegression()
    model.fit(X, y)

    # сохраним модель
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Модель обучена и сохранена")

if __name__ == "__main__":
    train_model()
