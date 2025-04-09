import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
import os

def train_model():
    # считываем данные
    df_train = pd.read_csv("train/data_preprocessed.csv")

    # Разделяем их
    X_train = df_train.drop("target", axis=1)
    y_train = df_train["target"]

    # Испольуем логистическую регрессию
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Сохраним всё
    if not os.path.exists("model"):
        os.makedirs("model")

    with open("model/model.pkl", "wb") as f:
        pickle.dump(model, f)

def main():
    train_model()

if __name__ == "__main__":
    main()
    print("Model training done.")
