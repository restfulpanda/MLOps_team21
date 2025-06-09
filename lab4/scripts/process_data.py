import pandas as pd
import os

data_path = 'lab4/data/titanic.csv'

if not os.path.exists(data_path):
    print(f"Файл не найден по пути: {data_path}")
else:
    df = pd.read_csv(data_path)

    df_processed = df[['Pclass', 'Sex', 'Age']].copy()

    # Заполняем пропуски в 'Age'
    age_mean = df_processed['Age'].mean()
    df_processed['Age'].fillna(age_mean, inplace=True)
    df_processed['Age'] = df_processed['Age'].astype(int)

    # Применяем One-Hot к 'Sex'
    df_processed = pd.get_dummies(df_processed, columns=['Sex'], drop_first=True)

    # Перезаписываем
    df_processed.to_csv(data_path, index=False)

    print(f"Датасет обработан: добавлен One-Hot Encoding для 'Sex'.")