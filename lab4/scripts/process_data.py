import pandas as pd
import os

data_path = 'lab4/data/titanic.csv'

if not os.path.exists(data_path):
    print(f"Файл не найден по пути: {data_path}")
else:
    # Загружаем датасет
    df = pd.read_csv(data_path)

    # Выбираем только необходимые колонки
    df_processed = df[['Pclass', 'Sex', 'Age']].copy()

    # Перезаписываем
    df_processed.to_csv(data_path, index=False)

    print(f"Датасет обработан. В файле '{data_path}' оставлены только колонки Pclass, Sex, Age.")