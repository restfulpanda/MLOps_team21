# lab4/scripts/process_data.py
import pandas as pd
import os

data_path = 'lab4/data/titanic.csv'

# Загружаем датасет
df = pd.read_csv(data_path)

# Выбираем только необходимые колонки: Pclass, Sex, Age
df_processed = df[['Pclass', 'Sex', 'Age']].copy()

# Перезаписываем исходный файл обработанными данными
df_processed.to_csv(data_path, index=False)

print(f"Датасет обработан. В файле '{data_path}' оставлены только колонки Pclass, Sex, Age.")