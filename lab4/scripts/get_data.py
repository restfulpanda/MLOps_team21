import pandas as pd
from catboost.datasets import titanic
import os

os.makedirs('lab4/data', exist_ok=True)

# Загружаем датасет
titanic_df = titanic()[0]

# Сохраняем в CSV
titanic_df.to_csv('lab4/data/titanic.csv', index=False)

print("Датасет 'titanic.csv' успешно создан и сохранен в папке 'lab4/data'.")