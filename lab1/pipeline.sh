#!/bin/bash

pip install -r requirements.txt

# генерируем
python3 data_creation.py

# предобрабатываем
python3 data_preprocessing.py

# обучаем
python3 model_preparation.py

# тестируем
python3 model_testing.py
