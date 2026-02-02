# Указываем официальный образ Python
FROM python:3.10-slim

# Устанавливаем рабочую папку внутри контейнера
WORKDIR /app

# Копируем файл с зависимостями
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем все файлы проекта в рабочую папку
COPY . .

# Указываем команду, которая запустит бота при старте контейнера
CMD ["python", "signal_bot.py"]
