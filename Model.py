import nltk
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('tweet_emotions.csv')

# Загружаем нужные данные и целевую переменную
messages = data['content']  # список сообщений
emotions = data['sentiment']  # список эмоций, соответствующих сообщениям

# Инициализируем TweetTokenizer и загружаем стоп-слова
nltk.download('wordnet')
tokenizer = TweetTokenizer()
nltk.download('stopwords')
stopwords = set(stopwords.words("english"))

# Преобразуем сообщения
lemmatizer = WordNetLemmatizer()
processed_messages = []
for message in messages:
    # Разбиваем сообщение на токены
    tokens = tokenizer.tokenize(message)
    # Удаляем запятые и точки
    tokens = [token.replace(',', '').replace('.', '') for token in tokens]
    # Приводим все слова к нижнему регистру и лемматизируем
    tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens]
    # Удаляем стоп-слова
    tokens = [token for token in tokens if token not in stopwords]
    # Объединяем токены обратно в строку
    processed_message = ' '.join(tokens)
    processed_messages.append(processed_message)

# Векторизуем сообщения
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(processed_messages)
y = emotions

# Разбиваем данные на тренировочные и тестовые
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучаем модель методом случайный лес
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Выводим предсказание для тестовых данных
predictions = model.predict(X_test)
print(predictions)
