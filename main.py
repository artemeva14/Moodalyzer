import nltk
import telebot
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from telebot import types

import Model
from Token import t
bot = telebot.TeleBot(t)


# Обработчик команды '/start'
@bot.message_handler(commands=['start'])
def start(message):
    keyboard_menu = types.ReplyKeyboardMarkup(resize_keyboard=True)
    button1 = types.KeyboardButton("Распознать эмоцию")
    button2 = types.KeyboardButton("Закончить сеанс")
    keyboard_menu.add(button1, button2)
    bot.send_message(message.chat.id, 'Привет! Я бот, который умеет распознавать эмоции по текстовому сообщению', reply_markup=keyboard_menu)


#Обработчик текстовых сообщений
@bot.message_handler(content_types='text')
def message_reply(message):
    if message.text == "Распознать эмоцию" or message.text == "Распознать еще одну эмоцию":
        msg = bot.send_message(message.chat.id, 'Введите имя игрока, которого сейчас добавляете')
        bot.register_next_step_handler(msg, give_emotion)
    if message.text == "Закончить сеанс":
        game_is_end(message)
    if message.text == "Вернуться в меню":
        keyboard_menu = types.ReplyKeyboardMarkup(resize_keyboard=True)
        button1 = types.KeyboardButton("Распознать еще одну эмоцию")
        button2 = types.KeyboardButton("Закончить сеанс")
        keyboard_menu.add(button1, button2)
        bot.send_message(message.chat.id, "Выберети дальнейшее действие", reply_markup=keyboard_menu)


def give_emotion(user_message):
    from nltk.corpus import stopwords
    message = user_message
    # Инициализируем TweetTokenizer и загружаем стоп-слова
    nltk.download('wordnet')
    tokenizer = TweetTokenizer()
    nltk.download('stopwords')
    stopwords = set(stopwords.words("english"))

    # Преобразуем сообщения
    lemmatizer = WordNetLemmatizer()
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

    # Векторизуем сообщения
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(processed_message)
    predictions = Model.model.predict(X)
    print(predictions)
    bot.send_message(user_message.chat.id, predictions)

def game_is_end(user_message):
    bot.send_message(user_message.chat.id, 'Ждем тебя снова!')
    bot.send_photo(user_message.chat.id, open('end_photo.jpg', 'rb'))


# Запуск бота
bot.infinity_polling()
