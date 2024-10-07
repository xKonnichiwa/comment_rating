from django.shortcuts import render
from .forms import ReviewForm
from .models import Review
import joblib
import numpy as np

# Загрузка модели и векторизатора
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, 'models')

sentiment_model = joblib.load(os.path.join(model_path, 'voting_classifier.pkl'))
rating_model = joblib.load(os.path.join(model_path, 'rf_rating_regressor.pkl'))
vectorizer = joblib.load(os.path.join(model_path, 'tfidf_vectorizer.pkl'))

def predict_sentiment_and_rating(review_text):

    """
    Предсказание настроения и рейтинга для отзыва о фильме.

    Функция выполняет следующие шаги:
    1. Предобрабатывает входной текст отзыва с помощью внутренней функции `preprocess_text`.
    2. Преобразует текст в числовой вектор с использованием ранее обученного векторизатора (TF-IDF).
    3. Использует предобученную модель `sentiment_model` для предсказания настроения (положительный или отрицательный).
    4. Применяет предобученную модель `rating_model` для оценки рейтинга по шкале от 1 до 10.

    Параметры:
    ----------
    review_text : str
        Исходный текст отзыва, который необходимо проанализировать.

    Возвращаемые значения:
    ----------------------
    sentiment : str
        Предсказанный статус настроения: "Положительный" или "Отрицательный".
    rating : int
        Предсказанный рейтинг отзыва по шкале от 1 до 10.

    Пример:
    -------
    >>> review_text = "This movie was fantastic! The acting was brilliant."
    >>> predict_sentiment_and_rating(review_text)
    ('Положительный', 9)
    """
    # Предобработка текста (используем ту же функцию, что и при обучении)
    def preprocess_text(text):

        """
        Предобработка текста отзыва: удаление HTML-тегов, приведение к нижнему регистру, удаление стоп-слов и лемматизация.

        Параметры:
        ----------
        text : str
            Исходный текст, который необходимо предобработать.

        Возвращаемое значение:
        ----------------------
        str
            Предобработанный текст.
        """
        import re
        import nltk
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        

        nltk.download('stopwords')
        nltk.download('wordnet')
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()

        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^a-zA-Z]', ' ', text)
        text = text.lower()
        tokens = text.split()
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
        text = ' '.join(tokens)
        return text

    processed_text = preprocess_text(review_text)
    vectorized_text = vectorizer.transform([processed_text])

    # Предсказание настроения
    sentiment = sentiment_model.predict(vectorized_text)
    sentiment = 'Положительный' if sentiment[0] == 1 else 'Отрицательный'

    # Предсказание рейтинга
    rating = rating_model.predict(vectorized_text)
    rating = int(np.clip(np.round(rating), 1, 10)[0])

    return sentiment, rating

def review_view(request):

    """
    Обработка формы отзыва и предсказание настроения и рейтинга для введённого текста.

    Функция принимает HTTP-запрос, обрабатывает форму с текстом отзыва, выполняет предсказание 
    с помощью моделей настроения и рейтинга, а затем сохраняет результат в базу данных и отображает 
    результаты на соответствующей странице.

    Параметры:
    ----------
    request : HttpRequest
        Объект HTTP-запроса, содержащий данные, отправленные пользователем (POST) или пустой запрос (GET).

    Логика работы:
    --------------
    1. Если запрос методом POST, проверяется, валидна ли форма `ReviewForm`.
    2. Если форма валидна, создаётся экземпляр модели `Review` на основе введённого текста.
    3. С помощью функции `predict_sentiment_and_rating` определяются предсказанные значения настроения и рейтинга.
    4. Эти значения записываются в экземпляр модели `Review`, и он сохраняется в базу данных.
    5. Пользователю отображается результат предсказания на странице `result.html`.
    6. Если запрос методом GET, возвращается пустая форма для ввода нового отзыва.

    Возвращаемое значение:
    ----------------------
    HttpResponse
        Сформированная HTML-страница с результатами предсказания (в случае POST) или форма для ввода отзыва (в случае GET).

    Пример использования:
    ---------------------
    При отправке формы с отзывом на странице:
    http://127.0.0.1:8000/

    POST-запрос:
    ------------
    Данные формы: {'text': 'The movie was great, with excellent acting and thrilling story!'}
    
    Пример предсказанного результата:
    ---------------------------------
    Отзыв: 'The movie was great, with excellent acting and thrilling story!'
    Предсказанное настроение: Положительный
    Предсказанный рейтинг: 8 из 10

    GET-запрос:
    -----------
    Пустая форма отображается на странице для ввода нового отзыва.
    """
    if request.method == 'POST':
        form = ReviewForm(request.POST)
        if form.is_valid():
            review_instance = form.save(commit=False)
            sentiment, rating = predict_sentiment_and_rating(review_instance.text)
            review_instance.predicted_sentiment = sentiment
            review_instance.predicted_rating = rating
            review_instance.save()
            return render(request, 'result.html', {'review': review_instance})
    else:
        form = ReviewForm()
    return render(request, 'index.html', {'form': form})
