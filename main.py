import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, mean_absolute_error
import nltk
from nltk.corpus import stopwords
import re
import xgboost as xgb
import joblib
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier


# Укажите путь к каталогу с данными
data_dir = r'C:\Users\Алексей\Documents\GitHub\Class_comment_films\aclImdb_v1\aclImdb'

# Функция для чтения отзывов из файлов
def load_data_from_dir(directory):
    """
    Загрузка данных из директории с отзывами и их соответствующими метками и рейтингами.

    Функция проходит по директории `directory`, в которой должны находиться две подпапки: 
    'pos' (положительные отзывы) и 'neg' (отрицательные отзывы). Каждая подпапка содержит 
    текстовые файлы отзывов с именами, включающими идентификатор и рейтинг, например:
    '123_7.txt' (где 123 — идентификатор, а 7 — рейтинг от 1 до 10). 

    Параметры:
    ----------
    directory : str
        Путь к корневой директории, содержащей поддиректории 'pos' и 'neg'.

    Возвращаемые значения:
    ----------------------
    data : list of str
        Список текстов отзывов.
    labels : list of int
        Список меток (1 для положительных отзывов, 0 для отрицательных).
    ratings : list of int
        Список оценок отзывов по шкале от 1 до 10.

    Пример:
    -------
    >>> data, labels, ratings = load_data_from_dir("aclImdb/train")
    >>> print(data[0])  # Вывод текста первого отзыва
    >>> print(labels[0])  # Метка первого отзыва (1 или 0)
    >>> print(ratings[0])  # Рейтинг первого отзыва (от 1 до 10)
    """
    data = []
    labels = []
    ratings = []
    for label_type in ['pos', 'neg']:
        dir_name = os.path.join(directory, label_type)
        for fname in os.listdir(dir_name):
            if fname.endswith('.txt'):
                # Извлечение рейтинга из имени файла, предполагается формат "something_rating.txt"
                try:
                    # Пример имени файла: "123_7.txt" -> rating = 7
                    rating = int(fname.split('_')[1].split('.')[0])
                except (IndexError, ValueError):
                    rating = 5  # Средний рейтинг, если не удалось извлечь
                with open(os.path.join(dir_name, fname), encoding='utf-8') as f:
                    data.append(f.read())
                labels.append(1 if label_type == 'pos' else 0)
                ratings.append(rating)
    return data, labels, ratings

# Загрузка обучающих данных
train_data, train_labels, train_ratings = load_data_from_dir(os.path.join(data_dir, 'train'))

# Загрузка тестовых данных
test_data, test_labels, test_ratings = load_data_from_dir(os.path.join(data_dir, 'test'))

# Загрузка стоп-слов и инициализация лемматизатора
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Функция предобработки текста
def preprocess_text(text):

    """
    Предобработка текста отзыва.

    Функция выполняет следующие этапы предобработки:
    1. Удаляет HTML-теги из текста.
    2. Удаляет все небуквенные символы и цифры, оставляя только буквы.
    3. Приводит текст к нижнему регистру.
    4. Токенизирует текст (разделяет на слова).
    5. Удаляет стоп-слова и выполняет лемматизацию (приведение к базовой форме слова).
    6. Объединяет обработанные слова обратно в строку.

    Параметры:
    ----------
    text : str
        Исходный текст отзыва, который необходимо предобработать.

    Возвращаемое значение:
    ----------------------
    str
        Предобработанный текст отзыва.

    Пример:
    -------
    >>> raw_text = "<div>Это пример текста с HTML-тегами и числами: 123!</div>"
    >>> preprocess_text(raw_text)
    'example text html tag number'
    """
    # Удаление HTML-тегов
    text = re.sub(r'<.*?>', '', text)
    # Удаление небуквенных символов и цифр
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # Приведение к нижнему регистру
    text = text.lower()
    # Токенизация
    tokens = text.split()
    # Удаление стоп-слов и лемматизация
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    # Объединение обратно в строку
    text = ' '.join(tokens)
    return text

# Предобработка обучающих данных
train_data = [preprocess_text(review) for review in train_data]

# Предобработка тестовых данных
test_data = [preprocess_text(review) for review in test_data]

# Преобразование текстов в числовые векторы
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_data)
X_test = vectorizer.transform(test_data)

# ---- Реализация моделей классификации ----

# 1. Логистическая регрессия
print("\n=== Логистическая регрессия ===")
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, train_labels)

# Предсказание логистической регрессии
lr_predictions = lr_model.predict(X_test)
print(classification_report(test_labels, lr_predictions))
lr_accuracy = accuracy_score(test_labels, lr_predictions)
print(f'Точность логистической регрессии: {lr_accuracy * 100:.2f}%')


# 2. XGBoost
print("\n=== XGBoost ===")
# XGBoost работает напрямую с плотными матрицами (numpy array), поэтому конвертируем данные
X_train_xgb = X_train.toarray()
X_test_xgb = X_test.toarray()

xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    max_depth=6,
    n_estimators=500,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss'
)

xgb_model.fit(X_train_xgb, train_labels)
xgb_predictions = xgb_model.predict(X_test_xgb)
print(classification_report(test_labels, xgb_predictions))
xgb_accuracy = accuracy_score(test_labels, xgb_predictions)
print(f'Точность XGBoost: {xgb_accuracy * 100:.2f}%')


# 3. Random Forest
print("\n=== Random Forest ===")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, train_labels)

# Предсказание Random Forest
rf_predictions = rf_model.predict(X_test)
print(classification_report(test_labels, rf_predictions))
rf_accuracy = accuracy_score(test_labels, rf_predictions)
print(f'Точность Random Forest: {rf_accuracy * 100:.2f}%')


# 4. Gradient Boosting
print("\n=== Gradient Boosting ===")
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
gb_model.fit(X_train, train_labels)

# Предсказание Gradient Boosting
gb_predictions = gb_model.predict(X_test)
print(classification_report(test_labels, gb_predictions))
gb_accuracy = accuracy_score(test_labels, gb_predictions)
print(f'Точность Gradient Boosting: {gb_accuracy * 100:.2f}%')


# 5. LightGBM
print("\n=== LightGBM ===")
lgbm_model = LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
lgbm_model.fit(X_train, train_labels)

# Предсказание LightGBM
lgbm_predictions = lgbm_model.predict(X_test)
print(classification_report(test_labels, lgbm_predictions))
lgbm_accuracy = accuracy_score(test_labels, lgbm_predictions)
print(f'Точность LightGBM: {lgbm_accuracy * 100:.2f}%')


# ---- Реализация Voting Classifier ----
print("\n=== Voting Classifier ===")

# Определение базовых моделей для голосования
voting_clf = VotingClassifier(
    estimators=[
        ('lr', lr_model),
        ('xgb', xgb_model),
        ('rf', rf_model),
        ('gb', gb_model),
        ('lgbm', lgbm_model)
    ],
    voting='soft'  # 'hard' для жёсткого голосования
)

# Обучение Voting Classifier
voting_clf.fit(X_train, train_labels)

# Предсказание Voting Classifier
voting_predictions = voting_clf.predict(X_test)
print(classification_report(test_labels, voting_predictions))
voting_accuracy = accuracy_score(test_labels, voting_predictions)
print(f'Точность Voting Classifier: {voting_accuracy * 100:.2f}%')


# ---- Реализация модели регрессии ----

# Random Forest Regressor
print("\n=== Random Forest Regressor ===")
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, train_ratings)

# Предсказание Random Forest Regressor
rf_rating_predictions = rf_regressor.predict(X_test)
rf_rating_predictions = np.clip(np.round(rf_rating_predictions), 1, 10)
rf_mae = mean_absolute_error(test_ratings, rf_rating_predictions)
print(f'Random Forest Regressor - MAE: {rf_mae:.2f}')

# ---- Сохранение моделей ----
print("\n=== Сохранение моделей ===")

# Путь к директории models
model_dir = r'C:\Users\Алексей\Documents\GitHub\Class_comment_films\movie_review_project\models'

# Убедитесь, что директория существует
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    print(f"Создана директория: {model_dir}")
else:
    print(f"Директория уже существует: {model_dir}")

# Сохранение моделей классификации
joblib.dump(lr_model, os.path.join(model_dir, 'sentiment_classifier.pkl'))
print("Сохранена модель логистической регрессии.")

joblib.dump(xgb_model, os.path.join(model_dir, 'xgb_sentiment_classifier.pkl'))
print("Сохранена модель XGBoost.")

joblib.dump(rf_model, os.path.join(model_dir, 'rf_sentiment_classifier.pkl'))
print("Сохранена модель Random Forest.")

joblib.dump(gb_model, os.path.join(model_dir, 'gb_sentiment_classifier.pkl'))
print("Сохранена модель Gradient Boosting.")

joblib.dump(lgbm_model, os.path.join(model_dir, 'lgbm_sentiment_classifier.pkl'))
print("Сохранена модель LightGBM.")

joblib.dump(voting_clf, os.path.join(model_dir, 'voting_classifier.pkl'))
print("Сохранена модель Voting Classifier.")

# Сохранение моделей регрессии
joblib.dump(rf_regressor, os.path.join(model_dir, 'rf_rating_regressor.pkl'))
print("Сохранена модель Random Forest Regressor.")

# Сохранение векторизатора
joblib.dump(vectorizer, os.path.join(model_dir, 'tfidf_vectorizer.pkl'))
print("Сохранён векторизатор TF-IDF.")
