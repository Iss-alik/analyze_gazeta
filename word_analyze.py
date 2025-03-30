import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from instructions import stop_words, key_words
from nltk.tokenize import word_tokenize
from pymystem3 import Mystem

# Функция для очистки текста
def clean_text(text):
    # Удаляем все, кроме русских букв, цифр и пробелов
    text = re.sub(r'[^А-Яа-я\s]', '', text)
    # Преобразуем текст в нижний регистр
    text = text.lower()

    # Удаляем стоп-слова
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    text = " ".join(tokens)

    m = Mystem()
    lemma_text = ''.join(m.lemmatize(text))

    # Удаляем стоп-слова
    tokens = word_tokenize(lemma_text)
    tokens = [word for word in tokens if word not in stop_words]
    clean_text = " ".join(tokens)


    return clean_text

# Функция для чтения всех текстовых файлов из папки
def read_documents_from_folder(folder_path):
    documents = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith(".txt"):
            with open(file_path, "r", encoding= 'utf8', errors= 'ignore') as file:
                text = file.read()
                cleaned_text = clean_text(text)
                documents.append(cleaned_text)
    return documents

# Путь к папке с текстовыми файлами
folder_path = "text_files/recovered/"

# Чтение и очистка документов
documents = read_documents_from_folder(folder_path)

#vectorizer = TfidfVectorizer()
#X = vectorizer.fit_transform(documents)

# Получение слов с низким TF-IDF
#tfidf_scores = zip(vectorizer.get_feature_names_out(), X.sum(axis=0).A1)
#low_tfidf_words = [word for word, score in tfidf_scores if score < 0.1]

#print(low_tfidf_words)

from collections import Counter


# Токенизация всех слов
all_words = [word for doc in documents for word in doc.split()]

# Подсчёт частотности
word_freq = Counter(all_words)

m = Mystem()
lemma_key_words = m.lemmatize(' '.join(key_words))

for key_word in lemma_key_words:
    if key_word  !=  " ":
        print(key_word + " - " + str(word_freq[key_word]))

# Вывод самых частых слов
print(word_freq.most_common(30))
