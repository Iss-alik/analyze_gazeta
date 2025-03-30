from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import numpy as np

import gensim
import gensim.corpora as corpora
from gensim.corpora import Dictionary

from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel

import matplotlib.pyplot as plt
import pandas as pd
import os, logic  


# Функция для чтения всех текстовых файлов из папки
def read_documents_from_folder(folder_path):
    documents = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith(".txt"):
            with open(file_path, "r", encoding = 'utf8', errors='replace') as file:
                text = file.read()
                cleaned_text = logic.clean_text(text)
                documents.append(cleaned_text)
    return documents

def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = LdaModel(corpus=corpus, num_topics=num_topics, id2word=id2word)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values

def preprocess(doc):
    return [word for word in doc.split() if word.isalpha()]

# Пример документов
documents = [
    "Это пример текста, используемый для моделирования тем.",
    "Моделирование тем полезно для анализа больших текстовых массивов.",
    "TF-IDF помогает выделять значимые слова из текста.",
    "Этот текст используется как пример для топик моделинга.",
    "Топик моделинг помогает выявить скрытые темы в текстах."
]

# Шаг 1: Преобразование текста в матрицу TF-IDF
vectorizer = TfidfVectorizer(
    max_features=1000,  # Максимальное количество уникальных слов
    ngram_range=(1, 1)  # Учитываем только отдельные слова (униграммы)
)
tfidf_matrix = vectorizer.fit_transform(documents)

# Шаг 2: Применение NMF для топик моделинга
num_topics = 3  # Количество тем
nmf_model = NMF(n_components=num_topics, random_state=42)
W = nmf_model.fit_transform(tfidf_matrix)  # Матрица документов и тем
H = nmf_model.components_  # Матрица тем и слов

# Шаг 3: Вывод ключевых слов для каждой темы
feature_names = vectorizer.get_feature_names_out()
for topic_idx, topic in enumerate(H):
    print(f"Тема {topic_idx + 1}:")
    top_keywords = [feature_names[i] for i in topic.argsort()[:-6:-1]]  # Топ-5 слов
    print(", ".join(top_keywords))
    print()

# Шаг 4: Распределение тем по документам
for doc_idx, topic_dist in enumerate(W):
    print(f"Документ {doc_idx + 1}:")
    top_topic_idx = np.argmax(topic_dist)
    print(f"Основная тема: Тема {top_topic_idx + 1}")
    print()
