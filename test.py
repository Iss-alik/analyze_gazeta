from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from collections import Counter
from itertools import islice
import os


def filter_tokens(tokens, stop_words, punctuation):
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [word for word in tokens if word not in punctuation]
    tokens = [stemmer.stem(word) for word in tokens]
    tokens = [x for x in tokens if len(x) > 2]
    return tokens 

stemmer = SnowballStemmer("russian")
stop_words = set(stopwords.words('russian'))
punctuation = [',', '!', '?', '.', ':', ';', '-', '—', '_', '«', ')', '(', '»', '©']
key_words = ['немец', 'гитлеровцы', 'месть', 'ненависть', 'училище', "офицеры", "Что"]

path_to_dir = "recovered_text_files/"
for file in os.listdir(path_to_dir):
    document = open(f"{path_to_dir}/{file}", 'r')

    text = document.read().lower()
    document.close()

    tokens = word_tokenize(text)
    #filtered_tokens = filter_tokens(tokens, stop_words, punctuation)
    filtered_tokens = [x for x in tokens if len(x) > 2]
    text = ' '.join([word for word in filtered_tokens])

    words = Counter(filtered_tokens)
    dic = {}
    for key, value in words.items():
        dic[key] = value

    # Сортируем словарь по значению в убывающем порядке
    sorted_dict = dict(sorted(dic.items(), key=lambda item: item[1], reverse=True))

    x = 20  # количество элементов, которые нужно вывести

    # Используем islice для взятия первых x элементов
    first_x_elements = dict(islice(sorted_dict.items(), x))

    print(first_x_elements)

    for key in key_words:
        value = dic.get(key)  # используем get, чтобы избежать ошибки
        if value is not None:
            print(f"Ключ '{key}' имеет значение: {value}")
