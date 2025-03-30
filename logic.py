import nltk
from nltk.tokenize import word_tokenize
from pymystem3 import Mystem
from instructions import stop_words
import re

# Делим текст на предложения 
def split_text_into_tokens(text):
    tokens = word_tokenize(text)
    return tokens

# Соеденяем части в чанк
def parts_to_chunks(parts, length_of_chunk = None, number_of_chunks = None): # один из двух length_of_chunk или number_of_chunks должен быть неизвестен
  numbers_of_parts = len(parts) # определяем сколько частей идет на вход 

  if(not length_of_chunk):# Если нам не извесна длина чанка
    length_of_chunk = int(numbers_of_parts/number_of_chunks) # определяем сколько частей нужно укладывать в один чанк 
  
  if(not number_of_chunks): # Если нам не известна количество чанка
    number_of_chunks = int(numbers_of_parts/length_of_chunk) + 1 

  chunks = [] # Лист чанков
  for x in range(number_of_chunks): # перебираем чанки
    chunk = parts[length_of_chunk * x : length_of_chunk * (x+1)]
    chunks.append(" ".join(chunk)) # отправляем чанк в лист чанков
  return chunks

# Переобразовываем чанки в лист словарей с которым будет работать ИИ
def chunks_to_dictionary(chunks):
  document_chunked = [] # Создаем лист

  for chunk in chunks: # Перебираем чанки 
    dictinoary_of_chunk = {"data": {"text": chunk}} # Запихиваем чанк в словарь
    document_chunked.append(dictinoary_of_chunk) # Запихиваем словарь в лист
  return document_chunked 

# Переобразовываем части в словарь с которым будет работать ИИ
def parts_to_dictionary(parts, length_of_chunk = None, number_of_chunks = None):
  chunks = parts_to_chunks(parts, length_of_chunk, number_of_chunks) # отправляем части для сбора чанков
  document_chunked = chunks_to_dictionary(chunks) # из чанков собираем словарь для ИИ
  return document_chunked # Возвращаем словарь для работы ИИ

# считаем количесто предложений 
def number_of_sentences(tokens):
  special_signs = ['.','!','?'] # знаки которые обозначают что предложение закончено
  all_special_signs = [token for token in tokens if token in special_signs] # лист со всеми знаками в тектсе 
  number_of_sentences = len(all_special_signs) # длина листа, равна количеству знаков
  return number_of_sentences

# считеам количество слогов
def number_of_syllables(tokens):
  number_of_syllables = 0 
  vowels = ['а', 'у' 'о' 'и', 'э', 'ы', 'я', 'ю' ,'е', 'ё'] # лист согласных

  # перебираем каждый токен
  for token in tokens:
    if len(token) > 1: # если он длинее чем один символ то берем в общий счет
      for symbol in token: # перебираем каждый символ в токене
        if symbol in vowels: # одна глассная ровна одному слогу 
          number_of_syllables += 1

  return number_of_syllables

# подсчет насколько текст тяжело читать
def evalute_complexity_of_text(tokens):
  total_syllables = number_of_syllables(tokens) # количество слогов
  total_sentences = number_of_sentences(tokens) # количество предложений 
  total_words = len([token for token in tokens if len(token) > 1]) # количество токенов чья длина больше одного 
  
  complexity = 206.835 - 1.015*(total_words/total_sentences) - 84.6*(total_syllables / total_words) # сложность чтения
  return complexity

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
    tokens = [word for word in tokens if word not in stop_words ]
    
    clean_text = " ".join(tokens)

    return clean_text

def create_report(text):
  pass