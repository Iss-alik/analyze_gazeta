import nltk


# Делим текст на предложения 
def split_text_into_sentences(text):
    sentences = nltk.sent_tokenize(text)
    return sentences

# Соеденяем части в чанк
def parts_to_chunks(parts, number_of_chunks):
  numbers_of_parts = len(parts) # определяем сколько частей идет на вход 
  length_of_chunk = int(numbers_of_parts/number_of_chunks) # определяем сколько частей нужно укладывать в один чанк 
  chunks = [] # Лист чанков
  for x in range(number_of_chunks): # перебираем чанки
    chunk = " " # Пустой чанк 
    for y in range(length_of_chunk): # перебираем элементы чанка
      index = y + length_of_chunk * x # определяем какой индекс у элемента который нужно добавить в чанк 
      chunk += parts[index] # вставляем элемент в чанк
    chunks.append(chunk) # отправляем чанк в лист чанков
  return chunks

# Переобразовываем чанки в лист словарей с которым будет работать ИИ
def chunks_to_dictionary(chunks):
  document_chunked = [] # Создаем лист

  for chunk in chunks: # Перебираем чанки 
    dictinoary_of_chunk = {"data": {"text": chunk}} # Запихиваем чанк в словарь
    document_chunked.append(dictinoary_of_chunk) # Запихиваем словарь в лист
  return document_chunked 

# Переобразовываем части в словарь с которым будет работать ИИ
def parts_to_dictionary(parts, number_of_chunks = 8):
  chunks = parts_to_chunks(parts, number_of_chunks) # отправляем части для сбора чанков
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

def create_report(text):
  pass