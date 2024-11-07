import cohere
import nltk
import os
from logic import *
from recover import * 
import instructions

# Закомиченый код это для эксперементов 
co = cohere.ClientV2('AH0agiBtbhKCBApQrt9OscLj22VjX6JV0O9KBwrN')


path_to_dir = "./recovered_text_files"

# Сообщение которое дается для ИИ от системы (не от юзера)
system_message = """## Задание и контекст
Вы получите ряд фрагментов из выпуска газеты, которые вместе состовляют один выпуск газеты.\
Как помощник, вы должны генерировать ответы на запросы пользователя на основе информации, предоставленной во фрагментах.\
Убедитесь, что ваши ответы точны и правдивы, и что вы ссылаетесь на свои источники, где это уместно, чтобы ответить на запросы,\
независимо от их сложности. Так-же отвечайте на русском языке"""

#message = "Тебе будет предоставлен выпуск газеты и в ней есть несколько разных публикаций с разными темами. Посчитай количество публикации в газете и сколько из них приуствует ненависть/месть к немцам. Выведи соотношение "

# Перебираем достпные нам текстовый файлы
for file in os.listdir(path_to_dir):
  document = open(f"{path_to_dir}/{file}", 'r')

  text = document.read()
  document.seek(0)
 
  lines = document.readlines() # линии из текста
  document.close()

  #text = recover(text, file)

  # Создаем чанкизированй документ на основе деления по предложениям
  sentence = split_text_into_sentences(text)
  document_chunked = parts_to_dictionary(parts = sentence) # или же (parts = lines)
  
  # работа с ИИ (читать документацию к cohere)
  response = co.chat(
    model = "command-r-plus-08-2024",
    documents = document_chunked,
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": instructions.question_1 + instructions.question_2}
    ], 
  )
  print(f"Анализ файла {file}:")
  print(response.message.content[0].text, "\n")