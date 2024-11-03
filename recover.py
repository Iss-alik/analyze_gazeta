from logic import *
import cohere
co = cohere.ClientV2('AH0agiBtbhKCBApQrt9OscLj22VjX6JV0O9KBwrN')
system_message = """## Задание и контекст
Вы получите ряд фрагментов из выпуска газеты, которые вместе состовляют один выпуск газеты.\
Как помощник, вы должны генерировать ответы на запросы пользователя на основе информации, предоставленной во фрагментах.\
Убедитесь, что ваши ответы точны и правдивы, и что вы ссылаетесь на свои источники, где это уместно, чтобы ответить на запросы,\
независимо от их сложности. Так-же отвечайте на русском языке"""

def recover(raw_text, file):
    sentence = split_text_into_sentences(raw_text)
    raw_text_chunked = parts_to_dictionary(parts = sentence)
    response = co.chat(
    model = "command-r-plus-08-2024",
    documents = raw_text_chunked,
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": "Тебе дана версия текста с потерей данных. Востанови данный текст"}
    ],)

    result = response.message.content[0].text
    file = open(f'text_files/recovered_{file}', 'a')
    file.write(result)#.encode('utf8'))
    file.close()
    return result