import os
import re 
from recover import *
import logic
import time

# /analyze_gazeta  март 45/
path_to_dir = "./analyze_gazeta/text_files/original/февраль 45/"

pattern = re.compile(r'[А-Яа-я0-9.,!? \n-]+')

def text_to_chunks(path):
    chunks = []

    with open(path, "r", encoding="utf-8") as file:
        all_text = "".join( pattern.findall(file.read()) )

    raw_chunks = re.split(r'\n{2,}', all_text)

    for chunk in raw_chunks:
        number_of_symbols = len(chunk)
        standart_size = 5000

        if number_of_symbols > standart_size:
            tokens_of_chunk = logic.split_text_into_tokens(chunk)
            chunks += logic.parts_to_chunks(parts = tokens_of_chunk, number_of_chunks = int(number_of_symbols/standart_size) + 1)

        else:
            chunks.append(chunk) 

    return chunks


def text_to_chunks2(path):
    chunks = []

    with open(path, "r", encoding="utf-8") as file:
        all_text = "".join( pattern.findall(file.read()))
        tokens = logic.split_text_into_tokens(all_text)
        chunks += logic.parts_to_chunks( parts = tokens, length_of_chunk = 400 )

    return chunks

# Открываем папку с исходником
for file in os.listdir(path_to_dir):
    # Достаем имя файла 
    file_name = file[0:-4] 

    chunked_raw_text = text_to_chunks2(path_to_dir + file)
    all_recovered_text= ""


    for chunk in chunked_raw_text:
        recovered_chunk = recover2(chunk)
        all_recovered_text += recovered_chunk + "\n"
        time.sleep(2)  # пауза между запросами

    try:
        recovered = open(f'./analyze_gazeta/text_files/recovered/{file_name}.txt', 'w', encoding = 'utf8')
        recovered.write(all_recovered_text)
        recovered.close() 

    except Exception as error:
        print(f"Error with file {file_name} \n\
              Error is {type(error)}: {error}")

