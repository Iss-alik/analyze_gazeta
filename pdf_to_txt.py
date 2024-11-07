from pypdf import PdfReader
import os 

# Открываем папку с исходником
for file in os.listdir("pdf_files"):
    # Достаем имя файла 
    file_name = file[0:-4] 

    # Создаем элемент класса PdfReader
    reader = PdfReader(f'pdf_files/{file}')
    for page in reader.pages: # Перебираем страницы файлы
        # достаем текст пдф файла
        text = str(page.extract_text())
        file = open(f'text_files/new_{file_name}.txt', 'a', encoding='utf8') # Открываем файл
        file.write(text) # Записываем в конец файла
        file.close() 