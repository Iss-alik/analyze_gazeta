import logic
import os
from collections import Counter

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
                
    return [doc.split() for doc in documents] 

tokenized_docs = read_documents_from_folder("./sample")

freq = Counter()
for doc in tokenized_docs:
    freq.update(doc)

# Выведи топ 50 слов
for word, count in freq.most_common(50):
    print(word, count)

