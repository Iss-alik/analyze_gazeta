import re
import os



folder_path = "./sample"


# Функция для чтения всех текстовых файлов из папки
def count_number_of_articles(folder_path):
    number_of_articles = 0
    counter = 0
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith(".txt"):
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    text = file.read()
                    atricles = re.split(r'\n{3}', text)
                    number_of_articles += len(atricles)
            except Exception as e:
                print(f"Error reading {file_name}: {e}")
                counter += 1

    print("Number of unreaded documents: ", counter)
    return number_of_articles


print(count_number_of_articles(folder_path))

