from gensim.models.wrappers import LdaMallet
from gensim.models import LdaModel
import logic # кастомный файл 
import os
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt

import pickle



os.environ['MALLET_TEMP'] = 'C:/Users/nyni1/Desktop/Temp'
os.environ['MALLET_HOME'] = "C:/Users/nyni1/Desktop/Projects/Image_reader/mallet-2.0.8"
mallet_path = "C:/Users/nyni1/Desktop/Projects/Image_reader/mallet-2.0.8/bin/mallet.bat"

#analyze_gazeta/
folder_path = "./sample"

num_topics = 13

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
    return documents # Тут адаптиреутся исходный текст для анализа. Текст из файлов делиться на массив, где один документ одил элемент


# Тут тоже не помню, надо у гпт спросить 
def convertldaGenToldaMallet(mallet_model):
    model_gensim = LdaModel(
        id2word=mallet_model.id2word, num_topics=mallet_model.num_topics,
        alpha=mallet_model.alpha, eta=0,
    )
    model_gensim.state.sstats[...] = mallet_model.wordtopics
    model_gensim.sync_state()
    return model_gensim



if __name__ == '__main__':
    # Чтение и очистка документов
    
    docs = read_documents_from_folder(folder_path)

    tokenized_docs = [doc.split() for doc in docs] # В документах делаем токены

    id2word = Dictionary(tokenized_docs)
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in tokenized_docs]
   
    Mallet_model = LdaMallet(mallet_path, corpus = corpus, num_topics = num_topics, id2word = id2word)
    #Магия


    Lda_model = convertldaGenToldaMallet(Mallet_model)
    
    # Объекты для сохранения
    data_to_save = {
        "model": Lda_model,
        "corpus": corpus,
        "id2word": id2word
    }

    # Сохранение всех объектов в один файл что бы потом отдельным кодом все визуализировать, потому что зависимости паломаны 
    with open('lda_data.pkl', 'wb') as f:
        pickle.dump(data_to_save, f)

    # Number of topics to display
    num_words = 15  # Number of words per topic to display

    print("Topics discovered by the LDA model:\n")
    for topic_id in range(num_topics):
        # Get the topic as a list of tuples (word, probability)
        topic = Lda_model.show_topic(topic_id, num_words)
        topic_words = ", ".join([f"{word} ({prob:.2f})" for word, prob in topic])
        print(f"Topic {topic_id + 1}: {topic_words}\n")

