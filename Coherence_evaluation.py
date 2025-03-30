from gensim.models.wrappers import LdaMallet
from gensim.models import LdaModel
import logic
import os
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt

import pickle


os.environ['MALLET_TEMP'] = 'C:/Users/nyni1/Desktop/Temp'
os.environ['MALLET_HOME'] = "C:/Users/nyni1/Desktop/Projects/Image_reader/mallet-2.0.8"
mallet_path = "C:/Users/nyni1/Desktop/Projects/Image_reader/mallet-2.0.8/bin/mallet.bat"

#analyze_gazeta/
folder_path = "./text_files/recovered/"

limit=15; start=8; step=1

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
    return documents

def preprocess(doc):
    return [word for word in doc.split() if word.isalpha()]

def convertldaGenToldaMallet(mallet_model):
    model_gensim = LdaModel(
        id2word=mallet_model.id2word, num_topics=mallet_model.num_topics,
        alpha=mallet_model.alpha, eta=0,
    )
    model_gensim.state.sstats[...] = mallet_model.wordtopics
    model_gensim.sync_state()
    return model_gensim

def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


if __name__ == '__main__':
    # Чтение и очистка документов
    
    docs = read_documents_from_folder(folder_path)

    tokenized_docs = [doc.split() for doc in docs] # В документах делаем токены

    id2word = Dictionary(tokenized_docs)
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in tokenized_docs]
    model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=tokenized_docs, start = start, limit = limit, step = step)

    # Show graph
    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()

    best_result_index = coherence_values.index(max(coherence_values))
    optimal_model = model_list[best_result_index]
    # Select the model and print the topics
    model_topics = optimal_model.show_topics(formatted=False)
    print(f'''The {x[best_result_index]} topics gives the highest coherence score 
    of {coherence_values[best_result_index]}''')


    optimal_model = convertldaGenToldaMallet(optimal_model)

    # Объекты для сохранения
    data_to_save = {
        "model": optimal_model,
        "corpus": corpus,
        "id2word": id2word
    }

    # Сохранение всех объектов в один файл
    with open('lda_data.pkl', 'wb') as f:
        pickle.dump(data_to_save, f)

    num_topics = optimal_model.num_topics
    num_words = 10
    print("Topics discovered by the LDA model:\n")
    for topic_id in range(num_topics):
        # Get the topic as a list of tuples (word, probability)
        topic = optimal_model.show_topic(topic_id, num_words)
        topic_words = ", ".join([f"{word} ({prob:.2f})" for word, prob in topic])
        print(f"Topic {topic_id + 1}: {topic_words}\n")