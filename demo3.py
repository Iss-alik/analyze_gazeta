import gensim
import gensim.corpora as corpora
from gensim.corpora import Dictionary

from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel

import matplotlib.pyplot as plt
import pandas as pd
import os, logic  


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
        model = LdaModel(corpus=corpus, num_topics=num_topics, id2word=id2word)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values

def preprocess(doc):
    return [word for word in doc.split() if word.isalpha()]

if __name__ == '__main__':
    # Чтение и очистка документов
    #analyze_gazeta/
    folder_path = "./text_files/recovered/"
    docs = read_documents_from_folder(folder_path)

    tokenized_docs = [doc.split() for doc in docs]

    id2word = Dictionary(tokenized_docs)
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in tokenized_docs]

    model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=tokenized_docs, start = 3, limit = 12, step = 2)
    
    best_result_index = coherence_values.index(max(coherence_values))
    optimal_model = model_list[best_result_index]

    # Show graph
    x = range(3, 12, 2)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()

    num_topics = optimal_model.num_topics
    num_words = 10
    print("Topics discovered by the LDA model:\n")
    for topic_id in range(num_topics):
        # Get the topic as a list of tuples (word, probability)
        topic = optimal_model.show_topic(topic_id, num_words)
        topic_words = ", ".join([f"{word} ({prob:.2f})" for word, prob in topic])
        print(f"Topic {topic_id + 1}: {topic_words}\n")