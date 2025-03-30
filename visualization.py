import pickle
import pyLDAvis
import pyLDAvis.gensim
import pyLDAvis.gensim_models as gensimvis

with open('lda_data.pkl', 'rb') as f:
    loaded_data = pickle.load(f)

# Извлечение данных
Lda_model_loaded = loaded_data["model"]
corpus_loaded = loaded_data["corpus"]
id2word_loaded = loaded_data["id2word"]

prepared_data = gensimvis.prepare(Lda_model_loaded, corpus_loaded, id2word_loaded, mds='pcoa')  # Используем gensim_models
pyLDAvis.save_html(prepared_data, 'lda_visualization.html')