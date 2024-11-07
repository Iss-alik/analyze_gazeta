from nltk.tokenize import word_tokenize
import os
from nltk.text import Text
from instructions import key_words
from logic import evalute_complexity_of_text

import matplotlib.pyplot as plt
from nltk.draw.dispersion import dispersion_plot
from nltk.book import *


path_to_dir = "recovered_text_files/"
for file in os.listdir(path_to_dir):
    document = open(f"{path_to_dir}/{file}", 'r')

    text = document.read().lower()
    document.close()

    tokens = word_tokenize(text, language="russian")

    #print(evalute_complexity_of_text(tokens))

    tokens = [x for x in tokens if len(x) > 2]
    cla_text = Text(tokens)
    
    cla_text.collocations()

    dispersion_plot(cla_text, key_words, ignore_case=True, title=file)
    plt.show()

    #fd = FreqDist(cla_text)
    #print(len(set(cla_text)))
    #print(fd.most_common(20), fd[key_words[0]])


