# $pip install nltk

import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
print(len(list(wn.all_synsets(pos='n'))))
print(len(list(wn.all_synsets(pos='v'))))
print(len(list(wn.all_synsets(pos='a'))))
print(len(list(wn.all_synsets(pos='r'))))
