!pip install nltk
import nltk
from nltk.corpus import movie_reviews
nltk.download('movie_reviews')
from nltk import FreqDist, classify, ConfusionMatrix

from nltk.corpus import movie_reviews
nltk.download('movie_reviews')
# create datasets
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
word_features = list(all_words)[:2000] # [_document-classify-all-words]

def document_features(document): # [_document-classify-extractor]
    document_words = set(document) # [_document-classify-set]
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features

featuresets = [(document_features(d), c) for (d,c) in documents]
train_set, test_set = featuresets[100:], featuresets[:100]


# model 2: Decision Tree
classifier2 = nltk.DecisionTreeClassifier.train(train_set)
nltk.classify.accuracy(classifier2, test_set)