# $pip install nltk

import nltk
from nltk.corpus import wordnet as wn

word_pairs = [
    ("car", "automobile"),
    ("gem", "jewel"),
    ("journey", "voyage"),
    ("boy", "lad"),
    ("coast", "shore"),
    ("asylum", "madhouse"),
    ("magician", "wizard"),
    ("midday", "noon"),
    ("furnace", "stove"),
    ("food", "fruit"),
    ("bird", "cock"),
    ("bird", "crane"),
    ("tool", "implement"),
    ("brother", "monk"),
    ("lad", "brother"),
    ("crane", "implement"),
    ("journey", "car"),
    ("monk", "oracle"),
    ("cemetery", "woodland"),
    ("food", "rooster"),
    ("coast", "hill"),
    ("forest", "graveyard"),
    ("shore", "woodland"),
    ("monk", "slave"),
    ("coast", "forest"),
    ("lad", "wizard"),
    ("chord", "smile"),
    ("glass", "magician"),
    ("rooster", "voyage"),
    ("noon", "string"),
]

sim_score = []
rank = []
for (w1, w2) in word_pairs:
    s1 = wn.synsets(w1)[0]
    s2 = wn.synsets(w2)[0]
    sim_score.append(s1.path_similarity(s2))

for sim in sim_score:
    rank.append(sim_score.index(sim) + 1)    
print(rank)

scores = [87, 75, 75, 50, 32, 32]
ranks = []
for score in scores:
    #print(scores.index)
    ranks.append(scores.index(score) + 1)
print(ranks)