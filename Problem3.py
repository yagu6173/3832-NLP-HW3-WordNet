# $!pip install nltk
import nltk
from nltk.corpus import brown
nltk.download('tagsets_json')
nltk.download('tagsets')

taggedWords = brown.tagged_words()
singularNouns = [w for (w, p) in taggedWords if p == 'NN']
pluralNouns = [w for (w, p) in taggedWords if p == 'NNS']

singular_freq = nltk.FreqDist(singularNouns)
plural_freq = nltk.FreqDist(pluralNouns)

more = []
for plural in plural_freq:
    if plural[-1] == 's': 
        sing = plural[:-1]
        sing_freq = singular_freq[sing]
        if sing_freq > 0 and sing_freq < plural_freq[plural]:
            more.append(plural)
        
print(len(more))


tag_set = set(brown.tagged_words())
tag_dict = {}
for word, tag in tag_set:
    if word not in tag_dict:
        tag_dict[word] = set(tag)
    else:
        tag_dict[word].add(tag)

most_tag = max(tag_dict, key=lambda key: len(tag_dict[key]))
most_tag,tag_dict[most_tag],len(tag_dict[most_tag])

for target_tag in tag_dict[most_tag]:
    print(nltk.help.brown_tagset(target_tag))

length = len(taggedWords)
previous = []
for i in range(1,length):
    if taggedWords[i][1] == "NN":
        previous.append(taggedWords[i-1])
print(len(previous))

prev_tag_fd = nltk.FreqDist(tag for (word,tag) in previous)
print(prev_tag_fd.most_common(10))

#nltk.help.brown_tagset()

