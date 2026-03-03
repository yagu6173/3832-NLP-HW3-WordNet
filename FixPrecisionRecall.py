import nltk
from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy
from nltk import FreqDist, classify, ConfusionMatrix
import timeit


# Helper function to extract features
def extract_features(words):
    return {word: True for word in words}
# Load movie reviews dataset
documents = [(list(movie_reviews.words(fileid)), category)
    for category in movie_reviews.categories()
    for fileid in movie_reviews.fileids(category)]
# Shuffle the documents ... why do we shuffle the documents?
import random
random.seed(42)
random.shuffle(documents)


# Split dataset into training and test sets (80% training, 20% testing)
train_size = int(len(documents) * 0.8)
train_set = [(extract_features(doc), category) for (doc, category) in documents[:train_size]]
test_set = [(extract_features(doc), category) for (doc, category) in documents[train_size:]]
# Train the Naive Bayes classifier
# Train the Naive Bayes classifier
print("Starting training")
startTime =timeit.default_timer()
classifier = NaiveBayesClassifier.train(train_set)
endTime =timeit.default_timer()
print("Total time: ", endTime - startTime)
# Test the classifier and print accuracy
print(f"Accuracy: {accuracy(classifier, test_set):.4f}")
# Get the actual and predicted labels
test_actual = [label for (features, label) in test_set]
test_predicted = [classifier.classify(features) for (features, label) in test_set]

# Build the confusion matrix
cm = ConfusionMatrix(test_actual, test_predicted)
print("Confusion Matrix:")
print(cm)
# Calculate precision and recall for each label (category)
labels = set(test_actual)

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

accuracy = accuracy_score(test_actual, test_predicted)
print(f"Overall Accuracy:\t {accuracy:.4f}")

precision = precision_score(test_actual, test_predicted, average=None)
recall = recall_score(test_actual, test_predicted, average=None)
fscore = f1_score(test_actual, test_predicted, average=None)

labelNames = sorted(set(labels))
for index in range(len(labelNames)):
    label = labelNames[index]
    print(f"Precision for {label}:\t {precision[index]:.4f}")
    print(f"Recall for {label}:\t\t {recall[index]:.4f}")
    print(f"F1-score for {label}:\t {fscore[index]:.4f}")

print("\n\n")
print("Most important features: ")
# Print top 10 most informative features
classifier.show_most_informative_features(10) # can use for Naive Bayes or maxent

# if decision tree, you could print out a portion of the tree
# Print the tree structure
# print(classifier.pseudocode(depth=5))  # Limits depth to avoid huge output


