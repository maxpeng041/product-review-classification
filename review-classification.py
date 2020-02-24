import nltk
import random
import numpy as np
from nltk.corpus import PlaintextCorpusReader
from sklearn.model_selection import KFold

neg_root = './DC1/neg'
pos_root = './DC1/pos'
neg_lists = PlaintextCorpusReader(neg_root, '.*.txt')
pos_lists = PlaintextCorpusReader(pos_root, '.*.txt')

corpus = []
neg_corpus = [(list(w.lower() for w in neg_lists.words(fileid)), 0)
              for fileid in neg_lists.fileids()]
pos_corpus = [(list(w.lower() for w in pos_lists.words(fileid)), 1)
              for fileid in pos_lists.fileids()]
for i in neg_corpus:
    corpus.append(i)
for i in pos_corpus:
    corpus.append(i)

random.shuffle(corpus)

print(list(word for review in corpus for word in review[0])[:2000])

all_words = nltk.FreqDist(word.lower() for review in corpus for word in review[0])
word_features = list(all_words)#[:2000]
print(list(all_words))

def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features

print(document_features(corpus[0][0]))

featuresets = [(document_features(d), l) for (d, l) in corpus]
train_set, test_set = featuresets[int(len(featuresets)*.2):], featuresets[:int(len(featuresets)*.2)]
classifier = nltk.NaiveBayesClassifier.train(train_set)

print(nltk.classify.accuracy(classifier, test_set))
classifier.show_most_informative_features(10)

# cross validation
n_splits = 10
kf = KFold(n_splits=n_splits)
cv_accuracies = []
for train, test in kf.split(featuresets):
    train_data = np.array(featuresets)[train]
    test_data = np.array(featuresets)[test]
    classifier = nltk.NaiveBayesClassifier.train(train_data)
    cv_accuracies.append(nltk.classify.accuracy(classifier, test_data))
average = sum(cv_accuracies)/n_splits

print('Accuracies with ' + str(n_splits) + '-fold cross validation: ')
for cv_accuracy in cv_accuracies:
    print(cv_accuracy)

print('Average: ' + str(average))