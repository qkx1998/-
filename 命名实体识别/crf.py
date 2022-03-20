import pandas as pd
import os
import sklearn_crfsuite
import re
import sklearn_crfsuite
from sklearn_crfsuite import metrics
import joblib
import yaml
import warnings
warnings.filterwarnings('ignore')

data_path = os.getcwd().replace('code', 'data\\')

data = list()
data_sent_with_label = list()

# 读取数据
with open(data_path+'train_500.txt', mode='r', encoding='utf-8') as f:
    for line in f:
        if line.strip() == '':
            data.append(data_sent_with_label.copy())
            data_sent_with_label.clear()
        else:
            if len(line.strip().split(' ')) == 2:
                data_sent_with_label.append(tuple(line.strip().split(' ')))
            else:
                data_sent_with_label.append(('NAN', 'O'))
                
train = data[:400]
valid = data[400:]


def word2features(sent, i):
    word = sent[i][0]

    features = {
        'bias': 1.0,
        'word': word,
        'word.isdigit()': word.isdigit(),
    }
    if i > 0:
        word1 = sent[i-1][0]
        words = word1 + word
        features.update({
            '-1:word': word1,
            '-1:words': words,
            '-1:word.isdigit()': word1.isdigit(),
        })
    else:
        features['BOS'] = True

    if i > 1:
        word2 = sent[i-2][0]
        word1 = sent[i-1][0]
        words = word1 + word2 + word
        features.update({
            '-2:word': word2,
            '-2:words': words,
            '-3:word.isdigit()': word1.isdigit(),
        })

    if i > 2:
        word3 = sent[i - 3][0]
        word2 = sent[i - 2][0]
        word1 = sent[i - 1][0]
        words = word1 + word2 + word3 + word
        features.update({
            '-3:word': word3,
            '-3:words': words,
            '-3:word.isdigit()': word1.isdigit(),
        })

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        words = word1 + word
        features.update({
            '+1:word': word1,
            '+1:words': words,
            '+1:word.isdigit()': word1.isdigit(),
        })
    else:
        features['EOS'] = True

    if i < len(sent)-2:
        word2 = sent[i + 2][0]
        word1 = sent[i + 1][0]
        words = word + word1 + word2
        features.update({
            '+2:word': word2,
            '+2:words': words,
            '+2:word.isdigit()': word2.isdigit(),
        })

    if i < len(sent)-3:
        word3 = sent[i + 3][0]
        word2 = sent[i + 2][0]
        word1 = sent[i + 1][0]
        words = word + word1 + word2 + word3
        features.update({
            '+3:word': word3,
            '+3:words': words,
            '+3:word.isdigit()': word3.isdigit(),
        })

    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [ele[-1] for ele in sent]
  
  
X_train = [sent2features(s) for s in train]
y_train = [sent2labels(s) for s in train]

X_dev = [sent2features(s) for s in valid]
y_dev = [sent2labels(s) for s in valid]

crf_model = sklearn_crfsuite.CRF(algorithm='lbfgs',c1=0.25,c2=0.018,max_iterations=100,
                                 all_possible_transitions=True,verbose=True)
crf_model.fit(X_train, y_train)

labels=list(crf_model.classes_)
labels.remove("O")

y_pred = crf_model.predict(X_dev)

metrics.flat_f1_score(y_dev, y_pred, average='weighted', labels=labels)

sorted_labels = sorted(labels,key=lambda name: (name[1:], name[0]))
print(metrics.flat_classification_report( y_dev, y_pred, labels=sorted_labels, digits=3 ))

