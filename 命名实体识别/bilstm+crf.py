import pandas as pd
import os
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
import keras as k
from keras_contrib.layers import CRF
from sklearn.metrics import classification_report
from keras.callbacks import ModelCheckpoint

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

data_collect = []
for i in range(len(data)):
    tmp = pd.DataFrame(data[i])
    tmp['sentence_idx'] = i+1
    data_collect.append(tmp)
    
data = pd.concat(data_collect)
data.columns = ['word', 'tag', 'sentence_idx']

class SentenceGetter(object):
    
    def __init__(self, dataset):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, t) for w,t in zip(s["word"].values.tolist(), s["tag"].values.tolist())]
        self.grouped = self.data.groupby("sentence_idx").apply(agg_func)
        self.sentences = [s for s in self.grouped]
    
    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None
        
getter = SentenceGetter(data)      
sentences = getter.sentences

words = list(set(data["word"].values))
words.append("ENDPAD")

n_words = len(words); n_words

tags = []
for tag in set(data["tag"].values):
    if tag is np.nan or isinstance(tag, float):
        tags.append('unk')
    else:
        tags.append(tag)
print(tags)

n_tags = len(tags); n_tags

word2idx = {w: i for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(tags)}
idx2tag = {v: k for k, v in tag2idx.items()}

X = [[word2idx[w[0]] for w in s] for s in sentences]
X = pad_sequences(maxlen=70, sequences=X, padding="post",value=n_words - 1)

y_idx = [[tag2idx[w[1]] for w in s] for s in sentences]
y = pad_sequences(maxlen=70, sequences=y_idx, padding="post", value=tag2idx["O"])
y = [to_categorical(i, num_classes=n_tags) for i in y]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

word_embedding_size = 32

input = Input(shape=(70,))
model = Embedding(input_dim=n_words, output_dim=word_embedding_size, input_length=70)(input)
model = Bidirectional(LSTM(units=word_embedding_size, 
                           return_sequences=True, 
                           dropout=0.5, 
                           recurrent_dropout=0.5, 
                           kernel_initializer=k.initializers.he_normal()))(model)
model = LSTM(units=word_embedding_size * 2, 
             return_sequences=True, 
             dropout=0.5, 
             recurrent_dropout=0.5, 
             kernel_initializer=k.initializers.he_normal())(model)
model = TimeDistributed(Dense(n_tags, activation="relu"))(model)  # previously softmax output layer

crf = CRF(n_tags)  # CRF layer
out = crf(model)  # outpu
model = Model(input, out)
adam = k.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999)
model.compile(optimizer=adam, loss=crf.loss_function, metrics=[crf.accuracy, 'accuracy'])

filepath="ner-bi-lstm-td-model-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
history = model.fit(X_train, np.array(y_train), batch_size=256, epochs=20, validation_split=0.2, verbose=1, callbacks=callbacks_list)

i = 1
p = model.predict(np.array([X_test[i]]))
p = np.argmax(p, axis=-1)
gt = np.argmax(y_test[i], axis=-1)
print(gt)
print("{:14}: ({:5}): {}".format("Word", "True", "Pred"))
for idx, (w,pred) in enumerate(zip(X_test[i],p[0])):
    print("{:14}: ({:5}): {}".format(words[w],idx2tag[gt[idx]],tags[pred]))

p = model.predict(np.array(X_test))  

print(classification_report(np.argmax(y_test, 2).ravel(), np.argmax(p, axis=2).ravel(),labels=list(idx2tag.keys()), target_names=list(idx2tag.values())))

TP = {}
TN = {}
FP = {}
FN = {}
for tag in tag2idx.keys():
    TP[tag] = 0
    TN[tag] = 0    
    FP[tag] = 0    
    FN[tag] = 0    

def accumulate_score_by_tag(gt, pred):
    """
    For each tag keep stats
    """
    if gt == pred:
        TP[gt] += 1
    elif gt != 'O' and pred == 'O':
        FN[gt] +=1
    elif gt == 'O' and pred != 'O':
        FP[gt] += 1
    else:
        TN[gt] += 1

for i, sentence in enumerate(X_test):
    y_hat = np.argmax(p[i], axis=-1)
    gt = np.argmax(y_test[i], axis=-1)
    for idx, (w,pred) in enumerate(zip(sentence,y_hat)):
        accumulate_score_by_tag(idx2tag[gt[idx]],tags[pred])
        
for tag in tag2idx.keys():
    print(f'tag:{tag}')    
    print('\t TN:{:10}\tFP:{:10}'.format(TN[tag],FP[tag]))
    print('\t FN:{:10}\tTP:{:10}'.format(FN[tag],TP[tag]))    


