import re
import os
import numpy as np
import pandas as pd
import nltk 
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer # 语言转换
nltk.download('stopwords')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
os.environ["CUDA_VISIBLE_DEVICES"]="0"

df = pd.read_csv('train.csv', encoding = 'latin', header=None)
df.columns = ['sentiment', 'id', 'date', 'query', 'user_id', 'text']
df = df[['text','sentiment']]
df['sentiment'] = df['sentiment'].map(lambda x: 'negative' if x == 0 else 'positive')

#对文本进行预处理
stop_words = stopwords.words('english')
stemmer = SnowballStemmer('english')
text_cleaning_re = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

def preprocess(text, stem=False):
    text = re.sub(text_cleaning_re, ' ', str(text).lower()).strip()
    tokens = []
    for token in text.split():
        if token not in stop_words:
            if stem:
                #表示对每个词提取主干。比如 boys -> boy
                tokens.append(stemmer.stem(token))
            else:
                tokens.append(token)
    return " ".join(tokens)

df.text = df.text.apply(preprocess)

TRAIN_SIZE = 0.8
# 用在tokenizer步骤的参数，表示处理出现频率在前100000的单词
MAX_NB_WORDS = 100000
# 用在pad_sequences步骤的参数，表示将语句扩充或截断为30的长度
MAX_SEQUENCE_LENGTH = 30
#切分训练集和测试集
train_data, test_data = train_test_split(df, test_size=1-TRAIN_SIZE, random_state=7) 

# 分词器
tokenizer = Tokenizer(MAX_NB_WORDS)
# fit_on_texts:后面接训练的文本列表
tokenizer.fit_on_texts(train_data.text)

# 字典：返回单词对应的索引
word_index = tokenizer.word_index
# 记录字典大小，创建 emb 矩阵的时候要用到.
# 有 vocab_size 个词，我们就要建立 vocab_size * emb_size 大小的矩阵。
vocab_size = len(tokenizer.word_index) + 1

# 序列填充：大于此长度的序列将被截短，小于此长度的序列将在后部填0.
# 填充有不同的模式可调，填充的长度也可调。
# 经过这一步得到完整的训练数据和测试数据
x_train = pad_sequences(tokenizer.texts_to_sequences(train_data.text),
                        maxlen = MAX_SEQUENCE_LENGTH)
x_test = pad_sequences(tokenizer.texts_to_sequences(test_data.text),
                       maxlen = MAX_SEQUENCE_LENGTH)
                                     
# 得到训练用的标签
le = LabelEncoder()
le.fit(train_data.sentiment.to_list())
y_train = le.transform(train_data.sentiment.to_list())
y_test = le.transform(test_data.sentiment.to_list())
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

'''
模型训练方式有两种：
方法一：端到端的训练，在训练的过程中生成embedding。
方法二：加载网上预训练好的语料向量，本文使用的为glove.6B.300d.txt。
'''
# 方法一
EMBEDDING_DIM = 300
LR = 1e-3
BATCH_SIZE = 1024
EPOCHS = 10

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedding_sequences = Embedding(MAX_NB_WORDS, EMBEDDING_DIM)(sequence_input)
x = SpatialDropout1D(0.2)(embedding_sequences)
x = Conv1D(64, 5, activation='relu')(x)
x = Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2))(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
outputs = Dense(1, activation='sigmoid')(x)
model = Model(sequence_input, outputs)

model.compile(optimizer=Adam(learning_rate=LR), 
              loss='binary_crossentropy',
              metrics=['accuracy'])

ReduceLROnPlateau = ReduceLROnPlateau(factor=0.1,
                                      min_lr=0.01,
                                      monitor='val_loss',
                                      verbose=1)

history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
                    validation_data=(x_test, y_test), callbacks=[ReduceLROnPlateau])


#方法二
GLOVE_EMB = 'glove.6B.300d.txt'
EMBEDDING_DIM = 300
embeddings_index = {}

#加载预训练的向量
f = open(GLOVE_EMB, encoding='UTF-8')
for line in f:
    values = line.split()
    word = value = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

# 创建一个emb矩阵
embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
# 将文本单词对应的向量从预训练向量中取出来
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
        
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedding_sequences = Embedding(vocab_size,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)(sequence_input)
x = SpatialDropout1D(0.2)(embedding_sequences)
x = Conv1D(64, 5, activation='relu')(x)
x = Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2))(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
outputs = Dense(1, activation='sigmoid')(x)
model = Model(sequence_input, outputs)

model.compile(optimizer=Adam(learning_rate=LR), 
              loss='binary_crossentropy',
              metrics=['accuracy'])

ReduceLROnPlateau = ReduceLROnPlateau(factor=0.1,
                                      min_lr=0.01,
                                      monitor='val_loss',
                                      verbose=1)

history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
                    validation_data=(x_test, y_test), callbacks=[ReduceLROnPlateau])


