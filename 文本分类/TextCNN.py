'''
https://www.kaggle.com/mlwhiz/learning-text-classification-textcnn
'''
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import *
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.callbacks import *
os.environ["CUDA_VISIBLE_DEVICES"]="0"
df = pd.read_csv('train.csv')

TRAIN_SIZE = 0.8
train_data, test_data = train_test_split(df, test_size=1-TRAIN_SIZE, random_state=7) 

MAX_NB_WORDS = 100000
MAX_SEQUENCE_LENGTH = 72

# 分词和填充操作
tokenizer = Tokenizer(MAX_NB_WORDS)
tokenizer.fit_on_texts(train_data.question_text)

word_index = tokenizer.word_index
vocab_size = len(tokenizer.word_index) + 1

x_train = pad_sequences(tokenizer.texts_to_sequences(train_data.question_text),
                        maxlen = MAX_SEQUENCE_LENGTH)
x_test = pad_sequences(tokenizer.texts_to_sequences(test_data.question_text),
                       maxlen = MAX_SEQUENCE_LENGTH)

y_train = np.array(train_data.target).reshape(-1,1)
y_test = np.array(test_data.target).reshape(-1,1)

#得到训练集和测试集
np.random.seed(2)
train_idx = np.random.permutation(len(x_train))
test_idx = np.random.permutation(len(x_test))

x_train = x_train[train_idx]
x_test = x_test[test_idx]
y_train = y_train[train_idx]
y_test = y_test[test_idx]

#方法一
filter_sizes = [1,2,3,5]
num_filters = 36

EMBEDDING_DIM = 300
BATCH_SIZE = 512
EPOCHS = 10

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedding_sequences = Embedding(MAX_NB_WORDS, EMBEDDING_DIM)(sequence_input)
x = SpatialDropout1D(0.4)(embedding_sequences)
x = Reshape((MAX_SEQUENCE_LENGTH, EMBEDDING_DIM, 1))(x)
    
conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], EMBEDDING_DIM), kernel_initializer='normal',
                                                                                activation='elu')(x)
conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], EMBEDDING_DIM), kernel_initializer='normal',
                                                                                activation='elu')(x)
conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], EMBEDDING_DIM), kernel_initializer='normal',
                                                                                activation='elu')(x)
conv_3 = Conv2D(num_filters, kernel_size=(filter_sizes[3], EMBEDDING_DIM), kernel_initializer='normal',
                                                                                activation='elu')(x)
maxpool_0 = MaxPool2D(pool_size=(MAX_SEQUENCE_LENGTH - filter_sizes[0] + 1, 1))(conv_0)
maxpool_1 = MaxPool2D(pool_size=(MAX_SEQUENCE_LENGTH - filter_sizes[1] + 1, 1))(conv_1)
maxpool_2 = MaxPool2D(pool_size=(MAX_SEQUENCE_LENGTH - filter_sizes[2] + 1, 1))(conv_2)
maxpool_3 = MaxPool2D(pool_size=(MAX_SEQUENCE_LENGTH - filter_sizes[3] + 1, 1))(conv_3)
        
z = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2, maxpool_3])   
z = Flatten()(z)
z = Dropout(0.1)(z)    
outputs = Dense(1, activation="sigmoid")(z)
    
model = Model(inputs=sequence_input, outputs=outputs)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
              
history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, 
                    validation_data=(x_test, y_test))

#方法二
GLOVE_EMB = 'glove.6B.300d.txt'
EMBEDDING_DIM = 300
embeddings_index = {}

f = open(GLOVE_EMB, encoding='UTF-8')
for line in f:
    values = line.split()
    word = value = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
     
filter_sizes = [1,2,3,5]
num_filters = 36
EMBEDDING_DIM = 300
BATCH_SIZE = 512
EPOCHS = 10

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedding_sequences = Embedding(vocab_size,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)(sequence_input)
x = SpatialDropout1D(0.4)(embedding_sequences)
x = Reshape((MAX_SEQUENCE_LENGTH, EMBEDDING_DIM, 1))(x)
    
conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], EMBEDDING_DIM), kernel_initializer='normal',
                                                                                activation='elu')(x)
conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], EMBEDDING_DIM), kernel_initializer='normal',
                                                                                activation='elu')(x)
conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], EMBEDDING_DIM), kernel_initializer='normal',
                                                                                activation='elu')(x)
conv_3 = Conv2D(num_filters, kernel_size=(filter_sizes[3], EMBEDDING_DIM), kernel_initializer='normal',
                                                                                activation='elu')(x)
maxpool_0 = MaxPool2D(pool_size=(MAX_SEQUENCE_LENGTH - filter_sizes[0] + 1, 1))(conv_0)
maxpool_1 = MaxPool2D(pool_size=(MAX_SEQUENCE_LENGTH - filter_sizes[1] + 1, 1))(conv_1)
maxpool_2 = MaxPool2D(pool_size=(MAX_SEQUENCE_LENGTH - filter_sizes[2] + 1, 1))(conv_2)
maxpool_3 = MaxPool2D(pool_size=(MAX_SEQUENCE_LENGTH - filter_sizes[3] + 1, 1))(conv_3)
        
z = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2, maxpool_3])   
z = Flatten()(z)
z = Dropout(0.1)(z)    
outputs = Dense(1, activation="sigmoid")(z)
    
model = Model(inputs=sequence_input, outputs=outputs)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, 
                    validation_data=(x_test, y_test))
