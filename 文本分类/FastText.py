import os
import fasttext
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('train.csv')

# 划分训练集测试集
train, test = train_test_split(df, test_size=0.2, random_state=2)
train.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)

# 准备fasttext的输入数据，有特定的格式
train['label_format'] = 0
for i in range(train.shape[0]):
    train['label_format'][i] = '__label__' + str(train['category'][i]) + ' ' + str(train['text'][i])
   
test['label_format'] = 0
for i in range(test.shape[0]):
    test['label_format'][i] = '__label__' + str(test['category'][i]) + ' ' + str(test['text'][i])
    
train.label_format.to_csv('fasttext_train.txt',index=None,header=None)
test.label_format.to_csv('fasttext_test.txt',index=None,header=None)

# 训练模型（有监督的任务）
# 这里还有很多的参数可以调节：
# minCount：最小词频数
# wordgram：词ngram的最大长度
# dim: 词向量维度
# ws: 窗口大小
# 在train_unsupervised中还可以设置model参数，包括cbow和skip gram.
model = fasttext.train_supervised('fasttext_train.txt', 
                                  epoch=50,
                                  lr=0.05, 
                                  label_prefix='__label__',
                                  dim=300)
#测试模型
validation = model.test('fasttext_test.txt')

#得到字典词表
model.get_words()

# 获取对应单词的向量
model.get_word_vector('ve')

# 模型对单条样本做出预测
model.predict('mersin çıkışa geçmek istiyor hentbol erkekler süper_ligi nde mücadele')


## keras版本
VOCAB_SIZE = 100000
EMBEDDING_DIM = 300
MAX_SEQUENCE_LENGTH = 64

def FastText():
    model = Sequential()
    model.add(Embedding(input_dim=VOCAB_SIZE, 
                        output_dim=EMBEDDING_DIM,
                        input_length=MAX_SEQUENCE_LENGTH))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(7, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model
