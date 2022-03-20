import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertConfig, BertForTokenClassification
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
print(device)
import os

data_path = os.getcwd().replace('code', 'data\\')

with open(data_path + 'train_500.txt','r',encoding='utf-8') as f:
    tmp=[]
    cnt=1
    for line in tqdm(f.read().split('\n')):
        sentence_id=f'train_{cnt}'
        # print(line)
        if line!='\n' and len(line.strip())>0:
            word_tags=line.split(' ')
            if len(word_tags)==2:
                tmp.append([sentence_id]+word_tags)
            elif len(word_tags)==2:
                word=' '.join(word_tags[:-1])
                tag=word_tags[-1]
                tmp.append([sentence_id,word,tag])
        else:
            cnt+=1

data=pd.DataFrame(tmp, columns=['sentence_id','words','tags'])
data['sentence'] = data[['sentence_id','words','tags']].groupby(['sentence_id'])['words'].transform(lambda x: ' '.join(x))
data['word_labels'] = data[['sentence_id','words','tags']].groupby(['sentence_id'])['tags'].transform(lambda x: ','.join(x))

labels_to_ids = {k: v for v, k in enumerate(data.tags.unique())}
ids_to_labels = {v: k for v, k in enumerate(data.tags.unique())}

data = data[["sentence", "word_labels"]].drop_duplicates().reset_index(drop=True)

MAX_LEN = 91 # 120
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 2
EPOCHS = 5
LEARNING_RATE = 1e-05
MAX_GRAD_NORM = 5
MODEL_NAME='hfl/chinese-roberta-wwm-ext'
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME) # encode_plus()# 整体

def tokenize_and_preserve_labels(sentence, text_labels, tokenizer):
    """
  
    
    Word piece tokenization使得很难将词标签与单个subword进行匹配。
    这个函数每次次对每个单词进行一个分词，这样方便为每个subword保留正确的标签。 
    当然，它的处理时间有点慢，但它会帮助我们的模型达到更高的精度。
    """

    tokenized_sentence = []
    labels = []

    sentence = sentence.strip()

    for word, label in zip(sentence.split(), text_labels.split(",")):

        # 逐字分词
        tokenized_word = tokenizer.tokenize(word) # id
        n_subwords = len(tokenized_word) # 1

        # 将单个字分词结果追加到句子分词列表
        tokenized_sentence.extend(tokenized_word)

        # 标签同样添加n个subword，与原始word标签一致
        labels.extend([label] * n_subwords)

    return tokenized_sentence, labels
  

class dataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __getitem__(self, index):
        # 步骤 1: 对每个句子分词
        sentence = self.data.sentence[index]  
        word_labels = self.data.word_labels[index]  
        tokenized_sentence, labels = tokenize_and_preserve_labels(sentence, word_labels, self.tokenizer)
        
        # 步骤 2: 添加特殊token并添加对应的标签
        tokenized_sentence = ["[CLS]"] + tokenized_sentence + ["[SEP]"] # add special tokens
        labels.insert(0, "O") # 给[CLS] token添加O标签
        labels.insert(-1, "O") # 给[SEP] token添加O标签

        # 步骤 3: 截断/填充
        maxlen = self.max_len

        if (len(tokenized_sentence) > maxlen):
              # 截断
            tokenized_sentence = tokenized_sentence[:maxlen]
            labels = labels[:maxlen]
        else:
              # 填充
            tokenized_sentence = tokenized_sentence + ['[PAD]'for _ in range(maxlen - len(tokenized_sentence))]
            labels = labels + ["O" for _ in range(maxlen - len(labels))]

        # 步骤 4: 构建attention mask
        attn_mask = [1 if tok != '[PAD]' else 0 for tok in tokenized_sentence]
        
        # 步骤 5: 将分词结果转为词表的id表示
        ids = self.tokenizer.convert_tokens_to_ids(tokenized_sentence)

        label_ids = [labels_to_ids[label] for label in labels]
  
        
        return {
              'ids': torch.tensor(ids, dtype=torch.long),
              'mask': torch.tensor(attn_mask, dtype=torch.long),
              #'token_type_ids': torch.tensor(token_ids, dtype=torch.long),
              'targets': torch.tensor(label_ids, dtype=torch.long)
        } 
    
    def __len__(self):
        return self.len
      
      
from sklearn.model_selection import train_test_split
train_size = 0.8

train_dataset = data.sample(frac=train_size,random_state=200)
test_dataset = data.drop(train_dataset.index).reset_index(drop=True)
train_dataset = train_dataset.reset_index(drop=True)

print("FULL Dataset: {}".format(data.shape))
print("TRAIN Dataset: {}".format(train_dataset.shape))
print("TEST Dataset: {}".format(test_dataset.shape))

training_set = dataset(train_dataset, tokenizer, MAX_LEN)
testing_set = dataset(test_dataset, tokenizer, MAX_LEN)

# 创建torch的DataLoader
train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)

# 定义网络
model = BertForTokenClassification.from_pretrained(MODEL_NAME, num_labels=len(labels_to_ids))
model.to(device)

# 训练模型
ids = training_set[0]["ids"].unsqueeze(0)
mask = training_set[0]["mask"].unsqueeze(0)
targets = training_set[0]["targets"].unsqueeze(0) # 真实标签
ids = ids.to(device)
mask = mask.to(device)
targets = targets.to(device)
outputs = model(input_ids=ids, attention_mask=mask, labels=targets) # 输出有两个：一个为loss和一个为logits
initial_loss = outputs[0]
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

# 训练函数
def train(epoch):
    tr_loss, tr_accuracy = 0, 0
    nb_tr_examples, nb_tr_steps = 0, 0
    tr_preds, tr_labels = [], []
    # 将model设置为train模式
    model.train()
    
    for idx, batch in enumerate(training_loader):
        
        ids = batch['ids'].to(device, dtype = torch.long) #(4,91)
        mask = batch['mask'].to(device, dtype = torch.long) #(4,91)
        targets = batch['targets'].to(device, dtype = torch.long)#(4,91)
        
        
        outputs = model(input_ids=ids, attention_mask=mask, labels=targets)
        loss, tr_logits = outputs[0],outputs[1]
        # print(outputs.keys())
        # print(loss)
        tr_loss += loss.item()

        nb_tr_steps += 1
        nb_tr_examples += targets.size(0)
        
        if idx % 50==0:
            loss_step = tr_loss/nb_tr_steps
            print(f"Training loss per 50 training steps: {loss_step}")
           
        # 计算准确率
        flattened_targets = targets.view(-1) # 真实标签 大小 (batch_size * seq_len,)
        active_logits = tr_logits.view(-1, model.num_labels) # 模型输出shape (batch_size * seq_len, num_labels)
        flattened_predictions = torch.argmax(active_logits, axis=1) # 取出每个token对应概率最大的标签索引 shape (batch_size * seq_len,)
        # MASK：PAD
        active_accuracy = mask.view(-1) == 1 # shape (batch_size * seq_len,)
        targets = torch.masked_select(flattened_targets, active_accuracy)
        predictions = torch.masked_select(flattened_predictions, active_accuracy)
        
        tr_preds.extend(predictions)
        tr_labels.extend(targets)
        
        tmp_tr_accuracy = accuracy_score(targets.cpu().numpy(), predictions.cpu().numpy())
        tr_accuracy += tmp_tr_accuracy
    
        # 梯度剪切
        torch.nn.utils.clip_grad_norm_(
            parameters=model.parameters(), max_norm=MAX_GRAD_NORM
        )
        
        # loss反向求导
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_loss = tr_loss / nb_tr_steps
    tr_accuracy = tr_accuracy / nb_tr_steps
    print(f"Training loss epoch: {epoch_loss}")
    print(f"Training accuracy epoch: {tr_accuracy}")
    
# 开始训练
for epoch in range(EPOCHS):
    print(f"Training epoch: {epoch + 1}")
    train(epoch)
    
# 评估模型
def valid(model, testing_loader):
    #put model in evaluation mode
    model.eval()
    
    eval_loss, eval_accuracy = 0, 0
    nb_eval_examples, nb_eval_steps = 0, 0
    eval_preds, eval_labels = [], []
    
    with torch.no_grad():
        for idx, batch in enumerate(testing_loader):
            
            ids = batch['ids'].to(device, dtype = torch.long)
            mask = batch['mask'].to(device, dtype = torch.long)
            targets = batch['targets'].to(device, dtype = torch.long)
            
            # loss, eval_logits = model(input_ids=ids, attention_mask=mask, labels=targets)
            outputs = model(input_ids=ids, attention_mask=mask, labels=targets)
            loss, eval_logits = outputs[0],outputs[1]
            eval_loss += loss.item()

            nb_eval_steps += 1
            nb_eval_examples += targets.size(0)
        
            if idx % 100==0:
                loss_step = eval_loss/nb_eval_steps
                print(f"Validation loss per 100 evaluation steps: {loss_step}")
              
            # 计算准确率
            flattened_targets = targets.view(-1) # 大小 (batch_size * seq_len,)
            active_logits = eval_logits.view(-1, model.num_labels) # 大小 (batch_size * seq_len, num_labels)
            flattened_predictions = torch.argmax(active_logits, axis=1) # 大小 (batch_size * seq_len,)
            active_accuracy = mask.view(-1) == 1 # 大小 (batch_size * seq_len,)
            targets = torch.masked_select(flattened_targets, active_accuracy)
            predictions = torch.masked_select(flattened_predictions, active_accuracy)
            
            eval_labels.extend(targets)
            eval_preds.extend(predictions)
            
            tmp_eval_accuracy = accuracy_score(targets.cpu().numpy(), predictions.cpu().numpy())
            eval_accuracy += tmp_eval_accuracy
    
    #print(eval_labels)
    #print(eval_preds)

    labels = [ids_to_labels[id.item()] for id in eval_labels]
    predictions = [ids_to_labels[id.item()] for id in eval_preds]

    #print(labels)
    #print(predictions)
    
    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_steps
    print(f"Validation Loss: {eval_loss}")
    print(f"Validation Accuracy: {eval_accuracy}")

    return labels, predictions
  
  
labels, predictions = valid(model, testing_loader)

tmp=[]
for tags in data['word_labels']:
    tmp.extend(tags.split(','))
pd.Series(tmp).value_counts()

from sklearn.metrics import classification_report

print(classification_report([labels], [predictions])) 

# 预测
sentence = "手机三脚架网红直播支架桌面自拍杆蓝牙遥控三脚架摄影拍摄拍照抖音看电视神器三角架便携伸缩懒人户外支撑架【女神粉】自带三脚架+蓝牙遥控"

inputs = tokenizer(sentence, padding='max_length', truncation=True, max_length=MAX_LEN, return_tensors="pt")

# 加载到gpu
ids = inputs["input_ids"].to(device)
mask = inputs["attention_mask"].to(device)
# 输入到模型
outputs = model(ids, mask)
logits = outputs[0]

active_logits = logits.view(-1, model.num_labels) # 大小 (batch_size * seq_len, num_labels)
flattened_predictions = torch.argmax(active_logits, axis=1) # 大小 (batch_size*seq_len,) 

tokens = tokenizer.convert_ids_to_tokens(ids.squeeze().tolist())
token_predictions = [ids_to_labels[i] for i in flattened_predictions.cpu().numpy()]
wp_preds = list(zip(tokens, token_predictions)) # tuple = (wordpiece, prediction)

word_level_predictions = []
for pair in wp_preds:
  if (pair[0].startswith(" ##")) or (pair[0] in ['[CLS]', '[SEP]', '[PAD]']):
    # skip prediction
    continue
  else:
    word_level_predictions.append(pair[1])

# 拼接文本
str_rep = " ".join([t[0] for t in wp_preds if t[0] not in ['[CLS]', '[SEP]', '[PAD]']]).replace(" ##", "")
print(str_rep)
print(word_level_predictions)

# 保存模型
import os

directory = "./bert_ner_model"

if not os.path.exists(directory):
    os.makedirs(directory)

# 保存tokenizer
tokenizer.save_vocabulary(directory)
# 保存权重和配置文件
model.save_pretrained(directory)
print('All files saved')
print('This tutorial is completed')

