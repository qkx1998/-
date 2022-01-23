'''
基于bert_sklearn
'''
from bert_sklearn import BertClassifier

model = BertClassifier(max_seq_length=100, train_batch_size=16, bert_model='bert-base-uncased', validation_fraction=0)
model.epochs = 1
model.learning_rate = 1e-3

model.fit(train['text'], train['label'])
pred = model.predict(test['text'])

'''
基于simpletransformers
'''
from simpletransformers.classification import ClassificationModel

#模型配置文件位于bert_model文件夹下
model = ClassificationModel('bert', 'bert_model', 
                            num_labels=25, 
                            use_cuda=True,
                            args={'sliding_window': True,
                                  'logging_steps': 5,
                                  'train_batch_size': 64,
                                  'gradient_accumulation_steps': 16,
                                  'stride': 0.6,
                                  'max_seq_length': 100,
                                  'num_train_epochs': 1,
                                  })
model.train_model(train)

predictions, _ = model.predict([row['text'] for _, row in test.iterrows()])

'''
基于fast-bert版本
参考https://github.com/utterworks/fast-bert
'''
from fast_bert.data_cls import BertDataBunch

databunch = BertDataBunch(DATA_PATH, LABEL_PATH,
                          tokenizer='bert-base-uncased',
                          train_file='train.csv',
                          val_file='val.csv',
                          label_file='labels.csv',
                          text_col='text',
                          label_col='label',
                          batch_size_per_gpu=16,
                          max_seq_length=512,
                          multi_gpu=True,
                          multi_label=False,
                          model_type='bert')
from fast_bert.learner_cls import BertLearner
from fast_bert.metrics import accuracy
import logging

logger = logging.getLogger()
device_cuda = torch.device("cuda")
metrics = [{'name': 'accuracy', 'function': accuracy}]

learner = BertLearner.from_pretrained_model(
						databunch,
						pretrained_path='bert-base-uncased',
						metrics=metrics,
						device=device_cuda,
						logger=logger,
						output_dir=OUTPUT_DIR,
						finetuned_wgts_path=None,
						warmup_steps=500,
						multi_gpu=True,
						is_fp16=True,
						multi_label=False,
						logging_steps=50)

learner.lr_find(start_lr=1e-5,optimizer_type='lamb')
learner.fit(epochs=6,
			      lr=6e-5,
			      validate=True, 	# Evaluate the model after each epoch
			      schedule_type="warmup_cosine",
			      optimizer_type="lamb")

texts = ['I really love the Netflix original movies',
		     'this movie is not worth watching']
predictions = learner.predict_batch(texts)

'''
基于keras-bert
'''
import json
import numpy as np
import pandas as pdfrom keras_bert import load_trained_model_from_checkpoint, Tokenizer# 超参数
maxlen = 100
batch_size = 16
droup_out_rate = 0.5
learning_rate = 1e-5
epochs = 15

path_prefix = "./test"
# 预训练模型目录
config_path = path_prefix + "/chinese_L-12_H-768_A-12/bert_config.json"
checkpoint_path = path_prefix + "/chinese_L-12_H-768_A-12/bert_model.ckpt"
dict_path = path_prefix + "/chinese_L-12_H-768_A-12/vocab.txt"

# 读取数据
neg = pd.read_excel(path_prefix + "/data/neg.xls", header=None)
pos = pd.read_excel(path_prefix + "/data/pos.xls", header=None)

# 构建训练数据
data = []

for d in neg[0]:
    data.append((d, 0))

for d in pos[0]:
    data.append((d, 1))

# 读取字典
token_dict = load_vocabulary(dict_path)
# 建立分词器
tokenizer = Tokenizer(token_dict)

# 按照9:1的比例划分训练集和验证集
random_order = list(range(len(data)))
np.random.shuffle(random_order)
train_data = [data[j] for i, j in enumerate(random_order) if i % 10 != 0]
valid_data = [data[j] for i, j in enumerate(random_order) if i % 10 == 0]

def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


class data_generator:
    def __init__(self, data, batch_size=batch_size):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1
    def __len__(self):
        return self.steps
    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            np.random.shuffle(idxs)
            X1, X2, Y = [], [], []
            for i in idxs:
                d = self.data[i]
                text = d[0][:maxlen]
                x1, x2 = tokenizer.encode(first=text)
                y = d[1]
                X1.append(x1)
                X2.append(x2)
                Y.append([y])
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y = seq_padding(Y)
                    yield [X1, X2], Y
                    [X1, X2, Y] = [], [], []
                   
from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.optimizers import Adam

# trainable设置True对Bert进行微调
# 默认不对Bert模型进行调参
bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, , trainable=True)

x1_in = Input(shape=(None,))
x2_in = Input(shape=(None,))

x = bert_model([x1_in, x2_in])
x = Lambda(lambda x: x[:, 0])(x)
x = Dropout(droup_out_rate)(x)
p = Dense(1, activation='sigmoid')(x)

model = Model([x1_in, x2_in], p)
model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate),
    metrics=['accuracy']
)
model.summary()

train_D = data_generator(train_data)
valid_D = data_generator(valid_data)

model.fit_generator(
    train_D.__iter__(),
    steps_per_epoch=len(train_D),
    epochs=epochs,
    validation_data=valid_D.__iter__(),
    validation_steps=len(valid_D)
)


'''
基于transformers pt版本
'''
import torch
from transformers import *
from transformers import BertTokenizer, BertModel,BertForSequenceClassification,AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tqdm import tqdm, trange
import tensorflow as tf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0)

MAX_LEN = 128

#creates list of sentences and labels
sentences = train.text.values
labels = train.target.values

#initialize BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case=True)

#user tokenizer to convert sentences into tokenizer
input_ids  = [tokenizer.encode(sent,add_special_tokens=True,max_length=MAX_LEN) for sent in sentences]

# Pad our input tokens
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

# Create attention masks
attention_masks = []

# Create a mask of 1s for each token followed by 0s for padding
for seq in input_ids:
  seq_mask = [float(i>0) for i in seq]
  attention_masks.append(seq_mask)
  
# Use train_test_split to split our data into train and validation sets for training
train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, random_state=2018, test_size=0.1)
train_masks, validation_masks, _, _ = train_test_split(attention_masks, input_ids, random_state=2018, test_size=0.1)

# Convert all of our data into torch tensors, the required datatype for our model
train_inputs = torch.tensor(train_inputs)
validation_inputs = torch.tensor(validation_inputs)
train_labels = torch.tensor(train_labels)
validation_labels = torch.tensor(validation_labels)
train_masks = torch.tensor(train_masks)
validation_masks = torch.tensor(validation_masks)

# Select a batch size for training. For fine-tuning BERT on a specific task, the authors recommend a batch size of 16 or 32
batch_size = 16

# Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for loop, 
# with an iterator the entire dataset does not need to be loaded into memory
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

# Load BertForSequenceClassification, the pretrained BERT model with a single linear classification layer on top. 
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.cuda()

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5, eps=10e-8)

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)
  
# # Store our loss and accuracy for plotting
train_loss_set = []

# # Number of training epochs (authors recommend between 2 and 4)
epochs = 5

# trange is a tqdm wrapper around the normal python range
for _ in trange(epochs, desc="Epoch"):
  # Training
  # Set our model to training mode (as opposed to evaluation mode)
  model.train()
  
  # Tracking variables
  tr_loss = 0
  nb_tr_examples, nb_tr_steps = 0, 0
  
  # Train the data for one epoch
  for step, batch in enumerate(train_dataloader):
    # Add batch to GPU
    batch = tuple(t.to(device) for t in batch)
    # Unpack the inputs from our dataloader
    b_input_ids, b_input_mask, b_labels = batch
    # Clear out the gradients (by default they accumulate)
    optimizer.zero_grad()
    # Forward pass
    # loss = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
    outputs = model(b_input_ids,token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
    loss = outputs[0]
    train_loss_set.append(loss.item())    
    # Backward pass
    loss.backward()
    # Update parameters and take a step using the computed gradient
    optimizer.step()
    
    # Update tracking variables
    tr_loss += loss.item()
    nb_tr_examples += b_input_ids.size(0)
    nb_tr_steps += 1

  print("Train loss: {}".format(tr_loss/nb_tr_steps))
    
    
  # Validation
  # Put model in evaluation mode to evaluate loss on the validation set
  model.eval()

  # Tracking variables 
  eval_loss, eval_accuracy = 0, 0
  nb_eval_steps, nb_eval_examples = 0, 0

  # Evaluate data for one epoch
  for batch in validation_dataloader:
    # Add batch to GPU
    batch = tuple(t.to(device) for t in batch)
    # Unpack the inputs from our dataloader
    b_input_ids, b_input_mask, b_labels = batch
    # Telling the model not to compute or store gradients, saving memory and speeding up validation
    with torch.no_grad():
      # Forward pass, calculate logit predictions
      outputs =  model(b_input_ids,token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
      loss, logits = outputs[:2]
    
    # Move logits and labels to CPU
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()
    tmp_eval_accuracy = flat_accuracy(logits, label_ids)
    eval_accuracy += tmp_eval_accuracy
    nb_eval_steps += 1
  print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
  
#creates list of sentences and labels
sentences = test.text.values
qids = all_test.qid.values

#user tokenizer to convert sentences into tokenizer
input_ids  = [tokenizer.encode(sent,add_special_tokens=True,max_length=MAX_LEN) for sent in sentences]

# Pad our input tokens
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

# Create attention masks
attention_masks = []

# Create a mask of 1s for each token followed by 0s for padding
for seq in input_ids:
  seq_mask = [float(i>0) for i in seq]
  attention_masks.append(seq_mask)

prediction_inputs = torch.tensor(input_ids)
prediction_masks = torch.tensor(attention_masks)

batch_size = 16 

prediction_data = TensorDataset(prediction_inputs, prediction_masks,)
prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

# Prediction on testing set
model.eval()
flat_pred = []
for batch in prediction_dataloader:
  # Add batch to GPU
  batch = tuple(t.to(device) for t in batch)
  # Unpack the inputs from our dataloader
  b_input_ids, b_input_mask = batch
  # Telling the model not to compute or store gradients, saving memory and speeding up prediction
  with torch.no_grad():
    # Forward pass, calculate logit predictions
    outputs =  model(b_input_ids,token_type_ids=None, attention_mask=b_input_mask)
    logits = outputs[0]
    logits = logits.detach().cpu().numpy() 
    flat_pred.extend(np.argmax(logits, axis=1).flatten())
    

'''
基于transformers tf版本
'''
from transformers import RobertaTokenizer, TFRobertaModel
from transformers import BertTokenizer, TFBertModel
from transformers import XLNetTokenizer, TFXLNetModel
from tensorflow.keras.optimizers import Adam


models = {'roberta-large':(RobertaTokenizer,'roberta-large',TFRobertaModel),
          #'roberta-base':(RobertaTokenizer,'roberta-base',TFRobertaModel),
          #'bert-large':(BertTokenizer, 'bert-large-uncased', TFBertModel),
          #'bert-base':(BertTokenizer, 'bert-base-uncased', TFBertModel),
          #'xlnet':(XLNetTokenizer, 'xlnet-large-cased', TFXLNetModel)
         }

tokenizer, model_type, model_name = models['roberta-large']

def make_inputs(tokenizer, model_type, serie, max_len= 70):

    tokenizer = tokenizer.from_pretrained(model_type, lowercase=True )
    tokenized_data = [tokenizer.encode_plus(text, max_length=max_len, 
                                            padding='max_length', 
                                            add_special_tokens=True,
                                            truncation = True) for text in serie]

    
    input_ids = np.array([text['input_ids'] for text in tokenized_data])
    attention_mask = np.array([text['attention_mask'] for text in tokenized_data])
    
    return input_ids, attention_mask

input_ids_train, attention_mask_train = make_inputs(tokenizer, model_type, train.text)

##### TPU or no TPU
def init_model(model_name, model_type, num_labels, Tpu = 'on', max_len = 70):
# ------------------------------------------------ with TPU --------------------------------------------------------------#
    if Tpu == 'on':
        # a few lines of code to get our tpu started and our data distributed on it
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        # print("All devices: ", tf.config.list_logical_devices('TPU'))

        strategy = tf.distribute.experimental.TPUStrategy(resolver)
        with strategy.scope():

            model_ = model_name.from_pretrained(model_type)
            # inputs
            input_ids = tf.keras.Input(shape = (max_len, ), dtype = 'int32')
            attention_masks = tf.keras.Input(shape = (max_len,), dtype = 'int32')
            
            outputs = model_([input_ids, attention_masks])

            if 'xlnet' in model_type:
                # cls is the last token in xlnet tokenization
                outputs = outputs[0]
                cls_output = tf.squeeze(outputs[:, -1:, :], axis=1)
            else:
                cls_output = outputs[1]

            final_output = tf.keras.layers.Dense(num_labels, activation = 'softmax')(cls_output)
            model = tf.keras.Model(inputs = [input_ids, attention_masks], outputs = final_output)
            model.compile(optimizer = Adam(lr = 1e-5), loss = 'categorical_crossentropy',
                        metrics = ['accuracy'], )
# ------------------------------------------------ without TPU --------------------------------------------------------------#
    else:
        model_ = model_name.from_pretrained(model_type)
        # inputs
        input_ids = tf.keras.Input(shape = (max_len, ), dtype = 'int32')
        attention_masks = tf.keras.Input(shape = (max_len,), dtype = 'int32')
        
        outputs = model_([input_ids, attention_masks])

        if 'xlnet' in model_type:
            # cls is the last token in xlnet tokenization
            outputs = outputs[0]
            cls_output = tf.squeeze(outputs[:, -1:, :], axis=1)
        else:
            cls_output = outputs[1]

        
        final_output = tf.keras.layers.Dense(num_labels, activation = 'softmax')(cls_output)

        model = tf.keras.Model(inputs = [input_ids, attention_masks], outputs = final_output)

        model.compile(optimizer = Adam(lr = 1e-5), loss = 'categorical_crossentropy',
                    metrics = ['accuracy'])
    return model
model = init_model(model_name, model_type, num_labels = 5, Tpu = 'on', max_len = 70)

train_y = tf.keras.utils.to_categorical(train.Sentiment, num_classes=5)
del train

model.fit([input_ids_train, attention_mask_train], train_y,
          validation_split=0.2, epochs = 3, batch_size = 16,
          shuffle = True, verbose=2)

input_ids_test, attention_mask_test = make_inputs(tokenizer, model_type, test.text, max_len= 70)

y_pred = model.predict([input_ids_test, attention_mask_test])
pred = np.argmax(y_pred, axis = 1)

'''
TF版本： https://www.kaggle.com/cdeotte/tensorflow-roberta-0-705
'''

