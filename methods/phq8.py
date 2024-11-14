import argparse
import random
import re
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, BertTokenizer, RobertaTokenizer, DistilBertTokenizer, AlbertTokenizer
from helpers.NLP_utils import pad_sequences

import warnings
warnings.filterwarnings('ignore')

#configuration
num_classes = 2
max_seq_len = 128
model_name = 'BERT'
lr = 1e-3
epochs = 5
batch_size = 32
question = 'doing_today'
random_seed = 42
run = 0

#argparse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int)
parser.add_argument("--question")
parser.add_argument("--model_name")
parser.add_argument("--lr", type=float)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--run", type=int)

args = parser.parse_args()

if args.__dict__["epochs"]  is not None:
    epochs = args.__dict__["epochs"]
if args.__dict__["question"]  is not None:
    question = args.__dict__["question"]
if args.__dict__["model_name"]  is not None:
    model_name = args.__dict__["model_name"]           
if args.__dict__["lr"]  is not None:
    lr = args.__dict__["lr"]  
if args.__dict__["batch_size"]  is not None:
    batch_size = args.__dict__["batch_size"]
if args.__dict__["run"]  is not None:
    run = args.__dict__["run"]

print('Model:', model_name, 'LR:', lr, 'BS:', batch_size, 'Seed:', random_seed, 'Run:', run)
    
#cuda
if torch.cuda.is_available():     
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
    
#tokenizer function
def tokenizer_func(model_name, X):
    if model_name == 'BERT':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    elif model_name == 'MentalBERT':
        tokenizer = AutoTokenizer.from_pretrained('mental/mental-bert-base-uncased', do_lower_case=True)
    elif model_name == 'RoBERTa':
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True)
    elif model_name == 'DistilBERT':
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)
    elif model_name == 'ALBERT':
        tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2', do_lower_case=True)
    else:
        print('Model not available!')

    input_ids = []
    for x in X:
        encoded_x = tokenizer.encode(x)
        input_ids.append(encoded_x)

    input_ids = pad_sequences(input_ids, maxlen=max_seq_len, dtype="long", value=0, truncating="post", padding="post")

    attention_masks = []
    for i in input_ids:
        att_mask = [int(token_id > 0) for token_id in i]
        attention_masks.append(att_mask)
        
    return input_ids, attention_masks

#loading model
if model_name == 'BERT':
    from models.bert import BERT_clf
    model = BERT_clf(num_classes)
elif model_name == 'MentalBERT':
    from models.mentalbert import MentalBERT_clf
    model = MentalBERT_clf(num_classes)
elif model_name == 'RoBERTa':
    from models.roberta import RoBERTa_clf
    model = RoBERTa_clf(num_classes)
elif model_name == 'DistilBERT':
    from models.distilbert import DistilBERT_clf
    model = DistilBERT_clf(num_classes)
elif model_name == 'ALBERT':
    from models.albert import ALBERT_clf
    model = ALBERT_clf(num_classes)

model.to('cuda')

#setting seed value
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)

#setting up train and test
train = pd.read_csv('data/'+question+'/'+'train.tsv', delimiter='\t', 
                    header=None, names=['index', 'label', 'label_notes', 'sentence', 'audio_features'])

test = pd.read_csv('data/'+question+'/'+'dev.tsv', delimiter='\t', 
                 header=None, names=['index', 'label', 'label_notes', 'sentence', 'audio_features'])

print('train, test:', len(train), len(test))
    
#preparing train data
train_sessions = []
for txt in train.audio_features:
    t = re.search("[0-9]{3}", txt).group()
    train_sessions.append(int(t))
    
train['Participant_ID'] = train_sessions

train_y = pd.read_csv('data/labels/'+question+'/train3.csv').drop(['PHQ_Score', 'PTSD Severity'], axis=1)

train = train_y.join(train.set_index('Participant_ID'), on='Participant_ID').drop(['index', 'label', 'label_notes', 'audio_features'], axis=1)
train = train.drop_duplicates()

train_X = train.sentence.values

train_phq8 = train.label_phq8_10.values

input_ids, attention_masks = tokenizer_func(model_name, train_X)
    
train_inputs = torch.tensor(input_ids)
train_masks = torch.tensor(attention_masks)
train_phq8_labels = torch.tensor(train_phq8)

train_data = TensorDataset(train_inputs, train_masks, train_phq8_labels)

train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

#preparing test data
test_sessions = []
for txt in test.audio_features:
    t = re.search("[0-9]{3}", txt).group()
    test_sessions.append(int(t))
    
test['Participant_ID'] = test_sessions

test_y = pd.read_csv('data/labels/'+question+'/test3.csv').drop(['PHQ_Score', 'PTSD Severity'], axis=1)

test = test_y.join(test.set_index('Participant_ID'), on='Participant_ID').drop(['index', 'label', 'label_notes', 'audio_features'], axis=1)
test = test.drop_duplicates()

test_X = test.sentence.values

test_phq8 = test.label_phq8_10.values

input_ids, attention_masks = tokenizer_func(model_name, test_X)

test_inputs = torch.tensor(input_ids)
test_masks = torch.tensor(attention_masks)
test_phq8_labels = torch.tensor(test_phq8)

test_data = TensorDataset(test_inputs, test_masks, test_phq8_labels)

test_sampler = RandomSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)
    
#loading optimizer
from transformers import AdamW 
optimizer = AdamW(model.parameters(), lr = lr, eps = 2e-8)

#creating the learning rate scheduler
from transformers import get_linear_schedule_with_warmup
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)

#storing training loss, total loss (train and test), f1, and balanced accuracy
tr_loss_phq8 = []

loss_tr_values = []
loss_te_values = []

f1_phq8_values = []
ba_phq8_values = []
auc_phq8_values = []

TP_phq8 = []
TN_phq8 = []
FP_phq8 = []
FN_phq8 = []

#training loop
for epoch in range(epochs):

    print('')
    print('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))
    print('Training...')
    
    total_tr_loss = 0
    
    model.train()
    
    for step, batch in enumerate(train_dataloader):
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_phq8 = batch[2].to(device)

    
        model.zero_grad()

        output = model(b_input_ids, attention_mask=b_input_mask, labels=b_phq8)
        
        tr_loss = output[0]
        tr_loss_phq8.append(tr_loss)

        total_tr_loss += tr_loss.item()

        tr_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        scheduler.step()
    
    loss_tr = total_tr_loss/len(train_dataloader)
    loss_tr_values.append(loss_tr)
    
    print('')
    print('Average training loss: {0:.2f}'.format(total_tr_loss/len(train_dataloader)))
    print('')
    print('Testing...')
    
    total_te_loss = 0
    
    #testing loop
    model.eval()
    
    predictions, ground_truth = [], []
    
    for batch in test_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_phq8 = batch[2].to(device)
        
        output = model(b_input_ids, attention_mask=b_input_mask, labels=b_phq8)

        loss = output[0]
        logits = output[1]
        
        total_te_loss += loss.item()
        
        pred = torch.argmax(logits, dim=1)      
        predictions.append(pred.detach().cpu().numpy())
        ground_truth.append(b_phq8.detach().cpu().numpy())
        
    loss_te = total_te_loss/len(test_dataloader)
    loss_te_values.append(loss_te)
    
    print('')
    print('Average testing loss: {0:.2f}'.format(total_te_loss/len(test_dataloader)))
    
    flat_predictions = [item for sublist in predictions for item in sublist]
    flat_ground_truth = [item for sublist in ground_truth for item in sublist]

    TP_a = 0
    FN_a = 0
    TN_a = 0
    FP_a = 0

    for i in range(len(flat_predictions)):
        if flat_ground_truth[i] == 1:
            if flat_predictions[i] == flat_ground_truth[i]:
                TP_a += 1
            else:
                FN_a += 1
        elif flat_ground_truth[i] == 0:
            if flat_predictions[i] == flat_ground_truth[i]:
                TN_a += 1
            else:
                FP_a += 1
    TP_phq8.append(TP_a)
    TN_phq8.append(TN_a)
    FP_phq8.append(FP_a)
    FN_phq8.append(FN_a)
    
    f1_phq8 = f1_score(flat_ground_truth, flat_predictions, average='binary')
    f1_phq8_values.append(f1_phq8)
    
    ba_phq8 = balanced_accuracy_score(flat_ground_truth, flat_predictions)
    ba_phq8_values.append(ba_phq8)
    
    auc_phq8 = roc_auc_score(flat_ground_truth, flat_predictions)
    auc_phq8_values.append(auc_phq8)
    
#creating dataframe of values
value_dict = {'model': model_name, 'epoch': range(epochs),
              'run': run, 'lr': lr,
              'batch_size': batch_size, 'question': question,
              'TP_phq8': TP_phq8, 'TN_phq8': TN_phq8,
              'FP_phq8': FP_phq8, 'FN_phq8':FN_phq8,
              'F1_phq8': f1_phq8_values,
              'BA_phq8': ba_phq8_values,
              'AUC_phq8': auc_phq8_values,
              'training_loss': loss_tr_values, 'testing_loss': loss_te_values,
              'training': 'STL'}
values = pd.DataFrame(data=value_dict)

#saving as .csv file
values.to_csv('results/'+model_name+'_EDAIC_'+question+'_phq8.csv', mode='a', index=False)