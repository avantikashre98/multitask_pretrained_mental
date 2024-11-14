import argparse
import random
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score
import torch
from torch import nn
import torch.nn.functional as F
from helpers.AutomaticWeightedLoss import AutomaticWeightedLoss
from helpers.data import EDAIC

import warnings
warnings.filterwarnings('ignore')

#configuration
num_classes_a = 2
num_classes_b = 2
max_seq_len = 128
model_name = 'BERT'
lr = 1e-3
epochs = 5
batch_size = 32
data = 'EDAIC'
question = 'doing_today'
random_seed = 42
run = 0

#argparse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int)
parser.add_argument("--data")
parser.add_argument("--question")
parser.add_argument("--model_name")
parser.add_argument("--lr", type=float)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--run", type=int)

args = parser.parse_args()

if args.__dict__["epochs"]  is not None:
    epochs = args.__dict__["epochs"]
if args.__dict__["data"]  is not None:
    data = args.__dict__["data"]
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
    
def flat_accuracy(preds, labels):
    '''
    Function to calculate the accuracy of our predictions vs labels
    '''
    
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

#loading model
if model_name == 'BERT':
    from models.bert_mtl import BERT_clf
    model = BERT_clf(num_classes_a, num_classes_b)
elif model_name == 'MentalBERT':
    from models.mentalbert_mtl import MentalBERT_clf
    model = MentalBERT_clf(num_classes_a, num_classes_b)
elif model_name == 'RoBERTa':
    from models.roberta_mtl import RoBERTa_clf
    model = RoBERTa_clf(num_classes_a, num_classes_b)
elif model_name == 'DistilBERT':
    from models.distilbert_mtl import DistilBERT_clf
    model = DistilBERT_clf(num_classes_a, num_classes_b)
elif model_name == 'ALBERT':
    from models.albert_mtl import ALBERT_clf
    model = ALBERT_clf(num_classes_a, num_classes_b)

model.to(device)

#setting seed value
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
    
#preparing train & test data
if data == 'EDAIC':
    train_dataloader, test_dataloader = EDAIC(model_name, question, batch_size, max_seq_len, random_seed)
    
#loading optimizer
from transformers import AdamW 
awl = AutomaticWeightedLoss(2)
optimizer = AdamW([{'params': model.parameters()}, {'params': awl.parameters(), 'weight_decay': 0}])

#creating the learning rate scheduler
from transformers import get_linear_schedule_with_warmup
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)

#storing training loss, total loss (train and test), f1, and balanced accuracy
tr_loss_a = []
tr_loss_b = []

loss_tr_values = []
loss_te_values = []

f1_phq8_values = []
ba_phq8_values = []
auc_phq8_values = []

f1_ptsd_values = []
ba_ptsd_values = []
auc_ptsd_values = []

TP_phq8 = []
TN_phq8 = []
FP_phq8 = []
FN_phq8 = []

TP_ptsd = []
TN_ptsd = []
FP_ptsd = []
FN_ptsd = []

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
        b_ptsd = batch[3].to(device)
    
        model.zero_grad()

        output_a, output_b = model(b_input_ids, attention_mask=b_input_mask, labels_a=b_phq8, labels_b=b_ptsd)
        
        loss_a = output_a[0]
        tr_loss_a.append(loss_a)
        
        loss_b = output_b[0]
        tr_loss_b.append(loss_b)
        
        tr_loss = awl(loss_a, loss_b)

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
    
    predictions_a, ground_truth_a = [], []
    predictions_b, ground_truth_b = [], []
    
    for batch in test_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_phq8 = batch[2].to(device)
        b_ptsd = batch[3].to(device)
        
        output_a, output_b = model(b_input_ids, attention_mask=b_input_mask, labels_a=b_phq8, labels_b=b_ptsd)

        loss_a = output_a[0]
        logits_a = output_a[1]
        
        loss_b = output_b[0]
        logits_b = output_b[1]
        
        loss = awl(loss_a, loss_b)
        
        total_te_loss += loss.item()
        
        pred_a = torch.argmax(logits_a, dim=1)      
        predictions_a.append(pred_a.detach().cpu().numpy())
        ground_truth_a.append(b_phq8.detach().cpu().numpy())
        
        pred_b = torch.argmax(logits_b, dim=1)      
        predictions_b.append(pred_b.detach().cpu().numpy())
        ground_truth_b.append(b_ptsd.detach().cpu().numpy())
        
    loss_te = total_te_loss/len(test_dataloader)
    loss_te_values.append(loss_te)
    
    print('')
    print('Average testing loss: {0:.2f}'.format(total_te_loss/len(test_dataloader)))
    
    flat_predictions_a = [item for sublist in predictions_a for item in sublist]
    flat_ground_truth_a = [item for sublist in ground_truth_a for item in sublist]
    
    flat_predictions_b = [item for sublist in predictions_b for item in sublist]
    flat_ground_truth_b = [item for sublist in ground_truth_b for item in sublist]

    TP_a = 0
    FN_a = 0
    TN_a = 0
    FP_a = 0

    for i in range(len(flat_predictions_a)):
        if flat_ground_truth_a[i] == 1:
            if flat_predictions_a[i] == flat_ground_truth_a[i]:
                TP_a += 1
            else:
                FN_a += 1
        elif flat_ground_truth_a[i] == 0:
            if flat_predictions_a[i] == flat_ground_truth_a[i]:
                TN_a += 1
            else:
                FP_a += 1
    TP_phq8.append(TP_a)
    TN_phq8.append(TN_a)
    FP_phq8.append(FP_a)
    FN_phq8.append(FN_a)

    TP_b = 0
    FN_b = 0
    TN_b = 0
    FP_b = 0

    for i in range(len(flat_predictions_b)):
        if flat_ground_truth_b[i] == 1:
            if flat_predictions_b[i] == flat_ground_truth_b[i]:
                TP_b += 1
            else:
                FN_b += 1
        elif flat_ground_truth_b[i] == 0:
            if flat_predictions_b[i] == flat_ground_truth_b[i]:
                TN_b += 1
            else:
                FP_b += 1
    TP_ptsd.append(TP_b)
    TN_ptsd.append(TN_b)
    FP_ptsd.append(FP_b)
    FN_ptsd.append(FN_b)
    
    f1_phq8 = f1_score(flat_ground_truth_a, flat_predictions_a, average='binary')
    f1_phq8_values.append(f1_phq8)
    
    f1_ptsd = f1_score(flat_ground_truth_b, flat_predictions_b, average='binary')
    f1_ptsd_values.append(f1_ptsd)
    
    ba_phq8 = balanced_accuracy_score(flat_ground_truth_a, flat_predictions_a)
    ba_phq8_values.append(ba_phq8)
    
    ba_ptsd = balanced_accuracy_score(flat_ground_truth_b, flat_predictions_b)
    ba_ptsd_values.append(ba_ptsd)
    
    auc_phq8 = roc_auc_score(flat_ground_truth_a, flat_predictions_a)
    auc_phq8_values.append(auc_phq8)
    
    auc_ptsd = roc_auc_score(flat_ground_truth_b, flat_predictions_b)
    auc_ptsd_values.append(auc_ptsd)
    
#creating dataframe of values
value_dict = {'model': model_name, 'epoch': range(epochs),
              'run': run, 'lr': lr, 'batch_size': batch_size, 
              'data': data, 'question': question,
              'TP_phq8': TP_phq8, 'TN_phq8': TN_phq8,
              'FP_phq8': FP_phq8, 'FN_phq8':FN_phq8,
              'TP_ptsd': TP_ptsd, 'TN_ptsd': TN_ptsd,
              'FP_ptsd': FP_ptsd, 'FN_ptsd':FN_ptsd,
              'F1_phq8': f1_phq8_values, 'F1_ptsd': f1_ptsd_values,
              'BA_phq8': ba_phq8_values, 'BA_ptsd': ba_ptsd_values,
              'AUC_phq8': auc_phq8_values, 'AUC_ptsd': auc_ptsd_values,
              'training_loss': loss_tr_values, 'testing_loss': loss_te_values,
              'training': 'MTL_awl'}
values = pd.DataFrame(data=value_dict)

#saving as .csv file
values.to_csv('results/'+model_name+'_'+data+'_'+question+'_awl.csv', mode='a', index=False)