import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import DistilBertModel

class DistilBERT_clf(nn.Module):
    def __init__(self, num_classes_a, num_classes_b):
        super(DistilBERT_clf, self).__init__()
        self.num_classes_a = num_classes_a
        self.num_classes_b = num_classes_b
        
        #PreTrained
        self.distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        
        #LSTM
        self.lstm = nn.LSTM(768, 128, 1, batch_first=True)
        
        #Attention
        self.W_s1 = nn.Linear(128, 350)
        self.W_s2 = nn.Linear(350, 30)
        self.fc_layer = nn.Linear(30*128, 2000) 
        
        #Task A layers
        self.taska_layer = nn.Linear(2000, self.num_classes_a)
        
        #Task B layers
        self.taskb_layer = nn.Linear(2000, self.num_classes_b)
        
    def attention_net(self, lstm_output):
        attn_weight_matrix = self.W_s2(F.tanh(self.W_s1(lstm_output)))
        attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)
        attn_weight_matrix = F.softmax(attn_weight_matrix, dim=2)
        return attn_weight_matrix
    
    def forward(self, input_id, attention_mask, labels_a, labels_b):
        outputs = self.distilbert(input_ids=input_id, attention_mask=attention_mask)
        
        last_hidden_state = outputs.last_hidden_state
        
        lstm_input = last_hidden_state
        lstm_out, (ht, ct) = self.lstm(torch.tensor(lstm_input))
        
        attn_weight_matrix = self.attention_net(lstm_out)
        hidden_matrix = torch.bmm(attn_weight_matrix, lstm_out)
        attention_output = hidden_matrix.view(-1, hidden_matrix.size()[1]*hidden_matrix.size()[2])
        
        logits = self.fc_layer(attention_output)
        
        out_a = self.taska_layer(logits)
        outputs_a = (out_a,) + outputs[2:]
        
        out_b = self.taskb_layer(logits)
        outputs_b = (out_b,) + outputs[2:]
        
        loss_fct = CrossEntropyLoss() #loss function
        
        if labels_a is not None:
            loss_a = loss_fct(out_a.view(-1, self.num_classes_a), labels_a.view(-1))
            outputs_a = (loss_a,) + outputs_a
            
        if labels_b is not None:
            loss_b = loss_fct(out_b.view(-1, self.num_classes_b), labels_b.view(-1))
            outputs_b = (loss_b,) + outputs_b
        
        return outputs_a, outputs_b