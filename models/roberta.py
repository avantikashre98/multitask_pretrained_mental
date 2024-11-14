import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import RobertaModel

class RoBERTa_clf(nn.Module):
    def __init__(self, num_classes):
        super(RoBERTa_clf, self).__init__()
        self.num_classes = num_classes
        
        #PreTrained
        self.roberta = RobertaModel.from_pretrained("FacebookAI/roberta-base")
        
        #LSTM
        self.lstm = nn.LSTM(768, 128, 1, batch_first=True)
        
        #Attention
        self.W_s1 = nn.Linear(128, 350)
        self.W_s2 = nn.Linear(350, 30)
        self.fc_layer = nn.Linear(30*128, 2000) 
        self.classifier = nn.Linear(2000, self.num_classes)
        
    def attention_net(self, lstm_output):
        attn_weight_matrix = self.W_s2(F.tanh(self.W_s1(lstm_output)))
        attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)
        attn_weight_matrix = F.softmax(attn_weight_matrix, dim=2)
        return attn_weight_matrix
    
    def forward(self, input_id, attention_mask, labels=None):
        outputs = self.roberta(input_ids=input_id, attention_mask=attention_mask)
        
        last_hidden_state = outputs.last_hidden_state
        
        lstm_input = last_hidden_state
        lstm_out, (ht, ct) = self.lstm(torch.tensor(lstm_input))
        
        attn_weight_matrix = self.attention_net(lstm_out)
        hidden_matrix = torch.bmm(attn_weight_matrix, lstm_out)
        attention_output = hidden_matrix.view(-1, hidden_matrix.size()[1]*hidden_matrix.size()[2])
        
        logits = self.classifier(self.fc_layer(attention_output))
        outputs = (logits,) + outputs[2:]
        
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
            outputs = (loss,) + outputs
        
        return outputs