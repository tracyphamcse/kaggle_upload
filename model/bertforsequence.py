from transformers import BertPreTrainedModel
from transformers import BertModel
from torch.autograd import Variable

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from config.config import *

class BertBase(BertPreTrainedModel):
    
    def __init__(self, config):
        super(BertBase, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)

        # Classifier
        self.batch_size = BATCH_SIZE
        self.dropout = DROPOUT
        self.hidden_dim = HIDDEN_DIM
        self.relu_dim_1 = RELU_DIM_1
        self.relu_dim_2 = RELU_DIM_2
        
        self.hidden2dense = nn.Linear(config.hidden_size, self.relu_dim_1)
        self.dense2dense = nn.Linear(self.relu_dim_1, self.relu_dim_2)
        self.dense2label = nn.Linear(self.relu_dim_2, self.config.num_labels)
        self.dropout = nn.Dropout(0.2)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        x = outputs[1]
        x = self.dropout(x)
        y = self.hidden2dense(x)
        y = self.dropout(y)
        y = self.dense2dense(y)
        y = self.dropout(y)
        logits = self.dense2label(y)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
    
class BertBiLSTM(BertPreTrainedModel):
    
    def __init__(self, config):
        super(BertBiLSTM, self).__init__(config)
        
        # BERT
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        
        # Classifier
        self.batch_size = BATCH_SIZE
        self.dropout = DROPOUT
        self.hidden_dim = HIDDEN_DIM
        self.relu_dim_1 = RELU_DIM_1
        self.relu_dim_2 = RELU_DIM_2
        
        self.lstm = nn.LSTM(input_size=config.hidden_size, hidden_size=self.hidden_dim, num_layers=2)

        self.hidden2dense = nn.Linear(config.hidden_size + self.hidden_dim, self.relu_dim_1)
        self.dense2dense = nn.Linear(self.relu_dim_1, self.relu_dim_2)
        self.dense2label = nn.Linear(self.relu_dim_2, self.config.num_labels)
        self.dropout = nn.Dropout(0.2)
        
        self.init_weights()
        self.lstm_hidden = self.init_hidden()
        
    def init_hidden(self):
        # first is the hidden h
        # second is the cell c
        return (Variable(torch.zeros(2, self.batch_size, self.hidden_dim)).to(DEVICE),
                Variable(torch.zeros(2, self.batch_size, self.hidden_dim)).to(DEVICE))
    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        
        # Value of [CLS]
        x_pool = outputs[1]          
        
        # The last hidden-state is the first element of the output tuple
        # Go through LSTM model
        # (remove first token [CLS]
        x_sequence = outputs[0]  
        x_sequence = x_sequence.view(MAX_SEQ_LENGTH, self.batch_size, -1)
        lstm_out, self.lstm_hidden = self.lstm(x_sequence, self.lstm_hidden)
       
        # Concat CLS with LSTM before pass through FC
        x = torch.cat((x_pool, lstm_out[-1]), dim=1)
        
        y = self.hidden2dense(x)
        y = self.dropout(y)
        y = self.dense2dense(y)
        y = self.dropout(y)
        logits = self.dense2label(y)
        
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)