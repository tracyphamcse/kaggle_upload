import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss, MSELoss

from transformers import BertPreTrainedModel
from transformers import BertModel
from config.config import *

class BertLSTM(BertPreTrainedModel):
    
    def __init__(self, config):
        super(BertLSTM, self).__init__(config)
        
        # BERT
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        
        # Classifier
        self.batch_size = BATCH_SIZE
        self.dropout = DROPOUT
        self.hidden_dim = HIDDEN_DIM
        self.relu_dim_1 = RELU_DIM_1
        self.relu_dim_2 = RELU_DIM_2
        
        self.lstm_1 = nn.LSTM(input_size=config.hidden_size, hidden_size=self.hidden_dim, num_layers=2)
        self.lstm_2 = nn.LSTM(input_size=config.hidden_size, hidden_size=self.hidden_dim, num_layers=2)

        self.hidden2dense = nn.Linear(config.hidden_size + self.hidden_dim * 2, self.relu_dim_1)
        self.dense2label = nn.Linear(self.relu_dim_1, self.config.num_labels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
        self.init_weights()
        self.lstm_hidden_1 = self.init_hidden()
        self.lstm_hidden_2 = self.init_hidden()
        
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
        
        question = x_sequence[:, 1 : MAX_QUES_LENGTH + 1]
        question = question.reshape(MAX_QUES_LENGTH, self.batch_size, -1)
        lstm_out_1, self.lstm_hidden_1 = self.lstm_1(question, self.lstm_hidden_1)
        
        answer = x_sequence[:, MAX_QUES_LENGTH + 2 : MAX_QUES_LENGTH + MAX_ANSW_LENGTH + 2]
        answer = answer.reshape(MAX_ANSW_LENGTH, self.batch_size, -1)
        lstm_out_2, self.lstm_hidden_2 = self.lstm_2(answer, self.lstm_hidden_2)
        
       
        # Concat CLS with LSTM before pass through FC
        x = torch.cat((x_pool, lstm_out_1[-1], lstm_out_2[-1]), dim=1)
        
        y = self.dropout(x)
        y = self.hidden2dense(x)
        y = self.relu(y)
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
    
    
class BertBow(BertPreTrainedModel):
    
    def __init__(self, config):
        super(BertBow, self).__init__(config)
        
        # BERT
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        
        # Classifier
        self.batch_size = BATCH_SIZE
        self.dropout = DROPOUT
        self.hidden_dim = HIDDEN_DIM
        self.relu_dim_1 = RELU_DIM_1
        self.relu_dim_2 = RELU_DIM_2

        self.hidden2dense = nn.Linear(config.hidden_size + self.hidden_dim * 2, self.relu_dim_1)
        self.relu = nn.ReLU()
        self.dense2label = nn.Linear(self.relu_dim_1, self.config.num_labels)
        self.dropout = nn.Dropout(0.2)
        
        self.init_weights()
       
    
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
        
        question = x_sequence[:, 1 : MAX_QUES_LENGTH + 1]
        question = torch.sum(question, dim=1)
        
        answer = x_sequence[:, MAX_QUES_LENGTH + 2 : MAX_QUES_LENGTH + MAX_ANSW_LENGTH + 2]
        answer = torch.sum(answer, dim=1)
        
       
        # Concat CLS with LSTM before pass through FC
        x = torch.cat((x_pool, question, answer), dim=1)
        
        y = self.dropout(x)
        y = self.hidden2dense(x)
        y = self.relu(y)
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
    
    
class BertCNN(BertPreTrainedModel):
    
    def __init__(self, config):
        super(BertCNN, self).__init__(config)
        
        # BERT
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        
        # Classifier
        self.batch_size = BATCH_SIZE
        self.dropout = DROPOUT
        self.hidden_dim = HIDDEN_DIM
        self.relu_dim_1 = RELU_DIM_1
        self.relu_dim_2 = RELU_DIM_2
        
        self.cnn_1 = nn.Conv2d(1, NUM_FILTERS, (KERNEL, EMBEDDING_DIM))
        self.cnn_2 = nn.Conv2d(1, NUM_FILTERS, (KERNEL, EMBEDDING_DIM))

        self.hidden2dense = nn.Linear(OUT_MAXPOOL_DIM, self.relu_dim_1)
        self.dense2label = nn.Linear(self.relu_dim_1, self.config.num_labels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
        self.init_weights()
        
    
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
        
        question = x_sequence[:, 1 : MAX_QUES_LENGTH + 1]
        question = question.unsqueeze(1)  # (N, Ci, W, D)
        question = F.relu(self.cnn_1(question)).squeeze(3)
        question = F.max_pool1d(question, question.size(2)).squeeze(2)
        
        
        answer = x_sequence[:, MAX_QUES_LENGTH + 2 : MAX_QUES_LENGTH + MAX_ANSW_LENGTH + 2]
        answer = answer.unsqueeze(1)  # (N, Ci, W, D)
        answer = F.relu(self.cnn_1(answer)).squeeze(3)
        answer = F.max_pool1d(answer, answer.size(2)).squeeze(2)
        
       
        # Concat CLS with LSTM before pass through FC
        x = torch.cat((x_pool, question, answer), dim=1)
        
        y = self.dropout(x)
        y = self.hidden2dense(x)
        y = self.relu(y)
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
    