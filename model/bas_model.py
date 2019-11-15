import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from config.config import *


class BertBAS(nn.Module):

    def __init__(self):
        super(BertBAS, self).__init__()

        self.batch_size = BATCH_SIZE
        self.dropout = DROPOUT
        self.hidden_dim = HIDDEN_DIM
        self.relu_dim_1 = RELU_DIM_1
        self.relu_dim_2 = RELU_DIM_2
        self.label_size = LABEL_SIZE
        
        
#         self.transformer_config = TRANSFORMER_CONFIG.from_pretrained(MODEL_PATH)
#         self.transformer_model = TRANSFORMER_MODEL.from_pretrained(MODEL_PATH, from_tf=False, config=self.transformer_config)
        self.embedding_dim = EMBEDDING_DIM
        self.transformer_model = nn.Embedding(119547, self.embedding_dim)
        
        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim, bidirectional=True)
        self.hidden = self.init_hidden()
        
        self.hidden2dense = nn.Linear(self.embedding_dim, 1024)
        self.dense2label = nn.Linear(1024, self.label_size)
        self.dropout = nn.Dropout(0.2)
  

    def init_hidden(self):
        # first is the hidden h
        # second is the cell c
        return (Variable(torch.zeros(2, self.batch_size, self.hidden_dim)).to(DEVICE),
                Variable(torch.zeros(2, self.batch_size, self.hidden_dim)).to(DEVICE))

    def forward(self, batch):
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'token_type_ids': batch[2] if MODEL_TYPE in ['bert', 'xlnet'] else None # XLM don't use segment_ids
                 }
#         x = self.transformer_model(**inputs)[1]      # The last hidden-state is the first element of the output tuple
        x = self.transformer_model(batch[0])
        x = x.view(MAX_SEQ_LENGTH, self.batch_size, -1)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        y = self.hidden2dense(lstm_out[-1])
        y = self.dropout(y)
        y = self.dense2label(y)
        y = F.log_softmax(y, dim=1)
        return y
    
    
class BertBiLSTM2Denses(nn.Module):

    def __init__(self):
        super(BertBiLSTM2Denses, self).__init__()

        self.batch_size = BATCH_SIZE
        self.dropout = DROPOUT
        self.hidden_dim = HIDDEN_DIM
        self.relu_dim_1 = RELU_DIM_1
        self.relu_dim_2 = RELU_DIM_2
        self.label_size = LABEL_SIZE
        
        
        self.transformer_config = TRANSFORMER_CONFIG.from_pretrained(MODEL_PATH, num_labels=2, finetuning_task="zalo")
        self.transformer_model = TRANSFORMER_MODEL.from_pretrained(MODEL_PATH, from_tf=False, config=self.transformer_config)
        self.embedding_dim = EMBEDDING_DIM
        
        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim, bidirectional=True)
        self.hidden = self.init_hidden()
        
        self.hidden2dense = nn.Linear(self.hidden_dim*2, self.relu_dim_1)
        self.dense2dense = nn.Linear(self.relu_dim_1, self.relu_dim_2)
        self.dense2label = nn.Linear(self.relu_dim_2, LABEL_SIZE)
        self.dropout = nn.Dropout(0.2)

        

    def init_hidden(self):
        # first is the hidden h
        # second is the cell c
        return (Variable(torch.zeros(2, self.batch_size, self.hidden_dim)).to(DEVICE),
                Variable(torch.zeros(2, self.batch_size, self.hidden_dim)).to(DEVICE))

    def forward(self, batch):
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'token_type_ids': batch[2] if MODEL_TYPE in ['bert', 'xlnet'] else None # XLM don't use segment_ids
                 }
        x = self.transformer_model(**inputs)[0]      # The last hidden-state is the first element of the output tuple
        x = x.view(MAX_SEQ_LENGTH, self.batch_size, -1)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        y = self.hidden2dense(lstm_out[-1])
        y = self.dropout(y)
        y = self.dense2dense(y)
        y = self.dropout(y)
        y = self.dense2label(y)
        y = F.log_softmax(y, dim=1)
        return y