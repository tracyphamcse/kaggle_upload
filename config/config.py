import torch

MODEL_TYPE = "bert"
CLASSIFIER_TYPE = "BertBiLSTM_BAS"
MODEL_PATH = "model/bert_multilingual_cased_vn_finetuned"



SEED = 1234
DEVICE = torch.device("cuda")
N_GPU = torch.cuda.device_count()
print("Devices: {}, {}".format(DEVICE, N_GPU))


DATA_DIR = "data/"
FILENAME = {
    "train_0": ["title_split_train_0.csv", "squad_vn.csv"],
    "valid_0": ["title_split_valid_0.csv"],
    
    "train_1": ["title_split_train_1.csv", "squad_vn.csv"],
    "valid_1": ["title_split_valid_1.csv"],
    
    "train_2": ["title_split_train_2.csv", "squad_vn.csv"],
    "valid_2": ["title_split_valid_2.csv"],
    
    "train_3": ["title_split_train_3.csv", "squad_vn.csv"],
    "valid_3": ["title_split_valid_3.csv"],
    
    "train_4": ["title_split_train_4.csv", "squad_vn.csv"],
    "valid_4": ["title_split_valid_4.csv"],
}

TRAIN_FILE = "train_0"
VALID_FILE = "valid_0"

DO_LOWER_CASE = False
MAX_SEQ_LENGTH = 352
MAX_QUES_LENGTH = 32
MAX_ANSW_LENGTH = 320
BATCH_SIZE = 8

# BAS 0.0001 | BERT 2e-5
LEARNING_RATE = 1.5e-5
# XLM 0.000025
ADAM_EPS = 1e-8

WARMUP_PROPORTION = 0.5
MAX_GRAD_NORM = 1.0
WEIGHT_DECAY = 0.01

NUM_TRAIN_EPOCHS = 10.0
LOGGING_STEPS = 50

EMBEDDING_DIM = 768 # BERT 768 | XLM 1280
HIDDEN_DIM = 768
RELU_DIM_1 = 1024
RELU_DIM_2 = 256
BATCH_SIZE = 8
DROPOUT = 0.2

LABEL_LIST = [0, 1]
LABEL_SIZE = 2

SEP_TOKEN = 102