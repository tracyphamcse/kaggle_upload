import torch

MODEL_TYPE = "bert"
CLASSIFIER_TYPES = ["BertForSequenceClassification", "BertLSTM", "BertBow", "BertCNN"]
MODEL_PATH = "model/bert_multilingual_cased_vn_finetuned"


SEED = 1234
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Devices: {}".format(DEVICE))


DATA_DIR = "data/"
FILENAME = {
    "train": ["train.csv"],
    "valid": ["valid.csv"],
}

TRAIN_FILE = "train"
VALID_FILE = "valid"

DO_LOWER_CASE = False
MAX_SEQ_LENGTH = 320
MAX_QUES_LENGTH = 32
MAX_ANSW_LENGTH = 288
BATCH_SIZE = 8

LEARNING_RATE = 1e-5
ADAM_EPS = 1e-8
WARMUP_PROPORTION = 0.25
MAX_GRAD_NORM = 1.0
WEIGHT_DECAY = 0.01

NUM_TRAIN_EPOCHS = 20.0

EMBEDDING_DIM = 768
HIDDEN_DIM = 768
RELU_DIM_1 = 1024
RELU_DIM_2 = 256
BATCH_SIZE = 8
DROPOUT = 0.2
KERNEL = 3
NUM_FILTERS = 200
OUT_MAXPOOL_DIM = 1168

LABEL_LIST = [0, 1]
LABEL_SIZE = 2