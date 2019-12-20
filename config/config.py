import torch

MODEL_TYPE = "bert"
# CLASSIFIER_TYPES = ["BertForSequenceClassification", "BertLSTM", "BertBow", "BertCNN"]
CLASSIFIER_TYPES = ["BertForSequenceClassification"]
MODEL_PATH = "bert-base-uncased"


SEED = 1234
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Devices: {}".format(DEVICE))


DATA_DIR = "/home/tracy/Kaggle/TF2_QA/data/"
FILENAME = {
    "train": ["sub_train_1.csv"],
    "valid": ["sub_val_1.csv"],
    "test" : ["test_1.csv"]
}

TRAIN_FILE = "train"
VALID_FILE = "valid"
TEST_FILE = "test"

DO_LOWER_CASE = True
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

LABEL_LIST = [0, 1, 2, 3]
LABEL_SIZE = len(LABEL_LIST)