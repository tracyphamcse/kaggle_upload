import torch
from pytorch_transformers import BertConfig, BertForSequenceClassification, BertTokenizer, XLMConfig, XLMForSequenceClassification, XLMTokenizer
# import torch_xla.core.xla_model as xm

# num_cores = 8
# DEVICE = (xm.get_xla_supported_devices(max_devices=num_cores) if num_cores != 0 else [])
# DEVICE = xm.xla_device()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Devices: {}".format(DEVICE))

N_GPU = torch.cuda.device_count()
SEED = 1234

DATA_DIR = "data/"
FILENAME = {
    "train_squad_vn_0": ["new_split_train_0.csv", "squad_vn.csv"],
    "valid_0": ["new_split_valid_0.csv"],
    "train_squad_vn_1": ["new_split_train_1.csv", "squad_vn.csv"],
    "valid_1": ["new_split_valid_1.csv"],
    "train_squad_vn_2": ["new_split_train_2.csv", "squad_vn.csv"],
    "valid_2": ["new_split_valid_2.csv"],
    "train_squad_vn_3": ["new_split_train_3.csv", "squad_vn.csv"],
    "valid_3": ["new_split_valid_3.csv"],
    "train_squad_vn_4": ["new_split_train_4.csv", "squad_vn.csv"],
    "valid_4": ["new_split_valid_4.csv"],
}
DO_LOWER_CASE = False
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 8

# 5e-6, 2.5e-5, 1.25e-4
LEARNING_RATE = 2e-5
# LEARNING_RATE = 1.
MIN_LR = 2e-5
MAX_LR = 5e-5
ADAM_EPS = 1e-8
# ADAM_EPS = 0.000025
WARMUP_PROPORTION = 0.5
MAX_GRAD_NORM = 1.0
WEIGHT_DECAY = 0.01

NUM_TRAIN_EPOCHS = 4.0
LOGGING_STEPS = 50
EVALUATE_DURING_TRAINING = False

# MODEL_TYPE = "bert_multilingual"
# MODEL_PATH = "model/bert_multilang_round3"
#
# MODEL_TYPE = "bert_vietnamese"
# MODEL_PATH = "model/bert_vietnamese_round17"

MODEL_TYPE = "xlm"
MODEL_PATH = "model/xlm"

MODEL_CLASSES = {
    'bert_multilingual': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'bert_vietnamese': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlm' : (XLMConfig, XLMForSequenceClassification, XLMTokenizer)
}

#LABEL_LIST = ["0", "1"]
LABEL_LIST = [0, 1]
