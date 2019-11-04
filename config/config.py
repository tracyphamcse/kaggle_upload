import torch
import torch_xla.core.xla_model as xm
from pytorch_transformers import BertConfig, BertForSequenceClassification, BertTokenizer, XLMConfig, XLMForSequenceClassification, XLMTokenizer

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# num_cores = 8
# DEVICE = (xm.get_xla_supported_devices(max_devices=num_cores) if num_cores != 0 else [])
DEVICE = xm.xla_device()
print("Devices: {}".format(DEVICE))

N_GPU = torch.cuda.device_count()
SEED = 1234

DATA_DIR = "data/"
FILENAME = {
    "train": "train_and_squad_vn.csv",
    "valid": "valid.csv",
    "test": "valid.csv",
}
DO_LOWER_CASE = False
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 1

# 5e-6, 2.5e-5, 1.25e-4
LEARNING_RATE = 2.5e-5
# ADAM_EPS = 1e-8
ADAM_EPS = 0.000025
WARMUP_PROPORTION = 0.1
MAX_GRAD_NORM = 1.0
WEIGHT_DECAY = 0.01

NUM_TRAIN_EPOCHS = 4.0
LOGGING_STEPS = 50
EVALUATE_DURING_TRAINING = False

MODEL_TYPE = "xlm"
MODEL_PATH = "model/xlm"
MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'bert_uncased': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlm' : (XLMConfig, XLMForSequenceClassification, XLMTokenizer)
}

#LABEL_LIST = ["0", "1"]
LABEL_LIST = [0, 1]
