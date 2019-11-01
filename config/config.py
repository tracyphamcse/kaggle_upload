import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_GPU = torch.cuda.device_count()
SEED = 1234

DATA_DIR = "data/"
FILENAME = {
    "train": "augmented_squad_vn.csv",
    "valid": "valid.csv",
    "test": "valid.csv",
}
DO_LOWER_CASE = False
MAX_SEQ_LENGTH = 512
BATCH_SIZE = 8

LEARNING_RATE = 2e-5
ADAM_EPS = 1e-8
WARMUP_PROPORTION = 0.1
MAX_GRAD_NORM = 1.0
WEIGHT_DECAY = 0.00

NUM_TRAIN_EPOCHS = 5.0
LOGGING_STEPS = 50
EVALUATE_DURING_TRAINING = False

MODEL_PATH = "model"
LABEL_LIST = [0, 1]

