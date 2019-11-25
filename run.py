from config.config import *
from transformers import BertForSequenceClassification
from model.bertforsequence import *

from train.train import train
from train.evaluate import evaluate

from transformers import BertConfig, BertTokenizer

# from utils.load_data import load_and_cache_examples
from utils.load_data_for_bas import load_and_cache_examples
from utils.utils import set_seed
from utils.log import get_logger, out_dir
logger = get_logger(__file__.split("/")[-1])


def main():
    
    TRANSFORMER_MODEL = BertBiLSTM_BAS_2
   
    set_seed()
    
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH, do_lower_case=DO_LOWER_CASE)
    train_dataset = load_and_cache_examples(tokenizer, TRAIN_FILE) 
    valid_dataset = load_and_cache_examples(tokenizer, VALID_FILE) 

    config = BertConfig.from_pretrained(MODEL_PATH, num_labels=2, finetuning_task="zalo")
    model = TRANSFORMER_MODEL.from_pretrained(MODEL_PATH, from_tf=False, config=config)
    print(model)
   
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': WEIGHT_DECAY},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", NUM_TRAIN_EPOCHS)
    logger.info("  Model = %s", MODEL_TYPE)
    logger.info("  Classifier = %s", CLASSIFIER_TYPE)
    logger.info("  Learning rate = %f", LEARNING_RATE)
    logger.info("  Adam epsilon = %f", ADAM_EPS)
    logger.info("  BiLSTM hidden dim = %f", HIDDEN_DIM)
    logger.info("  FC relu dim 1 = %f", RELU_DIM_1)
    logger.info("  FC relu dim 2 = %f", RELU_DIM_2)
    logger.info("  Batch size= %f", BATCH_SIZE)
    logger.info("  Dropout = %f", DROPOUT)
    
    logger.info("  CNN Kernel = %f", KERNEL)
    logger.info("  CNN Filters = %f", NUM_FILTERS)
    logger.info("  Max Pooling Dim = %f", OUT_MAXPOOL_DIM)
    
    
    model, global_step, tr_loss = train(train_dataset, valid_dataset, valid_dataset, model, tokenizer, optimizer_grouped_parameters)
    logger.info("global_step = %s, average loss = %s", global_step, tr_loss)

if __name__ == "__main__":
    main()
