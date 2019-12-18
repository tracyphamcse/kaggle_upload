import os
from transformers import BertForSequenceClassification
from transformers import BertConfig, BertTokenizer


from model.bertforsequence import *
from train.train import train
from train.evaluate import evaluate

from config.config import *
from utils.load_data import load_and_cache_examples
from utils.utils import set_seed
from utils.log import get_logger, out_dir
logger = get_logger(__file__.split("/")[-1])


def main():
    
    for CLASSIFIER_TYPE in CLASSIFIER_TYPES:
     
        set_seed()
    
        # Choose model
        if CLASSIFIER_TYPE == "BertForSequenceClassification":
            TRANSFORMER_MODEL = BertForSequenceClassification
            stored_dir = "runs/bert_base"
            
        elif CLASSIFIER_TYPE == "BertLSTM":
            TRANSFORMER_MODEL = BertLSTM
            stored_dir = "runs/bert_lstm"
            
        elif CLASSIFIER_TYPE == "BertBow":
            TRANSFORMER_MODEL = BertBow
            stored_dir = "runs/bert_bow"
            
        elif CLASSIFIER_TYPE == "BertCNN":
            TRANSFORMER_MODEL = BertCNN
            stored_dir = "runs/bert_cnn"
            
        else:
            print ("Model is not supported")
            
        if not os.path.exists(stored_dir):
                os.makedirs(stored_dir)


        # Prepare data
        tokenizer = BertTokenizer.from_pretrained(MODEL_PATH, do_lower_case=DO_LOWER_CASE)
        train_dataset = load_and_cache_examples(tokenizer, TRAIN_FILE) 
        valid_dataset = load_and_cache_examples(tokenizer, VALID_FILE) 
        

        # Load model
        config = BertConfig.from_pretrained(MODEL_PATH, num_labels=2, finetuning_task="zalo")
        model = TRANSFORMER_MODEL.from_pretrained(MODEL_PATH, from_tf=False, config=config)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': WEIGHT_DECAY},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

        
        # Model config
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
        

        # Train
        model, global_step, tr_loss = train(train_dataset, valid_dataset, model, tokenizer, optimizer_grouped_parameters, stored_dir)
        logger.info("global_step = %s, average loss = %s", global_step, tr_loss)
        logger.info("\n\n\n============**************===========\n\n\n")

if __name__ == "__main__":
    main()
