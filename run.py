from config.config import MODEL_PATH, DO_LOWER_CASE
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer

from train.train import train
from train.evaluate import evaluate

from utils.load_data import load_and_cache_examples
from utils.utils import set_seed
from utils.log import get_logger, out_dir
logger = get_logger(__file__.split("/")[-1])

MODEL_TYPE = "bert"

def main():
    set_seed()
    logger.info("Load {} model from {}".format(MODEL_TYPE, MODEL_PATH))
    config = BertConfig.from_pretrained(MODEL_PATH, num_labels=2, finetuning_task="zalo")
    print (config)
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH, from_tf=False, config=config)
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH, do_lower_case=DO_LOWER_CASE)

    train_dataset = load_and_cache_examples(tokenizer, "train")
    valid_dataset = load_and_cache_examples(tokenizer, "valid")
    test_dataset = load_and_cache_examples(tokenizer, "valid")

    model, global_step, tr_loss = train(train_dataset, valid_dataset, test_dataset, model, tokenizer)
    logger.info("global_step = %s, average loss = %s", global_step, tr_loss)


if __name__ == "__main__":
    main()
