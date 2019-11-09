from config.config import MODEL_PATH, MODEL_CLASSES, MODEL_TYPE, DO_LOWER_CASE, DEVICE, WEIGHT_DECAY

from train.train import train
from train.evaluate import evaluate

from utils.load_data import load_and_cache_examples
from utils.utils import set_seed
from utils.log import get_logger, out_dir
logger = get_logger(__file__.split("/")[-1])

# import torch_xla.distributed.data_parallel as dp

def main():
    set_seed()
    logger.info("Load {} model from {}".format(MODEL_TYPE, MODEL_PATH))
    config = MODEL_CLASSES[MODEL_TYPE][0].from_pretrained(MODEL_PATH, num_labels=2, finetuning_task="zalo")
    print (config)

    model = MODEL_CLASSES[MODEL_TYPE][1].from_pretrained(MODEL_PATH, from_tf=False, config=config)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': WEIGHT_DECAY},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    # model = dp.DataParallel(model, device_ids=DEVICE)

    tokenizer = MODEL_CLASSES[MODEL_TYPE][2].from_pretrained(MODEL_PATH, do_lower_case=DO_LOWER_CASE)
    if (MODEL_TYPE == "xlm"):
        tokenizer = MODEL_CLASSES[MODEL_TYPE][2].from_pretrained(MODEL_PATH, do_lowercase_and_remove_accent=DO_LOWER_CASE)

    train_dataset = load_and_cache_examples(tokenizer, "train_squad_vn_4")
    valid_dataset = load_and_cache_examples(tokenizer, "valid_4")

    model, global_step, tr_loss = train(train_dataset, valid_dataset, valid_dataset, model, tokenizer, optimizer_grouped_parameters)
    logger.info("global_step = %s, average loss = %s", global_step, tr_loss)

if __name__ == "__main__":
    main()
