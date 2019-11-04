from config.config import MODEL_PATH, MODEL_CLASSES, MODEL_TYPE, DO_LOWER_CASE

from train.train import train
from train.evaluate import evaluate

from utils.load_data import load_and_cache_examples
from utils.utils import set_seed
from utils.log import get_logger, out_dir
logger = get_logger(__file__.split("/")[-1])

import torch_xla.distributed.data_parallel as dp

def main():
    set_seed()
    logger.info("Load {} model from {}".format(MODEL_TYPE, MODEL_PATH))
    config = MODEL_CLASSES[MODEL_TYPE][0].from_pretrained(MODEL_PATH, num_labels=2, finetuning_task="zalo")
    print (config)

    model = MODEL_CLASSES[MODEL_TYPE][1].from_pretrained(MODEL_PATH, from_tf=False, config=config)
    model = dp.DataParallel(model, device_ids=DEVICE)

    tokenizer = MODEL_CLASSES[MODEL_TYPE][2].from_pretrained(MODEL_PATH, do_lower_case=DO_LOWER_CASE)
    if (MODEL_TYPE == "xlm"):
        tokenizer = MODEL_CLASSES[MODEL_TYPE][2].from_pretrained(MODEL_PATH, do_lowercase_and_remove_accent=DO_LOWER_CASE)

    train_dataset = load_and_cache_examples(tokenizer, "train")
    valid_dataset = load_and_cache_examples(tokenizer, "valid")
    test_dataset = load_and_cache_examples(tokenizer, "test")

    model, global_step, tr_loss = train(train_dataset, valid_dataset, test_dataset, model, tokenizer)
    logger.info("global_step = %s, average loss = %s", global_step, tr_loss)

    eval_loss, f1, result = evaluate(model, tokenizer, test_dataset, "test")
    logger.info("Final evaluate: eval_loss = %s, f1 = %s", eval_loss, f1)
    logger.info(result)

#     logger.info("Save model to {}".format(out_dir))
#     model.save_pretrained(out_dir)


if __name__ == "__main__":
    main()
