from tqdm import tqdm
import numpy as np
import os

import torch
from torch.utils.data import DataLoader, SequentialSampler
from sklearn.metrics import f1_score, classification_report

from config.config import BATCH_SIZE, DEVICE
from utils.load_data import load_and_cache_examples

from utils.log import get_logger, out_dir
logger = get_logger(__file__.split("/")[-1])

def evaluate(model, tokenizer, eval_dataset, prefix):

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=BATCH_SIZE)

    # Eval
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", BATCH_SIZE)

    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None

    for batch in eval_dataloader:
        model.eval()
        batch = tuple(t.to(DEVICE) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels':         batch[3]}

            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs['labels'].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=1)

    result = classification_report(preds, out_label_ids)
    f1 = f1_score(preds, out_label_ids)

    logger.info("***** Eval results {} *****".format(prefix))
    logger.info("F1 = {}\n".format(f1))
    logger.info(result)
    logger.info("\n===========================\n")

    return eval_loss, f1, result
