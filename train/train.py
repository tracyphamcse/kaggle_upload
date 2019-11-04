from tqdm import tqdm, trange

import torch
from pytorch_transformers import AdamW, WarmupLinearSchedule
from torch.utils.data import DataLoader, RandomSampler
from utils.log import out_dir

from config.config import (BATCH_SIZE, LEARNING_RATE, ADAM_EPS, WARMUP_PROPORTION, WEIGHT_DECAY,
                            NUM_TRAIN_EPOCHS, DEVICE, MAX_GRAD_NORM, LOGGING_STEPS, MODEL_TYPE, EVALUATE_DURING_TRAINING)

from utils.utils import set_seed
from train.evaluate import evaluate
from utils.log import get_logger
logger = get_logger(__file__.split("/")[-1])

def train(train_dataset, valid_dataset, test_dataset, model, tokenizer):

    """ Train the model """
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=BATCH_SIZE)

    # Prepare optimizer and schedule (linear warmup and decay)
    optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE, eps=ADAM_EPS)
    T_TOTAL = int(len(train_dataloader) * NUM_TRAIN_EPOCHS)
    WARMUP_STEP = int((T_TOTAL/10*5) * WARMUP_PROPORTION)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=WARMUP_STEP, t_total=T_TOTAL)
    tracker = xm.RateTracker()

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", NUM_TRAIN_EPOCHS)
    logger.info("  Model = %s", MODEL_TYPE)
    logger.info("  Learning rate = %f", LEARNING_RATE)
    logger.info("  Adam epsilon = %f", ADAM_EPS)
    logger.info("  Warmup step = %f", WARMUP_STEP)
    logger.info("  Total optimization steps = %f", T_TOTAL)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.to(DEVICE)
    model.zero_grad()
    set_seed()  # Added here for reproductibility (even between python 2 and 3)

    best_squad_f1 = 0
    best_zalo_f1 = 0
    best_f1 = 0

    train_iterator = trange(int(NUM_TRAIN_EPOCHS), desc="Epoch", disable=False)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False)
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(DEVICE) for t in batch)
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if MODEL_TYPE in ['bert', 'xlnet'] else None,  # XLM don't use segment_ids
                      'labels':         batch[3]}
            ouputs = model(**inputs)
            loss = ouputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

            tr_loss += loss.item()
            # optimizer.step()
            xm.optimizer_step(optimizer)
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += 1


        valid_loss, valid_f1, _ = evaluate(model, tokenizer, valid_dataset, "valid")

        if (valid_f1 > best_f1):
            best_f1 = valid_f1
            model.save_pretrained(out_dir)
            logger.info("======> SAVE BEST MODEL | F1 = " + str(valid_f1))

    return model, global_step, tr_loss / global_step
