import random
from tqdm import tqdm
import pandas as pd
import json

import torch
from torch.utils.data import DataLoader, SequentialSampler
from transformers import BertForSequenceClassification
from transformers import BertConfig, BertTokenizer
from model.bertforsequence import *

from config.config import *
from utils.load_data import *
from utils.utils import *


def main():
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH, do_lower_case=DO_LOWER_CASE)
    data = json.load(open("data/test.json",encoding='utf-8'))
    
    examples = []
    output_ids = []
    
    for testcase in data:

        test_id = testcase["__id__"]
        text_a = testcase["question"]

        for paragraph in testcase["paragraphs"]:
            answer_id = paragraph["id"]
            text_b = testcase["title"] + " . " + paragraph["text"]
            output_ids.append({"test_id": test_id, "answer_id": answer_id, "question": text_a, "answer": text_b})

            example = InputExample(guid="", text_a=remove_nonlatin(text_a), text_b=remove_nonlatin(text_b), label=0)
            examples.append(example)

            t = text_b.split(" . ")
            t = [x.strip(".") for x in t]

            if len(t) == 3:
                t = [t[0], t[2], t[1]]
                output_ids.append({"test_id": test_id, "answer_id": answer_id, "question": text_a, "answer": " . ".join(t) + " ."})
                example = InputExample(guid="", text_a=remove_nonlatin(text_a), text_b=remove_nonlatin(" . ".join(t) + " ."), label=0)
                examples.append(example)
            elif len(t) > 3:
                current_text = [" . ".join(t) + " ."]
                i = 0
                while (True):

                    if i == 5:
                        break
                    temp_1 = t[1:]
                    random.shuffle(temp_1)
                    temp_2 = [t[0]]
                    temp_2.extend(temp_1)
                    temp_text = " . ".join(temp_2) + " ."

                    if temp_text not in current_text:
                        output_ids.append({"test_id": test_id, "answer_id": answer_id, "question": text_a, "answer": temp_text})
                        example = InputExample(guid="", text_a=remove_nonlatin(text_a), text_b=remove_nonlatin(temp_text), label=0)
                        examples.append(example)
                        current_text.append(temp_text)
                        i += 1

    
    features = convert_examples_to_features(examples, tokenizer,
            cls_token_at_end=bool(MODEL_TYPE in ['xlnet']),           
            cls_token=tokenizer.cls_token,
            sep_token=tokenizer.sep_token,
            cls_token_segment_id=2 if MODEL_TYPE in ['xlnet'] else 1,
            pad_on_left=bool(MODEL_TYPE in ['xlnet']),                
            pad_token_segment_id=4 if MODEL_TYPE in ['xlnet'] else 0)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    
    
    
    for CLASSIFIER_TYPE in CLASSIFIER_TYPES:
        set_seed()
        distill_sampler = SequentialSampler(dataset)
        distill_dataloader = DataLoader(dataset, sampler=distill_sampler, batch_size=BATCH_SIZE)
    
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
        config = BertConfig.from_pretrained(stored_dir, num_labels=2, finetuning_task="zalo")
        model = TRANSFORMER_MODEL.from_pretrained(stored_dir, from_tf=False, config=config)
         
        model.to(DEVICE)
        model.eval()
        i = 2

        out_sent = []
        out_logit_0 = []
        out_logit_1 = []

        out_label = []
        for batch in tqdm(distill_dataloader):
            batch = tuple(t.to(DEVICE) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                              'attention_mask': batch[1],
                              'token_type_ids': batch[2] if MODEL_TYPE in ['bert', 'xlnet'] else None,
                              'labels':         batch[3]}
                try:
                    model.batch_size = len(batch[0])
                    model.lstm_hidden_1 = model.init_hidden()
                    model.lstm_hidden_2 = model.init_hidden()
                except:
                    pass
                outputs = model(**inputs)
                _, logits = outputs[:2]
                out_label.extend([0 if x[0] > x[1] else 1 for x in logits])
        for k, l in zip(output_ids, out_label):
            k[stored_dir] = l
        
    df = pd.DataFrame(output_ids)
    df.loc[:,'total'] = df.sum(numeric_only=True, axis=1)
    out = []
    for i in range(len(df)):
        if df["total"][i] > 2:
            out.append(df["test_id"][i] + "," + df["answer_id"][i])
    out = list(set(out))
    
    with open("data/submission.csv", "w") as f:
        for line in out:
            f.write(line + "\n")
        

if __name__ == "__main__":
    main()