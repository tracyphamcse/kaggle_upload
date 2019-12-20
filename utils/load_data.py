import csv
import os
import sys
import pandas as pd

import torch
from torch.utils.data import TensorDataset
from utils.utils import *
from config.config import *

from utils.log import get_logger
logger = get_logger(__file__.split("/")[-1])


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

class VietQAProcessor(object):
    """Processor for the racism data set."""

    def get_examples(self, prefix):
        """Gets the list of labels for this data set."""
        return self._create_examples(
            self._read_csv(DATA_DIR, FILENAME[prefix]), prefix)

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = remove_nonlatin(line[0])
            text_b = remove_nonlatin(line[1])
            label = line[2]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def _read_csv(cls, data_dir, filename, quotechar=None):
        """Reads a tab separated value file."""
        list_df = []
        for file in filename:
            list_df.append(pd.read_csv(os.path.join(data_dir, file), lineterminator='\n', sep="\t"))
        df = pd.concat(list_df)
        df = df.reset_index()
        del df["index"]
        lines = []
        for i in range(len(df)):
            if "label" in df:
                lines.append([df["question"][i], df["text"][i], df["label"][i]])
            else:
                lines.append([df["question"][i], df["text"][i], 0])
        return lines

def convert_examples_to_features(examples, tokenizer,
                                 cls_token_at_end=False, pad_on_left=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=1, pad_token_segment_id=0,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (1 for BERT, 2 for XLNet)
    """

    label_map = {label : i for i, label in enumerate(LABEL_LIST)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
            
        tokens_a = tokenizer.tokenize(example.text_a)
        if len(tokens_a) > MAX_QUES_LENGTH:
            tokens_a = tokens_a[:(MAX_QUES_LENGTH)]
        len_tokens_a = len(tokens_a)
        len_padding_a = MAX_QUES_LENGTH - len_tokens_a
        if len_tokens_a < MAX_QUES_LENGTH:
            tokens_a = tokens_a + [pad_token] * (MAX_QUES_LENGTH - len_tokens_a)
        tokens_a = tokens_a + [sep_token]
        input_mask_a = [1 if mask_padding_with_zero else 0] * len_tokens_a + [0 if mask_padding_with_zero else 1] * len_padding_a + [1 if mask_padding_with_zero else 0] 
        segment_ids_a = [sequence_a_segment_id] * len_tokens_a + [pad_token_segment_id] * len_padding_a + [sequence_a_segment_id]
        
        
        tokens_b = tokenizer.tokenize(example.text_b)
        if len(tokens_b) > MAX_ANSW_LENGTH:
            tokens_b = tokens_b[:(MAX_ANSW_LENGTH)]
        len_tokens_b = len(tokens_b)
        len_padding_b = MAX_ANSW_LENGTH - len_tokens_b 
        if len_tokens_b < MAX_ANSW_LENGTH:
            tokens_b = tokens_b + [pad_token] * (MAX_ANSW_LENGTH - len_tokens_b)
        tokens_b = tokens_b + [sep_token]
        input_mask_b = [1 if mask_padding_with_zero else 0] * len_tokens_b + [0 if mask_padding_with_zero else 1] * len_padding_b + [1 if mask_padding_with_zero else 0] 
        segment_ids_b = [sequence_b_segment_id] * len_tokens_b + [pad_token_segment_id] * len_padding_b + [sequence_b_segment_id]
        

        if cls_token_at_end:
            tokens_b = tokens_b + [cls_token]
            input_mask_b = input_mask_b + [1 if mask_padding_with_zero else 0]
            segment_ids_b = segment_ids_b + [cls_token_segment_id]
        else:
            tokens_a = [cls_token] + tokens_a
            input_mask_a = [1 if mask_padding_with_zero else 0] + input_mask_a
            segment_ids_a = [cls_token_segment_id] + segment_ids_a
    
        tokens = tokens_a + tokens_b
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = input_mask_a + input_mask_b
        segment_ids = segment_ids_a + segment_ids_b

        assert len(input_ids) == MAX_QUES_LENGTH + MAX_ANSW_LENGTH + 3
        assert len(input_mask) == MAX_QUES_LENGTH + MAX_ANSW_LENGTH + 3
        assert len(segment_ids) == MAX_QUES_LENGTH + MAX_ANSW_LENGTH + 3

        label_id = label_map[example.label]

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features

def load_and_cache_examples(tokenizer, prefix):
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(DATA_DIR, 'cached_{}_{}_{}'.format(
                                    prefix, MODEL_TYPE, MAX_SEQ_LENGTH))

    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", DATA_DIR)
        processor = VietQAProcessor()
        examples = processor.get_examples(prefix)

        features = convert_examples_to_features(examples, tokenizer,
            cls_token_at_end=bool(MODEL_TYPE in ['xlnet']),            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            sep_token=tokenizer.sep_token,
            cls_token_segment_id=2 if MODEL_TYPE in ['xlnet'] else 1,
            pad_on_left=bool(MODEL_TYPE in ['xlnet']),                 # pad on the left for xlnet
            pad_token_segment_id=4 if MODEL_TYPE in ['xlnet'] else 0)

        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset
