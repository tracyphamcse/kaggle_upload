import random
import numpy as np
import torch
from config.config import SEED, N_GPU

def set_seed():
    # Set random seed to all
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if N_GPU > 0:
        torch.cuda.manual_seed_all(SEED)

def truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
