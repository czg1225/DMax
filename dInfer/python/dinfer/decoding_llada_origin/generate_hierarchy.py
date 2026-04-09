from functools import partial
import math
import torch
import numpy as np
import torch.nn.functional as F
from .utils import get_num_transfer_tokens
from .parallel_strategy import get_transfer_index_hierarchy_remask, get_transfer_index_threshold, get_transfer_index_hierarchy_fast_v2

@ torch.no_grad()
def generate_hierarchy(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             mask_id=126336, decoding="origin", **kwargs):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        mask_id: The toke id of [MASK] is 126336.
        kwargs: additional parameters
    '''
    
    threshold = kwargs.get("threshold", None)

    if decoding == "origin":
        get_transfer_index_cur = get_transfer_index_threshold
    elif decoding == "hierarchy_fast_v2":
        get_transfer_index_cur = partial (get_transfer_index_hierarchy_fast_v2, low_threshold = kwargs["low_threshold"])
    elif decoding == "hierarchy_remasking":
        get_transfer_index_cur = partial (get_transfer_index_hierarchy_remask, low_threshold = kwargs["low_threshold"], remask_threshold = kwargs ["remask_threshold"])
    else:
        raise NotImplementedError (decoding)

    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0
    for num_block in range(num_blocks):
        block_st = prompt.shape[1] + num_block * block_length
        block_ed = prompt.shape[1] + (num_block + 1) * block_length
        x_block = x [:, block_st: block_ed]
        block_mask_index = (x_block == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        i = 0
        while True:
            nfe += 1
            mask_index = (x_block == mask_id)
            logits = model(x).logits
            # mask_index[:, prompt.shape[1] + (num_block + 1) * block_length:] = 0
            logits_block = logits [:, block_st: block_ed]
            x0, transfer_index = get_transfer_index_cur(logits_block, temperature,'low_confidence', mask_index, x_block, 
                num_transfer_tokens[:, i] if threshold is None else None, mask_id, threshold)
            x_block[transfer_index] = x0[transfer_index]
            i += 1
            if (x[:, block_st: block_ed] == mask_id).sum() == 0:
                break
    return x, nfe
