from functools import partial
import math
import torch
import numpy as np
from torch._prims_common import suggest_memory_format
import torch.nn.functional as F

from .parallel_strategy import get_transfer_index_threshold
from .utils import get_num_transfer_tokens


@ torch.no_grad()
def generate_merge (model, prompt, kvcache, steps=128, gen_length=128, block_length=128, temperature=0.,
             mask_id=126336, eos_id=126081, decoding = "merge", **kwargs):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        mask_id: The toke id of [MASK] is 126336(in llada) or 156895(llada moe).
        kwargs: additional parameters
    '''
    threshold = kwargs.get("threshold", None)
    
    parallel_decoding = kwargs.get("parallel_decoding", "threshold")
    early_stop = kwargs.get("early_stop", False)
    # added sliding window parameters
    pre_block_len = kwargs.get("prefix_look", 0)
    suf_block_len = kwargs.get("after_look", 0)
    warm_up_times = kwargs.get("warmup_times", 4)
    rank = kwargs.get("rank", 0)
    print(f'gen_len={gen_length}, block_len={block_length}')
    print(kwargs)
    
    use_sw = True
    
    if parallel_decoding == "threshold":
        get_transfer_index_cur = get_transfer_index_threshold
    else:
        raise NotImplementedError (parallel_decoding)

    prompt_len = prompt.shape[1]
    x = torch.full((1, prompt_len + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt_len] = prompt.clone()

    num_blocks = gen_length // block_length
    if gen_length % block_length:
        num_blocks += 1
        prompt_shift = min(block_length - gen_length % block_length, prompt_len)
        decoding_st = prompt_len - prompt_shift
    else:
        decoding_st = prompt_len

    #assert steps % num_blocks == 0
    steps = steps // num_blocks
    expected_tpf = kwargs.get("expected_tpf", 8)
    maximum_unroll_k = kwargs.get("maximum_unroll_k", 1)
    expected_unroll_k = max(min(maximum_unroll_k, block_length//expected_tpf), 1)

    nfe = 0
    if kvcache is None:
        for num_block in range(num_blocks):
            unroll_i = 0
            unroll_k = expected_unroll_k
            block_st = decoding_st + num_block * block_length
            block_ed = min(decoding_st + (num_block + 1) * block_length, gen_length + prompt_len)

            x_block = x [:, block_st: block_ed]
            block_mask_index = (x_block == mask_id)
            num_mask = block_mask_index.sum()
            if num_mask == 0:
                break
            unroll_k = max(min(expected_unroll_k, num_mask//expected_tpf), 1)
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

            i = 0
            while True:
                unroll_i += 1
                nfe += 1
                mask_index = (x_block == mask_id)
                logits = model(x).logits
                logits_block = logits [:, block_st: block_ed].clone()
                x0, transfer_index = get_transfer_index_cur(logits_block, temperature, mask_index, x_block, 
                    num_transfer_tokens=int(num_transfer_tokens[:, i]) if threshold is None else None, mask_id=mask_id, threshold=threshold, num_call=i+1)
                transfer_index = torch.logical_and(transfer_index, mask_index)
                x [:, block_st: block_ed] = torch.where(transfer_index, x0, x_block)
                #if rank == 0:
                #    print(f'block {num_block}:', x)
                i += 1
                if unroll_i%unroll_k==0:
                    if early_stop and (x == eos_id).any():
                        pos = torch.arange(x.shape[1], device = x.device).unsqueeze(0)
                        eos_mask = (x == eos_id)
                        first_eos = torch.where(eos_mask, pos, x.shape[1]).amin(dim=1)
                        after_first = pos > first_eos.unsqueeze(1)

                        x[after_first] = eos_id
                    num_mask = (x[:, block_st: block_ed] == mask_id).sum().cpu().item()
                    if num_mask == 0:
                        break
                    else:
                        unroll_k = max(min(unroll_k, num_mask//expected_tpf), 1)
    else:
        for num_block in range(num_blocks):
            unroll_i = 0
            block_st = decoding_st + num_block * block_length
            block_ed = min(decoding_st + (num_block + 1) * block_length, gen_length + prompt_len)
            pre_block_st = max(decoding_st, decoding_st + num_block * block_length - pre_block_len) 
            suf_block_ed = min(prompt_len + gen_length, block_ed + suf_block_len)

            x_block = x[:, block_st: block_ed]
            block_mask_index = (x_block == mask_id)
            num_mask = block_mask_index.sum().cpu().item()
            if num_mask == 0:
                break
            if num_block > 2:
                expected_unroll_k = 2
            unroll_k = max(min(expected_unroll_k, num_mask//expected_tpf), 1)
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
            i = 0
            flag_new_block = True
            
            while True:
                unroll_i += 1
                nfe += 1
                mask_index = (x_block == mask_id)
                
    
                if use_sw:
                    if num_block == 0:
                        if i < warm_up_times:
                            logits_block = model(x).logits[:, block_st: block_ed].clone()
                        else:
                            logits_block = kvcache.update(x, block_st, block_ed, flag_new_block).clone()
                            flag_new_block = False
                    else:
                        logits_block = kvcache.update(x, pre_block_st, suf_block_ed, flag_new_block)
                        logits_block = logits_block[:, block_st - pre_block_st: block_ed - pre_block_st].clone()
                        flag_new_block = False
                else:
                    logits_block = kvcache.update(x, pre_block_st, suf_block_ed, flag_new_block).clone()
                    flag_new_block = False
                
                x0, transfer_index = get_transfer_index_cur(logits_block, temperature, mask_index, x_block, 
                    num_transfer_tokens=int(num_transfer_tokens[:, i]) if threshold is None else None, mask_id=mask_id, threshold=threshold, num_call=i+1)
                transfer_index = torch.logical_and(transfer_index, mask_index)
                x [:, block_st: block_ed] = torch.where(transfer_index, x0, x_block)
                torch.distributed.broadcast(x, 0)
                i += 1
         
                if unroll_i%unroll_k==0:
                    unroll_i = 0
                    if early_stop and (x == eos_id).any():
                        pos = torch.arange(x.shape[1], device = x.device).unsqueeze(0)
                        eos_mask = (x == eos_id)
                        first_eos = torch.where(eos_mask, pos, x.shape[1]).amin(dim=1)
                        after_first = pos > first_eos.unsqueeze(1)


                        x[after_first] = eos_id
                    
                    num_mask = (x[:, block_st: block_ed] == mask_id).sum().cpu().item()
                    if num_mask == 0:
                        break
                    else:
                        unroll_k = max(min(unroll_k, num_mask//expected_tpf), 1)
    return x, nfe
