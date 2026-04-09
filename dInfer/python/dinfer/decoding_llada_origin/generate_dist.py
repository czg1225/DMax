# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# Modified from LLaDA repos: https://github.com/ML-GSAI/LLaDA

import torch
import numpy as np
import torch.nn.functional as F
import os
from transformers import AutoTokenizer, AutoModel
import torch.distributed as dist

from ..model.modeling_llada import LLaDAModelLM
from .utils import get_num_transfer_tokens, add_gumbel_noise, calculate_op_num
from .parallel_strategy import get_transfer_index, get_transfer_index_dynamic




@ torch.no_grad()
def generate_dist(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0., mask_id=126336, decoding="distributed", **kwargs):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''

    use_cache = kwargs.get("use_cache", False)
    dual_cache = kwargs.get("dual_cache", False)
    use_block = kwargs.get("use_block", False)
    remasking = kwargs.get('remasking', 'low_confidence') 
    threshold = kwargs.get('threshold', None)
    factor = kwargs.get('factor', None)
    
    rank = kwargs.get('rank', dist.get_rank())
    world_size = kwargs.get('world_size', dist.get_world_size())

    if use_block:
        generated_answer, nfe = generate_block_cache(model, prompt, rank, world_size, steps, gen_length, block_length, temperature, remasking, 
            mask_id, threshold, factor)        
    elif use_cache:
        if dual_cache:
            print('***distributed dual cache not tested!')
            generated_answer, nfe = generate_with_dual_cache(model, prompt, rank, world_size, steps, gen_length, block_length, temperature, remasking, 
                mask_id, threshold, factor)
        else:

            generated_answer, nfe = generate_with_cache(model, prompt, rank, world_size, steps, gen_length, block_length, temperature, remasking, 
                mask_id, threshold, factor)
    else:
        generated_answer, nfe = generate(model, prompt, rank, world_size, steps, gen_length, block_length, temperature, remasking, mask_id, threshold, factor)

    return generated_answer, nfe


@ torch.no_grad()
def generate(model, prompt, rank=0, world_size=1, steps=128, gen_length=128, block_length=128, temperature=0.,
             remasking='low_confidence', mask_id=126336, threshold=None, factor=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    op_num = 0
    total_length = prompt.shape[1] + gen_length
    if total_length % world_size != 0:
        total_length = (total_length // world_size + 1) * world_size
    x = torch.full((prompt.shape[0], total_length), mask_id, dtype=torch.long).to(model.device)
    if gen_length==steps:
        steps = total_length-prompt.shape[1]
        last_step = None
    else:
        if gen_length % block_length != 0:
            steps = steps // (gen_length // block_length)
            residual = (total_length - prompt.shape[1]) % block_length
            last_step = max(min(int(residual * steps / block_length)+1, residual), 1)
        else:
            steps = steps // (gen_length // block_length)
            last_step = None
    gen_length = total_length - prompt.shape[1]
    x[:, :prompt.shape[1]] = prompt.clone()

    # assert gen_length % block_length == 0
    num_blocks = (gen_length+block_length-1) // block_length
    
    # print(f'gen_length: {gen_length}, num_blocks: {num_blocks}, steps: {steps}, last_step: {last_step}')



    nfe = 0
    for num_block in range(num_blocks):
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = prompt.shape[1] + (num_block + 1) * block_length
        if num_block == num_blocks-1 and last_step is not None:
            steps = last_step
        block_mask_index = (x[:, current_block_start:current_block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        i = 0
        while (x[:, current_block_start:current_block_end] == mask_id).sum()>0:
            nfe += 1
            mask_index = (x == mask_id)
            mask_index[:, current_block_end:] = 0
            if world_size==1:
                logits = model(x.clone()).logits
                op_num += calculate_op_num(x)
                x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, i] if threshold is None else None, threshold)
                x[transfer_index] = x0[transfer_index]
            else:
                part = x.shape[1] // world_size
                partial_logits = model(x[:, rank*part:(rank+1)*part].clone()).logits
                op_num += calculate_op_num(x[:, rank*part:(rank+1)*part])
                B, L, V = partial_logits.shape
                logits = torch.empty(world_size, B, L, V, device=partial_logits.device, dtype=partial_logits.dtype)
                dist.all_gather_into_tensor(logits, partial_logits)
                logits = logits.permute(1, 0, 2, 3).reshape(B, world_size*L, V)
                x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, i] if threshold is None else None, threshold)
                x[transfer_index] = x0[transfer_index]                
            i += 1
    return x, nfe

@ torch.no_grad()
def generate_with_cache(model, prompt, rank=0, world_size=1, steps=128, gen_length=128, block_length=128, temperature=0.,
             remasking='low_confidence', mask_id=126336, threshold=None, factor=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    op_num = 0
    if prompt.shape[1]%world_size !=0:
        new_prompt = torch.full((prompt.shape[0], prompt.shape[1] + world_size - prompt.shape[1]%world_size), 126081, dtype=prompt.dtype, device=prompt.device)
        new_prompt[:, -prompt.shape[1]:] = prompt
        prompt = new_prompt
    # print(rank, prompt.shape, world_size, prompt.shape[1]%world_size)
        

    total_length = prompt.shape[1] + gen_length
    if total_length % world_size != 0:
        total_length = (total_length // world_size + 1) * world_size
    x = torch.full((prompt.shape[0], total_length), mask_id, dtype=torch.long).to(model.device)
    if gen_length==steps:
        steps = total_length-prompt.shape[1]
        last_step = None
    else:
        if gen_length % block_length != 0:
            steps = steps // (gen_length // block_length)
            residual = (total_length - prompt.shape[1]) % block_length
            last_step = max(min(int(residual * steps / block_length)+1, residual), 1)
        else:
            steps = steps // (gen_length // block_length)
            last_step = None
    gen_length = total_length - prompt.shape[1]
    x[:, :prompt.shape[1]] = prompt.clone()

    num_blocks = (gen_length+block_length-1) // block_length


    nfe = 0
    for num_block in range(num_blocks):
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = prompt.shape[1] + (num_block + 1) * block_length
        current_block_end = min(current_block_end, total_length)
        # print(rank, num_block, current_block_start, current_block_end, total_length)
        if num_block == num_blocks-1 and last_step is not None:
            steps = last_step
        block_mask_index = (x[:, current_block_start:current_block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        mask_index = (x == mask_id)
        mask_index[:, current_block_end:] = 0
        if world_size==1:
            output = model(x.clone(), use_cache=True)
            op_num += calculate_op_num(x)
            past_key_values = output.past_key_values            
            x0, transfer_index = get_transfer_index(output.logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, 0] if threshold is None else None, threshold)
            x[transfer_index] = x0[transfer_index]        
        else:
            part = x.shape[1] // world_size
            output = model(x[:, rank*part:(rank+1)*part].clone(), use_cache=True)
            op_num += calculate_op_num(x[:, rank*part:(rank+1)*part])
            partial_logits = output.logits
            past_key_values = output.past_key_values            
            B, L, V = partial_logits.shape
            if L*world_size<=2048:
                logits = torch.empty(world_size, B, L, V, device=partial_logits.device, dtype=partial_logits.dtype)
                dist.all_gather_into_tensor(logits, partial_logits)
                logits = logits.permute(1, 0, 2, 3).reshape(B, world_size*L, V)
                x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, 0] if threshold is None else None, threshold)
                x[transfer_index] = x0[transfer_index]                 
            else:
                assert part > 256
                if rank == world_size-1:
                    x0, transfer_index = get_transfer_index(partial_logits, temperature, remasking, mask_index[:, -part:], x[:, -part:], num_transfer_tokens[:, 0] if threshold is None else None, threshold)
                else:
                    x0 = torch.empty_like(x[:, -part:])
                    transfer_index = torch.empty_like(x[:, -part:], dtype=torch.bool)
                dist.broadcast(x0, src=world_size-1)
                dist.broadcast(transfer_index, src=world_size-1)
                x[:, -part:][transfer_index] = x0[transfer_index]                 
            

        new_past_key_values = []
        for i in range(len(past_key_values)):
            new_past_key_values.append(())
            for j in range(len(past_key_values[i])):
                new_past_key_values[i] += (past_key_values[i][j][:, :, :current_block_start],)    
        past_key_values = new_past_key_values
        nfe += 1
        i = 1

        while (x[:, current_block_start:current_block_end] == mask_id).sum()>0:
            nfe += 1
            mask_index = (x[:, current_block_start:] == mask_id)
            mask_index[:, block_length:] = 0
            
            new_past_key_values = []
            for ii in range(len(past_key_values)):
                new_past_key_values.append(())
                for jj in range(len(past_key_values[ii])):
                    new_past_key_values[ii] += (past_key_values[ii][jj].clone(),)    
            past_key_values = new_past_key_values
            
            if world_size==1:
                logits = model(x[:, current_block_start:].clone(), past_key_values=past_key_values, use_cache=True).logits
                op_num += calculate_op_num(x[:, current_block_start:], cache_length=total_length)
                x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, x[:, current_block_start:], num_transfer_tokens[:, i] if threshold is None else None, threshold)
                x[:, current_block_start:][transfer_index] = x0[transfer_index].clone()
            else:
                part = (total_length - current_block_start) // world_size
                partial_logits = model(x[:, current_block_start+rank*part:current_block_start+(rank+1)*part].clone(), past_key_values=past_key_values, use_cache=True).logits
                op_num += calculate_op_num(x[:, current_block_start+rank*part:current_block_start+(rank+1)*part], cache_length=total_length)
                B, L, V = partial_logits.shape
                logits = torch.empty(world_size, B, L, V, device=partial_logits.device, dtype=partial_logits.dtype)
                dist.all_gather_into_tensor(logits, partial_logits)
                logits = logits.permute(1, 0, 2, 3).reshape(B, world_size*L, V)
                x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, x[:, current_block_start:], num_transfer_tokens[:, i] if threshold is None else None, threshold)
                x[:, current_block_start:][transfer_index] = x0[transfer_index]                
            i += 1
    return x, nfe

@ torch.no_grad()
def generate_block_cache(model, prompt, rank=0, world_size=1, steps=128, gen_length=128, block_length=128, temperature=0.,
             remasking='low_confidence', mask_id=126336, threshold=None, factor=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    op_num = 0
    if prompt.shape[1]%world_size !=0:
        new_prompt = torch.full((prompt.shape[0], prompt.shape[1] + world_size - prompt.shape[1]%world_size), 126081, dtype=prompt.dtype, device=prompt.device)
        new_prompt[:, -prompt.shape[1]:] = prompt
        prompt = new_prompt

    total_length = prompt.shape[1] + gen_length
    if total_length % world_size != 0:
        total_length = (total_length // world_size + 1) * world_size
    x = torch.full((prompt.shape[0], total_length), mask_id, dtype=torch.long).to(model.device)
    if gen_length==steps:
        steps = total_length-prompt.shape[1]
        last_step = None
    else:
        if gen_length % block_length != 0:
            steps = steps // (gen_length // block_length)
            residual = (total_length - prompt.shape[1]) % block_length
            last_step = max(min(int(residual * steps / block_length)+1, residual), 1)
        else:
            steps = steps // (gen_length // block_length)
            last_step = None
    gen_length = total_length - prompt.shape[1]
    x[:, :prompt.shape[1]] = prompt.clone()

    num_blocks = (gen_length+block_length-1) // block_length

    past_key_values = None
    p = 0
    nfe = 0
    for num_block in range(num_blocks):
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = prompt.shape[1] + (num_block + 1) * block_length
        current_block_end = min(current_block_end, total_length)
        # print(rank, num_block, current_block_start, current_block_end, total_length)
        if num_block == num_blocks-1 and last_step is not None:
            steps = last_step
        block_mask_index = (x[:, current_block_start:current_block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        mask_index = (x == mask_id)
        mask_index[:, current_block_end:] = 0
        past_key_values = None
        if world_size==1:
            output = model(x[:, p:current_block_end].clone(), past_key_values=past_key_values, use_cache=True)
            op_num += calculate_op_num(x[:, p:current_block_end])
            past_key_values = output.past_key_values            
            x0, transfer_index = get_transfer_index(output.logits, temperature, remasking, mask_index[:, p:current_block_end], x[:, p:current_block_end], num_transfer_tokens[:, 0] if threshold is None else None, threshold)
            x[:, p:current_block_end][transfer_index] = x0[transfer_index]        
        else:
            part = (current_block_end-p) // world_size
            output = model(x[:, p+rank*part:p+(rank+1)*part].clone(), use_cache=True)
            op_num += calculate_op_num(x[:, p+rank*part:p+(rank+1)*part])
            partial_logits = output.logits
            past_key_values = output.past_key_values            
            B, L, V = partial_logits.shape
            if L*world_size<=20480:
                logits = torch.empty(world_size, B, L, V, device=partial_logits.device, dtype=partial_logits.dtype)
                dist.all_gather_into_tensor(logits, partial_logits)
                logits = logits.permute(1, 0, 2, 3).reshape(B, world_size*L, V)
                x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index[:, p:current_block_end], x[:, p:current_block_end], num_transfer_tokens[:, 0] if threshold is None else None, threshold)
                x[:, p:current_block_end][transfer_index] = x0[transfer_index]                                 

        new_past_key_values = []
        for i in range(len(past_key_values)):
            new_past_key_values.append(())
            for j in range(len(past_key_values[i])):
                new_past_key_values[i] += (past_key_values[i][j][:, :, :current_block_start].clone(),)

        past_key_values = new_past_key_values
        nfe += 1
        i = 1
        p = current_block_start

        while (x[:, current_block_start:current_block_end] == mask_id).sum()>0:
            nfe += 1
            mask_index = (x[:, current_block_start:current_block_end] == mask_id)
            mask_index[:, block_length:] = 0
            if world_size==1:
                logits = model(x[:, current_block_start:current_block_end].clone(), past_key_values=past_key_values, use_cache=True).logits
                op_num += calculate_op_num(x[:, current_block_start:current_block_end], cache_length=current_block_end)
                x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, x[:, current_block_start:current_block_end], num_transfer_tokens[:, i] if threshold is None else None, threshold)
                x[:, current_block_start:current_block_end][transfer_index] = x0[transfer_index]
            else:
                part = (current_block_end - current_block_start) // world_size
                partial_logits = model(x[:, current_block_start+rank*part:current_block_start+(rank+1)*part].clone(), past_key_values=past_key_values, use_cache=True).logits
                op_num += calculate_op_num(x[:, current_block_start+rank*part:current_block_start+(rank+1)*part], cache_length=current_block_end)
                B, L, V = partial_logits.shape
                logits = torch.empty(world_size, B, L, V, device=partial_logits.device, dtype=partial_logits.dtype)
                # print(rank, logits.shape, partial_logits.shape)
                dist.all_gather_into_tensor(logits, partial_logits)
                logits = logits.permute(1, 0, 2, 3).reshape(B, world_size*L, V)
                # print(rank, logits.shape, mask_index.shape, x[:, current_block_start:].shape, num_transfer_tokens[:, i].shape)
                x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, x[:, current_block_start:current_block_end], num_transfer_tokens[:, i] if threshold is None else None, threshold)
                x[:, current_block_start:current_block_end][transfer_index] = x0[transfer_index]                
            i += 1
    return x, nfe

@ torch.no_grad()
def generate_with_dual_cache(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
            remasking='low_confidence', mask_id=126336, threshold=None, factor=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0  
    for num_block in range(num_blocks):
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length

        block_mask_index = (x[:, current_block_start:current_block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        # cache init and update
        output = model(x, use_cache=True)
        past_key_values = output.past_key_values
        mask_index = (x == mask_id)
        mask_index[:, current_block_end:] = 0
        if factor is None:
            x0, transfer_index = get_transfer_index(output.logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, 0] if threshold is None else None, threshold)
        else:
            x0, transfer_index = get_transfer_index_dynamic(output.logits, temperature, remasking, mask_index, x, None, factor)
        x[transfer_index] = x0[transfer_index]
        nfe += 1

        i = 1
        replace_position = torch.zeros_like(x, dtype=torch.bool)
        replace_position[:, current_block_start:current_block_end] = 1
        while True:
            if (x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                break
            nfe += 1
            mask_index = (x[:, current_block_start:current_block_end] == mask_id)
            # cache position is the position between current_block_start and current_block_end
            logits = model(x[:, current_block_start:current_block_end], past_key_values=past_key_values, use_cache=True, replace_position=replace_position).logits

            if factor is None:
                x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, 
                                                x[:, current_block_start:current_block_end], num_transfer_tokens[:, i] if threshold is None else None, threshold)
            else:
                x0, transfer_index = get_transfer_index_dynamic(logits, temperature, remasking, mask_index, 
                                                x[:, current_block_start:current_block_end], None, factor)
            x[:, current_block_start:current_block_end][transfer_index] = x0[transfer_index]
            i += 1

    return x, nfe
