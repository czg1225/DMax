import torch
import numpy as np
from torch._C import dtype
import torch.nn.functional as F
import os
from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForCausalLM
import torch.distributed as dist
import time
import tqdm
from sglang.srt.server_args import ServerArgs
from sglang.srt.layers.moe import initialize_moe_config
from dinfer.model.modeling_llada2_moe_sglang import LLaDA2SGLangLM
from dinfer.decoding.diffusion_runner import ModelRunner
from dinfer.decoding import serving
from queue import Empty
from dinfer.decoding.serving import ServerGroup
from dinfer import BlockIteratorFactory, KVCacheFactory, BlockDiffusionLLM
from dinfer import ThresholdParallelDecoder,CreditThresholdParallelDecoder, HierarchyDecoder, BlockWiseDiffusionLLM, IterSmoothDiffusionLLM, VicinityCacheDiffusionLLM, IterSmoothWithVicinityCacheDiffusionLLM
import logging
import traceback
import json
from multiprocessing import Process
from pathlib import Path
import pytest

from dinfer.model import LLaDA2MoeModelLM
from dinfer import BlockIteratorFactory, KVCacheFactory, SamplingParams, DiffusionLLMServing
from dinfer import ThresholdParallelDecoder, BlockDiffusionLLMAttnmask, BlockDiffusionLLM
import difflib
import time

#model_path = '/mnt/dllm/luxiaocheng/moe-mini-v2-e256-1009-fp8-ml4-grouprouter-20T-mdmcpt-block-diffusion-bl32-4k-noshift-100B'
model_path = '/mnt/infra/dulun.dl/models/dllm-mini/block-diffusion-sft-2k-v2-full-bd/LLaDA2-mini-preview-ep4-v0'
#model_path = '/mnt/infra/dulun.dl/models/dllm-mini/block-diffusion-sft-2k-v2-full-bd/LLaDA2-mini-preview-ep4-v0'
dataset_path = '/ossfs/workspace/dumped_prompts'
dataset='openai_humaneval'

FILE_PATH = Path(__file__).resolve()
sample_path = FILE_PATH.with_name(f"{FILE_PATH.stem[:-17]}_sample.json")

model = None
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
decoder = ThresholdParallelDecoder(temperature=0, threshold=0.9, mask_id=156895, eos_id=156892) 



def test_bd_tpep():
  with open(sample_path, "r") as f:
    samples = json.load(f)
    input_ids = []
    for sample in samples:
      prompt = [sample['question']]
      prompt[0] = '<role>SYSTEM</role>detailed thinking off<|role_end|><role>HUMAN</role>'+prompt[0]+'<|role_end|><role>ASSISTANT</role>' 
      
      input_id = tokenizer(prompt)['input_ids']
      input_ids.append(torch.tensor(input_id))
    
    max_length = max([input_id.shape[1] for input_id in input_ids])
    batch_size = len(input_ids)
    batch_input_ids= torch.zeros((len(input_ids), max_length), dtype=torch.long).fill_(156895)

    print('batch_size:', batch_size, 'input shapes:', [input_id.shape[0] for input_id in input_ids])
    for j in range(len(input_ids)):
        batch_input_ids[j, :input_ids[j].shape[1]] = input_ids[j]

    sample_params1 = SamplingParams(threshold=0.9, cache='prefix', temperature=0., early_stop=True, cont_weight=0, prefix_look=0, 
            after_look=0, warmup_steps=0, enable_torch_compile=True, mask_id=156895, eos_id=156892, parallel_decoding='threshold', 
            use_credit=False, use_bd=True, max_length=max_length+64, ep_size=1, batch_size=batch_size, mini_batch_size=batch_size, use_naive_batching=True)
    dllm_server1 = DiffusionLLMServing(model_path, model_type='llada2-mini', sample_params=sample_params1, server_port=40570, num_gpus=4, dp_size=1, tpep_size=4, backend='sglang')

    out1 = dllm_server1.generate(batch_input_ids, gen_length=64, block_length=32)
    ans1 = []
    for j in range(batch_size):
      ans1.append(out1[j, input_ids[j].shape[1]:])
    #   print(f'========== {j} ==========\n', tokenizer.decode(out1[j], skip_special_tokens=True))
    dllm_server1.stop_serving()


    sample_params2 = SamplingParams(threshold=0.9, cache='prefix', temperature=0., early_stop=True, cont_weight=0, prefix_look=0, 
            after_look=0, warmup_steps=0, enable_torch_compile=True, mask_id=156895, eos_id=156892, parallel_decoding='threshold', 
            use_credit=False, use_bd=True, max_length=max_length+64, ep_size=1, batch_size=batch_size, mini_batch_size=1, use_naive_batching=False)
    dllm_server2 = DiffusionLLMServing(model_path, model_type='llada2-mini', sample_params=sample_params2, server_port=40570, num_gpus=4, dp_size=1, tpep_size=4, backend='sglang')

    out2 = dllm_server2.generate(batch_input_ids, gen_length=64, block_length=32)
    ans2 = []
    for j in range(batch_size):
      ans2.append(out2[j, input_ids[j].shape[1]:])
    dllm_server2.stop_serving()

    sample_params3 = SamplingParams(threshold=0.9, cache='prefix', temperature=0., early_stop=True, cont_weight=0, prefix_look=0, 
            after_look=0, warmup_steps=0, enable_torch_compile=True, mask_id=156895, eos_id=156892, parallel_decoding='threshold', 
            use_credit=False, use_bd=True, max_length=max_length+64, ep_size=1, batch_size=batch_size, mini_batch_size=batch_size, use_naive_batching=False)
    dllm_server3 = DiffusionLLMServing(model_path, model_type='llada2-mini', sample_params=sample_params3, server_port=40570, num_gpus=4, dp_size=1, tpep_size=4, backend='sglang')

    out3 = dllm_server3.generate(batch_input_ids, gen_length=64, block_length=32)
    ans3 = []
    for j in range(batch_size):
      ans3.append(out3[j, input_ids[j].shape[1]:])
    dllm_server3.stop_serving()

    for i in range(len(ans1)):
      matching_portion1 = (ans1[i] == ans2[i]).float().mean()
      print(f"matching_portion 1<->2: {matching_portion1}")
      matching_portion2 = (ans2[i] == ans3[i]).float().mean()
      print(f"matching_portion 2<->3: {matching_portion2}")
      print('ans1:', tokenizer.decode(ans1[i], skip_special_tokens=True))
      print('ans2:', tokenizer.decode(ans2[i], skip_special_tokens=True))
      print('ans3:', tokenizer.decode(ans3[i], skip_special_tokens=True))
      assert matching_portion1 > 0.6
      assert matching_portion2 > 0.6
      # assert(ans1[i] == ans2[i])
    
    return


if __name__ == '__main__':
  test_bd_tpep()
