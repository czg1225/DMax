import os
import logging
import random
import pytest
import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoConfig

from vllm import distributed
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config, get_current_vllm_config

from dinfer.model import LLaDAModelLM, LLaDAMoeModelLM
from dinfer import BlockWiseDiffusionLLM, ThresholdParallelDecoder, HierarchyDecoder
from dinfer import DiffusionLLMServing, SamplingParams
from dinfer.decoding.utils import BlockIteratorFactory
from dinfer.decoding.serving import find_continuous_ports, init_generator
from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp8Config
from dinfer.model.modeling_llada2_moe_sglang import LLaDA2SGLangLM
from dinfer.decoding.diffusion_runner import ModelRunner
from dinfer import BlockIteratorFactory, KVCacheFactory, BlockDiffusionLLM
from dinfer import ThresholdParallelDecoder,CreditThresholdParallelDecoder, HierarchyDecoder, BlockWiseDiffusionLLM, IterSmoothDiffusionLLM, VicinityCacheDiffusionLLM, IterSmoothWithVicinityCacheDiffusionLLM

LLADA2_MODEL = '/mnt/infra/dulun.dl/models/LLaDA2.0-MoE-preview/LLaDA2.0-Mini-fusemoe/checkpoint-14845_fusemoe' #mini preview

sample_params = SamplingParams(threshold=0.95, cache='prefix', temperature=0., early_stop=True, cont_weight=0, prefix_look=0, 
        after_look=0, warmup_steps=0, enable_torch_compile=True, mask_id=156895, eos_id=156892, parallel_decoding='threshold', 
        use_credit=False, use_bd=True, max_length=2048)

def get_prompts(tokenizer, mask_id, device, num=1):
    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she can run 6 kilometers per hour. How many kilometers can she run in 8 hours? "
    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
    input_ids1 = torch.tensor(tokenizer(prompt)['input_ids']).to(device).unsqueeze(0)
    len1 = input_ids1.shape[1]

    if num == 2:
        prompt = "Lily can run 12 kilometers per hour for 4 hours. How many kilometers can she run in 4 hours? "
        m = [{"role": "user", "content": prompt}, ]
        prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
        input_ids2 = torch.tensor(tokenizer(prompt)['input_ids']).to(device).unsqueeze(0)
        len2 = input_ids2.shape[1]
        ret = torch.zeros(2, max(len1, len2), dtype=input_ids1.dtype)
        ret[0, 0:len1] = input_ids1
        ret[1, 0:len2] = input_ids2
    else:
        ret = input_ids1

    return ret

def get_reference_response(master_port, input_ids):
    if 'PYTEST_XDIST_WORKER' in os.environ:
        worker_num = int(os.environ['PYTEST_XDIST_WORKER'].replace('gw', ''))
        gpu_id = worker_num % torch.cuda.device_count()
    else:
        gpu_id = 0
    
    torch.cuda.set_device(gpu_id)
    device = torch.device(gpu_id)

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(master_port)
    from sglang.srt import distributed
    distributed.init_distributed_environment(1, 0, 'env://', 0, 'nccl')
    distributed.initialize_model_parallel(1, 1, 1, backend='nccl')
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.layers.moe import initialize_moe_config
    from dinfer.model.modeling_llada2_moe_sglang import LLaDA2SGLangLM
    from dinfer.decoding.diffusion_runner import ModelRunner
    from sglang.srt.layers.dp_attention import initialize_dp_attention
    model_config = AutoConfig.from_pretrained(LLADA2_MODEL, trust_remote_code=True)


    server_args = ServerArgs(model_path=LLADA2_MODEL, enable_dp_attention=True, trust_remote_code=True, tp_size=1, dp_size = 1, pp_size = 1,
                            port=master_port+1, dist_init_addr="127.0.0.1:{}".format(master_port+2))
    try:
        from sglang.srt.server_args import set_global_server_args_for_scheduler
    except ImportError:
        pass
    else:
        set_global_server_args_for_scheduler(server_args)
    initialize_dp_attention(
        server_args=server_args,
        model_config=model_config,
    )
    initialize_moe_config(server_args)
    model = LLaDA2SGLangLM(config=model_config, expert_map_path='.').eval()
    torch.set_default_dtype(torch.bfloat16)
    model.load_weights(LLADA2_MODEL, device=device)
    initialize_moe_config(server_args)
    
    
    model = model.to(device)
    max_length = sample_params.max_length
    model = ModelRunner(model, device, server_args=server_args, max_length=max_length, enable_compile=sample_params.enable_torch_compile)

    dllm = init_generator(model, sample_params, backend='sglang', max_length=max_length)
    ref_res = dllm.generate(input_ids, gen_length=128, block_length=128).cpu()
    
    del dllm
    torch.cuda.empty_cache()
    return ref_res

def test_server_sglang():
    if 'PYTEST_XDIST_WORKER' in os.environ:
        worker_num = int(os.environ['PYTEST_XDIST_WORKER'].replace('gw', ''))
        gpu_id = worker_num % torch.cuda.device_count()
    else:
        gpu_id = 0
    
    device = torch.device(gpu_id)
    torch.cuda.set_device(gpu_id)
    print(f"[test_serving] Initializing MoE Reference on GPU {gpu_id}")
    tokenizer = AutoTokenizer.from_pretrained(LLADA2_MODEL, trust_remote_code=True, local_files_only=True)
    input_ids = get_prompts(tokenizer, mask_id=156895, device=device)   

    # obtain reference response
    port, _ = find_continuous_ports(num_ports=7)
    ref_res = get_reference_response(port, input_ids)


    print('Test sglang serving: DP == 1 and TPEP == 2')
    port, _ = find_continuous_ports(num_ports=7)
    dllm_server = DiffusionLLMServing(LLADA2_MODEL, model_type='llada2-mini', sample_params=sample_params, num_gpus=2, dp_size=1, tpep_size=2, backend='sglang',
                                    start_port=port, end_port=port+7
                                    )
    out1 = dllm_server.generate(input_ids, gen_length=128, block_length=128).cpu()
    assert torch.all(ref_res == out1)
    dllm_server.stop_serving()


    print('Test sglang serving: DP == 2 and TPEP == 2')
    port, _ = find_continuous_ports(num_ports=21)
    input_ids2 = torch.cat([input_ids, input_ids])
    dllm_server = DiffusionLLMServing(LLADA2_MODEL, model_type='llada2-mini', sample_params=sample_params, num_gpus=4, dp_size=2, tpep_size=2, backend='sglang',
                                    start_port=port, end_port=port+21
                                    )
    out2 = dllm_server.generate(input_ids2, gen_length=128, block_length=128).cpu()
    assert torch.all(ref_res[0][ref_res[0] != 156892] == out2[0][out2[0] != 156892])
    dllm_server.stop_serving()

