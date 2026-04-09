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

LLADA_MODEL_PATH = "/mnt/infra/myx/models/LLaDA-1.5/"
MOE_MODEL_PATH = '/mnt/infra/dulun.dl/models/LLaDA-MoE/fusemoe/step45567_converted_hf_fusemoe'

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


@pytest.fixture(scope="function")
def setup_llada_reference():
    """
    Sets up the standard LLaDA model to generate a ground-truth reference 
    before running the server.
    """
    # 1. GPU Selection
    if 'PYTEST_XDIST_WORKER' in os.environ:
        worker_num = int(os.environ['PYTEST_XDIST_WORKER'].replace('gw', ''))
        gpu_id = worker_num % torch.cuda.device_count()
    else:
        gpu_id = 0
    
    device = torch.device(gpu_id)
    torch.cuda.set_device(gpu_id)
    print(f"[test_serving] Initializing LLaDA Reference on GPU {gpu_id}")

    # 2. Init Distributed Env (Required for VLLM internals)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(40000 + random.randint(0, 1000) + gpu_id)
    
    try:
        distributed.init_distributed_environment(1, 0, 'env://', 0, 'nccl')
    except (RuntimeError, AssertionError) as e:
        print(f"Distributed environment already initialized: {e}")
    
    try:
        distributed.initialize_model_parallel(1, backend='nccl')
    except (RuntimeError, AssertionError) as e:
        print(f"Model parallel already initialized: {e}")

    # 3. Load Model
    with set_current_vllm_config(VllmConfig()):
        config = AutoConfig.from_pretrained(LLADA_MODEL_PATH, trust_remote_code=True, local_files_only=True)
        config.flash_attention = True
        config.train_max_sequence_length = 4096
        
        model = LLaDAModelLM.from_pretrained(LLADA_MODEL_PATH, torch_dtype=torch.bfloat16, config=config).eval()
        model = model.to(device)
        
        decoder = ThresholdParallelDecoder(gpu_id, threshold=0.9, use_float64=True)
        tokenizer = AutoTokenizer.from_pretrained(LLADA_MODEL_PATH, trust_remote_code=True, local_files_only=True)
    
    input_ids = get_prompts(tokenizer, mask_id=126336, device=device)
    
    yield model, decoder, tokenizer, input_ids, device
    
    # 4. Cleanup
    print(f"[test_serving] Cleaning up LLaDA Reference on GPU {gpu_id}")
    del model
    del decoder
    torch.cuda.empty_cache()
    try:
        distributed.destroy_model_parallel()
        distributed.destroy_distributed_environment()
    except:
        pass

@pytest.fixture(scope="function")
def setup_moe_reference():
    """
    Sets up the MoE model to generate a ground-truth reference 
    before running the server.
    """
    if 'PYTEST_XDIST_WORKER' in os.environ:
        worker_num = int(os.environ['PYTEST_XDIST_WORKER'].replace('gw', ''))
        gpu_id = worker_num % torch.cuda.device_count()
    else:
        gpu_id = 0
    
    device = torch.device(gpu_id)
    torch.cuda.set_device(gpu_id)
    print(f"[test_serving] Initializing MoE Reference on GPU {gpu_id}")
    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(50000 + random.randint(0, 1000) + gpu_id)
    
    try:
        distributed.init_distributed_environment(1, 0, 'env://', 0, 'nccl')
    except (RuntimeError, AssertionError) as e:
        print(f"Distributed environment already initialized: {e}")
    
    try:
        distributed.initialize_model_parallel(1, backend='nccl')
    except (RuntimeError, AssertionError) as e:
        print(f"Model parallel already initialized: {e}")
    
    parallel_config = ParallelConfig(enable_expert_parallel=True)
    with set_current_vllm_config(VllmConfig(parallel_config=parallel_config)):
        model_config = AutoConfig.from_pretrained(MOE_MODEL_PATH, trust_remote_code=True, local_files_only=True)
        model = LLaDAMoeModelLM(config=model_config).eval()
        model.load_weights(MOE_MODEL_PATH, torch_dtype=torch.bfloat16)
        model = model.to(device)
    
    decoder = ThresholdParallelDecoder(0, threshold=0.9, mask_id=156895, eos_id=156892, use_float64=True)
    tokenizer = AutoTokenizer.from_pretrained(MOE_MODEL_PATH, trust_remote_code=True, local_files_only=True)
    input_ids = get_prompts(tokenizer, mask_id=156895, device=device)
    
    yield model, decoder, tokenizer, input_ids, device
    
    print(f"[test_serving] Cleaning up MoE Reference on GPU {gpu_id}")
    del model
    del decoder
    torch.cuda.empty_cache()
    try:
        distributed.destroy_model_parallel()
        distributed.destroy_distributed_environment()
    except:
        pass


def test_llada_server(setup_llada_reference):
    model, decoder, tokenizer, input_ids, device = setup_llada_reference
    print('test serving of standard diffusion LLaDA')
    
    # 1. Generate Reference Result
    params = SamplingParams(temperature=0, threshold=0.9, mask_id=126336, eos_id=126081, early_stop=True, cache='', cont_weight=0, enable_torch_compile=True, use_bd=False)
    dllm = BlockWiseDiffusionLLM(model, decoder, BlockIteratorFactory(), early_stop=True)
    res1 = dllm.generate(input_ids, gen_length=256, block_length=32).cpu()
    
    del model
    torch.cuda.empty_cache()

    # 2. Test Serving: DP == 1 and TPEP == 1
    print('Test serving: DP == 1 and TPEP == 1')
    llm = DiffusionLLMServing(model=LLADA_MODEL_PATH, model_type='llada', backend='vllm', sample_params=params, num_gpus=1, server_port=random.randint(40000, 50000))
    
    try:
        res = llm.generate(input_ids, gen_length=256, block_length=32)
    finally:
        llm.stop_serving()
        
    assert res.shape == res1.shape
    res1 = res1.to(res.device)
    assert torch.all(res == res1)

def test_moe_server(setup_moe_reference):
    print('test serving of diffusion-MOE')
    model, decoder, tokenizer, input_ids, device = setup_moe_reference
    params = SamplingParams(temperature=0, threshold=0.9, mask_id=156895, eos_id=156892, early_stop=True, cache='', cont_weight=0, enable_torch_compile=False, use_bd=False)

    # 1. Generate Reference Result
    # setup EP context for reference generation
    parallel_config = ParallelConfig(enable_expert_parallel=True)
    with set_current_vllm_config(VllmConfig(parallel_config=parallel_config)):
        dllm = BlockWiseDiffusionLLM(model, decoder, BlockIteratorFactory(), early_stop=True)
        res1 = dllm.generate(input_ids, gen_length=256, block_length=32).cpu()
    
    # Release reference model memory
    del model
    torch.cuda.empty_cache()

    # 2. Test Serving: DP == 1 and TPEP == 1
    print('Test serving: DP == 1 and TPEP == 1')
    llm = DiffusionLLMServing(model=MOE_MODEL_PATH, model_type='llada-moe', backend='vllm', sample_params=params, num_gpus=1, server_port=random.randint(50000, 60000))
    try:
        res = llm.generate(input_ids, gen_length=256, block_length=32)
        assert res.shape == res1.shape
        res1 = res1.to(res.device)
        assert torch.all(res == res1)
    finally:
        llm.stop_serving()

    # 3. Test Serving: DP == 2 and TPEP == 1
    input_ids2 = torch.cat([input_ids, input_ids])
    print('Test serving: DP == 2 and TPEP == 1')
    llm = DiffusionLLMServing(model=MOE_MODEL_PATH, model_type='llada-moe', backend='vllm', sample_params=params, num_gpus=2, dp_size=2, tpep_size=1, server_port=random.randint(50000, 60000))
    try:
        res2 = llm.generate(input_ids2, gen_length=256, block_length=32)
        # Remove EOS and padding tokens before comparison
        assert torch.all(res2[0][res2[0] != 156892] == res[0][res[0] != 156892])
    finally:
        llm.stop_serving()

    # 4. Test Serving: DP == 2 and TPEP == 2
    print('Test serving: DP == 2 and TPEP == 2 (2 GPUs)')
    llm = DiffusionLLMServing(model=MOE_MODEL_PATH, model_type='llada-moe', backend='vllm', sample_params=params, num_gpus=2, dp_size=1, tpep_size=2, server_port=random.randint(50000, 60000))
    try:
        res = llm.generate(input_ids, gen_length=256, block_length=32)
    finally:
        llm.stop_serving()

    print('Test serving: DP == 2 and TPEP == 2 (4 GPUs)')
    input_ids2 = torch.cat([input_ids, input_ids])
    llm = DiffusionLLMServing(model=MOE_MODEL_PATH, model_type='llada-moe', backend='vllm', sample_params=params, num_gpus=4, dp_size=2, tpep_size=2, server_port=random.randint(40000, 50000))
    try:
        res2 = llm.generate(input_ids2, gen_length=256, block_length=32)
        assert torch.all(res2[0][res2[0] != 156892] == res[0][res[0] != 156892])
    finally:
        llm.stop_serving()