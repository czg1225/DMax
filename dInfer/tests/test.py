import os
import logging
from multiprocessing import Process
import random
import pytest

import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModel, AutoConfig
from vllm.config import CompilationConfig, ParallelConfig
from vllm.config import VllmConfig, set_current_vllm_config, get_current_vllm_config

from dinfer.model import LLaDAMoeModelLM, LLaDAModelLM
from dinfer import BlockWiseDiffusionLLM, VicinityCacheDiffusionLLM, IterSmoothDiffusionLLM, IterSmoothWithVicinityCacheDiffusionLLM, BlockWiseDiffusionLLMWithSP
from dinfer import ThresholdParallelDecoder, HierarchyDecoder
from dinfer import DiffusionLLMServing, SamplingParams

from dinfer.model.modeling_llada_fastdllm import LLaDAModelLM as LLaDAModelLM_fastdllm
from dinfer.decoding.generate_fastdllm import generate, generate_with_prefix_cache, generate_with_dual_cache
from dinfer.decoding.generate_dist import generate as generate_sp
from dinfer.decoding.generate_uniform import BaseDiffusionIteration
from dinfer.decoding.generate_hierarchy import generate_hierarchy
from dinfer.decoding.utils import TokenArray, DistAlignedTokenArray, BlockIterator, BlockIteratorFactory, KVCacheFactory, gather_sequence_block, BlockLoc
from dinfer.decoding.utils import DiffusionKVCacheManager
from dinfer.decoding.generate_merge import generate_merge
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from test_generate import IterSmoothDiffusionLLM as IterSmoothDiffusionLLM_test
from test_generate import IterSmoothWithVicinityCacheDiffusionLLM as IterSmoothWithVicinityCacheDiffusionLLM_test

model_path = "/mnt/dllm/model_hub/LLaDA-1.5/"
# model_path = "/data/myx/llm/vllm/model/LLaDA-1_5/"
moe_model_path = '/mnt/dllm/fengling/moe/workdir/7bA1b_anneal_15t_0827_500B_further_8k_enneal_train_4k_ep3_v7_1e-5/step45567_converted_hf_fusemoe'
# moe_model_path = '/data/dulun/models/llada-moe-sft/llada-moe-sft-model/7bA1b_anneal_19t_500B_further_8k_anneal_train_4k_ep3_v8p5/step45567_converted_hf_fusemoe/'

def test_block_iterator():
    prompt = torch.tensor([1, 2, 3, 4, 5, 6, 7]).view(1, 7)
    x = TokenArray(prompt, gen_length=10, mask_id=17, eos_id=18, device='cpu')
    it = BlockIterator(x, block_length=5)
    num_iters = 0
    for block_id, (block_loc, block) in enumerate(it):
        num_iters += 1
        assert block_loc.start == block_id * 5 + prompt.shape[1]
        assert block_loc.end == min((block_id + 1) * 5 + prompt.shape[1], prompt.shape[1] + 10)
    assert num_iters == 2

def test_token_array():
    prompt = torch.tensor([1, 2, 3, 4, 5, 6, 7]).view(1, 7)
    arr = TokenArray(prompt, gen_length=20, mask_id=17, eos_id=18, device='cpu')
    assert arr.total_length == prompt.shape[1] + 20
    assert torch.all(arr[:, 0:5] == prompt[:, 0:5])
    arr[:, 8:10] = torch.tensor([9, 10]).view(1, 2)

    arr = DistAlignedTokenArray(prompt, gen_length=20, mask_id=17, eos_id=18, device='cpu', rank=0, world_size=4)
    assert arr.total_length == prompt.shape[1] + 20 + 1
    assert torch.all(arr[:, 0:5] == prompt[:, 0:5])
    arr[:, 8:10] = torch.tensor([9, 10]).view(1, 2)

class SimulateBlockIterator:
    """ This class simulates the block iterator in VicinityCacheDiffusionLLM.
    """
    def __init__(self, x, block_length, mask_id):
        self.x = x
        self.iter = 0
        self.block_length = block_length
        self.mask_id = mask_id

    def __iter__(self):
        self.iter = 0
        return self

    def move_next(self):
        current_block_start = self.x.prompt.shape[1] + self.iter * self.block_length
        current_block_end = current_block_start + self.block_length
        current_block_end = min(current_block_end, self.x.total_length)
        # If all tokens have been decoded, move to the next block.
        if torch.all(self.x[:, current_block_start:current_block_end] != self.mask_id):
            self.iter += 1

    def __next__(self):
        self.move_next()
        current_block_start = self.x.prompt.shape[1] + self.iter * self.block_length
        if current_block_start >= self.x.total_length:
            raise StopIteration
        current_block_end = current_block_start + self.block_length
        current_block_end = min(current_block_end, self.x.total_length)
        return BlockLoc(current_block_start, current_block_end), self.x[:, current_block_start:current_block_end]

class SimulateBlockIteratorFactory:
    def create(self, x, block_length):
        return SimulateBlockIterator(x, block_length, 126336)

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

@ torch.no_grad()
def check_iteration(model, decoder, input_ids):
    print('check_iteration start')
    x1 = TokenArray(input_ids, 256, decoder.mask_id, decoder.eos_id, model.device)
    x2 = TokenArray(torch.cat([input_ids, input_ids]), 256, decoder.mask_id, decoder.eos_id, model.device)
    kv_cache1 = DiffusionKVCacheManager(cache_type='dual')
    kv_cache2 = DiffusionKVCacheManager(cache_type='dual')
    it1 = BlockIterator(x1, 32)
    it2 = BlockIterator(x2, 32)
    diff_iter1 = BaseDiffusionIteration()
    diff_iter2 = BaseDiffusionIteration()
    model = model.to(torch.float32)
    for block_id, ((block_loc1, block1), (block_loc2, block2)) in enumerate(zip(it1, it2)):
        output1 = model(x1.data, use_cache=True, output_hidden_states=True)
        output2 = model(x2.data, use_cache=True, output_hidden_states=True)
        for i in range(16):
            assert torch.all(output1.past_key_values._data[i] == output2.past_key_values._data[i][0])
            assert torch.all(output1.hidden_states[i][0] == output2.hidden_states[i][0]), output1.hidden_states[i][0] - output2.hidden_states[i][0]

        updated_cache1, logits1 = diff_iter1.forward(model, decoder, x1, kv_cache1, block1, block_loc1, block_id)
        updated_cache2, logits2 = diff_iter2.forward(model, decoder, x2, kv_cache2, block2, block_loc2, block_id)
        assert torch.all(logits1 == logits2)
        for layer_id, (layer1, layer2) in enumerate(zip(kv_cache1.past_key_values._data, kv_cache2.past_key_values._data)):
            assert torch.all(layer1 == layer2)
        assert torch.all(x1.data[0] == x2.data[0])
    print('check_iteration end')

def test_moe_diffusion():
    torch.cuda.set_device(0)
    device = torch.device(0)

    decoder = ThresholdParallelDecoder(0, threshold=0.9, mask_id=156895, eos_id=156892, use_float64=True)
    h_decoder = HierarchyDecoder(0, threshold=0.9, mask_id=156895, eos_id=156892, low_threshold=0.4)
    tokenizer = AutoTokenizer.from_pretrained(moe_model_path, trust_remote_code=True)
    input_ids = get_prompts(tokenizer, mask_id=156895, device=device)

    from vllm import distributed
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = random.randint(50000, 60000).__str__()
    distributed.init_distributed_environment(1, 0, 'env://', 0, 'nccl')
    distributed.initialize_model_parallel(1, backend='nccl')
    print("[Loading model]")
    # setup EP
    parallel_config = ParallelConfig(enable_expert_parallel = True)
    with set_current_vllm_config(VllmConfig(parallel_config = parallel_config)):
        model_config = AutoConfig.from_pretrained(moe_model_path, trust_remote_code=True)
        model = LLaDAMoeModelLM(config=model_config).eval()
        model.load_weights(moe_model_path, torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(moe_model_path, trust_remote_code=True)
        model = model.to(device)

        # # Test diffusion iteration.
        # check_iteration(model, decoder, input_ids)

        # Test generation without cache.
        print('Test block-wise diffusion MOE-LLM without KV-cache')
        dllm = BlockWiseDiffusionLLM(model, decoder, BlockIteratorFactory(), early_stop=True)
        res = dllm.generate(input_ids, gen_length=128, block_length=32)
        res1, nfe = generate(model, input_ids, gen_length=128, block_length=32, threshold=0.9, mask_id=156895, eos_id=156892)
        res2, nfe = generate_merge(model, input_ids, None, gen_length=128, block_length=32, threshold=0.9, mask_id=156895, eos_id=156892, parallel_decoding='threshold', early_stop=False,)
        res1 = res1[res1 != 156892]
        res2 = res2[res2 != 156892]
        assert res.shape[1] == len(res1)
        assert res.shape[1] == len(res2)
        res1 = res1.to(res.device)
        res2 = res2.to(res.device)
        assert torch.all(res == res1)
        assert torch.all(res == res2)

        # Test generation without cache with batch size == 2.
        #print('Test block-wise diffusion MOE-LLM without KV-cache and batch size == 2')
        #input_ids2 = get_prompts(tokenizer, mask_id=156895, device=device, num=2)
        #res11 = dllm.generate(input_ids2[0].unsqueeze(0), gen_length=128, block_length=32)
        #res12 = dllm.generate(input_ids2[1].unsqueeze(0), gen_length=128, block_length=32)
        #res2 = dllm.generate(input_ids2, gen_length=128, block_length=32)
        #assert res2.shape[0] == 2
        #res21 = res2[0]
        #res22 = res2[1]
        #res21 = res21[res21 != 156892]
        #res22 = res22[res22 != 156892]
        #assert res11.shape[1] == len(res21)
        #assert res12.shape[1] == len(res22)
        # assert torch.all(res11[0] == res21)
        # assert torch.all(res12[0] == res22)

        # Test generation with iteration smooth without kv-cache.
        print('Test block-wise diffusion MOE-LLM with iteration smooth without kv-cache')
        dllm = IterSmoothDiffusionLLM(model, decoder, BlockIteratorFactory(), early_stop=True)
        dllm1 = IterSmoothDiffusionLLM_test(model, decoder, BlockIteratorFactory(), early_stop=True)
        res = dllm.generate(input_ids, gen_length=128, block_length=32)
        res1 = dllm1.generate(input_ids, gen_length=128, block_length=32)
        assert dllm.num_forwards == dllm1.num_forwards
        assert dllm.cache_updates == 0
        assert res.shape[1] == res1.shape[1]
        res1 = res1.to(res.device)
        assert torch.all(res == res1)

        # Test generation with dual cache
        print('Test block-wise diffusion MOE-LLM with dual KV-cache')
        dllm = BlockWiseDiffusionLLM(model, decoder, BlockIteratorFactory(), early_stop=True, cache_factory=KVCacheFactory('dual'))
        res = dllm.generate(input_ids, gen_length=256, block_length=32)
        res1, nfe = generate_with_dual_cache(model, input_ids, gen_length=256, block_length=32, threshold=0.9, mask_id=156895, eos_id=156892)
        res1 = res1[res1 != 156892]
        assert res.shape[1] == len(res1)
        res1 = res1.to(res.device)
        assert torch.all(res == res1)

        # Test generation with dual cache with batch size == 2
        #print('Test block-wise diffusion MOE-LLM with dual KV-cache and batch size == 2')
        #dllm = BlockWiseDiffusionLLM(model, decoder, BlockIteratorFactory(), early_stop=True, cache_factory=KVCacheFactory('dual'))
        #input_ids2 = get_prompts(tokenizer, mask_id=156895, device=device, num=2)
        #res2 = dllm.generate(input_ids2, gen_length=256, block_length=32)
        #assert res2.shape[0] == 2
        #res21 = res2[0]
        #res22 = res2[1]
        #res21 = res21[res21 != 156892]
        #res22 = res22[res22 != 156892]
        #assert res1.shape[0] == len(res21)
        #assert res1.shape[0] == len(res22)
        # assert torch.all(res1 == res21)
        # assert torch.all(res1 == res22)

        # Test generation with iteration smooth with kv-cache.
        print('Test block-wise diffusion MOE-LLM with iteration smooth with kv-cache')
        dllm = IterSmoothDiffusionLLM(model, decoder, BlockIteratorFactory(), early_stop=True, cache_factory=KVCacheFactory('dual'))
        dllm1 = IterSmoothDiffusionLLM_test(model, decoder, BlockIteratorFactory(), early_stop=True, cache_factory=KVCacheFactory('dual'))
        res = dllm.generate(input_ids, gen_length=128, block_length=32)
        res1 = dllm1.generate(input_ids, gen_length=128, block_length=32)
        assert dllm.num_forwards == dllm1.num_forwards
        assert dllm.cache_updates > 0
        assert dllm.cache_updates == dllm1.cache_updates
        assert res.shape[1] == res1.shape[1]
        res1 = res1.to(res.device)
        assert torch.all(res == res1)

        # Test generation with iteration smooth and vicinity cache update.
        print('Test block-wise diffusion MOE-LLM with iteration smooth with vicinity cache update')
        dllm = IterSmoothWithVicinityCacheDiffusionLLM(model, decoder, BlockIteratorFactory(), early_stop=True, cache_factory=KVCacheFactory('dual'))
        dllm1 = IterSmoothWithVicinityCacheDiffusionLLM_test(model, decoder, BlockIteratorFactory(), early_stop=True, cache_factory=KVCacheFactory('dual'))
        res = dllm.generate(input_ids, gen_length=128, block_length=32)
        res1 = dllm1.generate(input_ids, gen_length=128, block_length=32)
        assert dllm.num_forwards == dllm1.num_forwards
        assert dllm.cache_updates > 0
        assert dllm.cache_updates == dllm1.cache_updates
        assert res.shape[1] == res1.shape[1]
        res1 = res1.to(res.device)
        assert torch.all(res == res1)

        # Test generation without cache.
        print('Test block-wise hierarchical diffusion MOE-LLM without KV-cache')
        dllm = BlockWiseDiffusionLLM(model, h_decoder, BlockIteratorFactory(), early_stop=True)
        res = dllm.generate(input_ids, gen_length=128, block_length=32)
        res1, nfe = generate_hierarchy(model, input_ids, gen_length=128, block_length=32, threshold=0.9, mask_id=156895, eos_id=156892,decoding='hierarchy_fast_v2',
                                        low_threshold=0.4, remask_threshold=0.4)
        res1 = res1[res1 != 156892]
        assert res.shape[1] == len(res1)
        res1 = res1.to(res.device)
        assert torch.all(res == res1)

    distributed.destroy_model_parallel()
    distributed.destroy_distributed_environment()
    

def test_diffusion():
    torch.cuda.set_device(0)
    device = torch.device(0)
    config = AutoConfig.from_pretrained(model_path)
    config.flash_attention = True
    model = LLaDAModelLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, config=config).eval()
    model = model.to(device)
    fastdllm_model = LLaDAModelLM_fastdllm.from_pretrained(model_path, torch_dtype=torch.bfloat16, config=config).eval()
    fastdllm_model = fastdllm_model.to(device)
    decoder = ThresholdParallelDecoder(0, threshold=0.9, use_float64=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    input_ids = get_prompts(tokenizer, mask_id=126336, device=device)
    batch_size = 1
    input_ids = torch.tensor(input_ids).to(device).repeat(batch_size, 1)

    print('Test sliding-window diffusion LLM with dual KV-cache')
    dllm = VicinityCacheDiffusionLLM(model, decoder, SimulateBlockIteratorFactory(), KVCacheFactory('dual'))
    res = dllm.generate(input_ids, gen_length=128, block_length=32)
    res1, nfe = generate_with_dual_cache(fastdllm_model, input_ids, gen_length=128, block_length=32, threshold=0.9)
    res1 = res1[res1 != 126081]
    assert res.shape[1] == len(res1)
    res1 = res1.to(res.device)
    assert torch.all(res == res1)

    # Test generation without cache.
    print('Test block-wise diffusion LLM without KV-cache')
    dllm = BlockWiseDiffusionLLM(model, decoder, BlockIteratorFactory(), early_stop=True)
    res = dllm.generate(input_ids, gen_length=128, block_length=32)
    res1, nfe = generate(fastdllm_model, input_ids, gen_length=128, block_length=32, threshold=0.9)
    res1 = res1[res1 != 126081]
    assert res.shape[1] == len(res1)
    res1 = res1.to(res.device)
    assert torch.all(res == res1)

    # Test generation with prefix cache
    print('Test block-wise diffusion LLM with prefix KV-cache')
    dllm = BlockWiseDiffusionLLM(model, decoder, BlockIteratorFactory(), early_stop=True, cache_factory=KVCacheFactory('prefix'))
    res = dllm.generate(input_ids, gen_length=128, block_length=32)
    res1, nfe = generate_with_prefix_cache(fastdllm_model, input_ids, gen_length=128, block_length=32, threshold=0.9)
    res1 = res1[res1 != 126081]
    assert res.shape[1] == len(res1)
    res1 = res1.to(res.device)
    assert torch.all(res == res1)

    # Test generation with dual cache
    print('Test block-wise diffusion LLM with dual KV-cache')
    dllm = BlockWiseDiffusionLLM(model, decoder, BlockIteratorFactory(), cache_factory=KVCacheFactory('dual'), early_stop=True)
    res = dllm.generate(input_ids, gen_length=128, block_length=32)
    res1, nfe = generate_with_dual_cache(fastdllm_model, input_ids, gen_length=128, block_length=32, threshold=0.9)
    res1 = res1[res1 != 126081]
    assert res.shape[1] == len(res1)
    res1 = res1.to(res.device)
    assert torch.all(res == res1)

def setup_distributed(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = random.randint(30000, 40000).__str__()
    print(f'rank={rank}, world size={world_size}')
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

def check_worker(rank, world_size, gpu):
    setup_distributed(rank, world_size)
    torch.cuda.set_device(gpu)
    device = torch.device(gpu)
    full_data = torch.arange(100).view(4, 25).to(device)

    # Partition size is smaller than block size.
    block_size = 6
    part_size = 4
    first_part_start = 1
    last_part_end = first_part_start + part_size * world_size
    assert last_part_end <= full_data.shape[1]
    partial_start = first_part_start + part_size * rank
    partial_end = partial_start + part_size
    part_data = full_data[:, partial_start:partial_end]
    # The accessed block must be covered by all parts.
    for block_start in range(first_part_start, last_part_end - block_size):
        block_end = block_start + block_size
        block_data = gather_sequence_block(part_data, partial_start, partial_end, block_start, block_end, rank, world_size)
        assert torch.all(block_data == full_data[:, block_start:block_end])

    # Partition size is larger than block size.
    block_size = 4
    part_size = 6
    first_part_start = 1
    last_part_end = first_part_start + part_size * world_size
    assert last_part_end <= full_data.shape[1]
    partial_start = first_part_start + part_size * rank
    partial_end = partial_start + part_size
    part_data = full_data[:, partial_start:partial_end]
    # The accessed block must be covered by all parts.
    for block_start in range(first_part_start, last_part_end - block_size):
        block_end = block_start + block_size
        block_data = gather_sequence_block(part_data, partial_start, partial_end, block_start, block_end, rank, world_size)
        assert torch.all(block_data == full_data[:, block_start:block_end])

    # Partition size is equal to block size.
    block_size = 4
    part_size = 4
    first_part_start = 1
    last_part_end = first_part_start + part_size * world_size
    assert last_part_end <= full_data.shape[1]
    partial_start = first_part_start + part_size * rank
    partial_end = partial_start + part_size
    part_data = full_data[:, partial_start:partial_end]
    # The accessed block must be covered by all parts.
    for block_start in range(first_part_start, last_part_end - block_size):
        block_end = block_start + block_size
        block_data = gather_sequence_block(part_data, partial_start, partial_end, block_start, block_end, rank, world_size)
        assert torch.all(block_data == full_data[:, block_start:block_end])

    dist.destroy_process_group()

def test_dist():
    num_gpus = 4
    procs = []
    for i, gpu in enumerate(range(num_gpus)):
        p = Process(target=check_worker, args=(i, num_gpus, i))
        procs.append(p)
        p.start()
    for p in procs:
        p.join()

def check_diffusion_worker(rank, world_size, gpu):
    setup_distributed(rank, world_size)
    torch.cuda.set_device(gpu)
    device = torch.device(gpu)

    config = AutoConfig.from_pretrained(model_path)
    config.flash_attention = True
    model = LLaDAModelLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, config=config).eval()
    model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    input_ids = get_prompts(tokenizer, mask_id=126336, device=device)
    batch_size = 1
    input_ids = torch.tensor(input_ids).to(device).repeat(batch_size, 1)

    # Test generation without cache.
    print('Test diffusion LLM without KV-cache')
    decoder = ThresholdParallelDecoder(0, threshold=0.9, use_float64=True)
    dllm = BlockWiseDiffusionLLMWithSP(rank, world_size, model, decoder, BlockIteratorFactory())
    res = dllm.generate(input_ids, gen_length=128, block_length=32)
    res1, nfe = generate_sp(model, input_ids, rank=rank, world_size=world_size, gen_length=128, block_length=32, threshold=0.9)
    res1 = res1[res1 != 126081]
    assert res.shape[1] == len(res1)
    res1 = res1.to(res.device)
    assert torch.all(res == res1)

    dist.destroy_process_group()

def test_diffusion_sp():
    num_gpus = 4
    procs = []
    for i, gpu in enumerate(range(num_gpus)):
        p = Process(target=check_diffusion_worker, args=(i, num_gpus, i))
        procs.append(p)
        p.start()
    for p in procs:
        p.join()

@pytest.mark.skip(reason="Produces errors during certain runs")
def test_moe_server(require_init=True):
    print('test serving of diffusion-MOE')
    params = SamplingParams(temperature=0, threshold=0.9, mask_id=156895, eos_id=156892, early_stop=True, cache='', cont_weight=0, enable_torch_compile=True)

    device = torch.device(0)
    tokenizer = AutoTokenizer.from_pretrained(moe_model_path, trust_remote_code=True)
    input_ids = get_prompts(tokenizer, mask_id=156895, device=device)
    input_ids = torch.tensor(input_ids)

    from vllm import distributed
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = random.randint(30000, 40000).__str__()
    distributed.init_distributed_environment(1, 0, 'env://', 0, 'nccl')
    distributed.initialize_model_parallel(1, backend='nccl')

    batch_size = 1
    decoder = ThresholdParallelDecoder(0, threshold=0.9, mask_id=156895, eos_id=156892)
    # setup EP
    parallel_config = ParallelConfig(enable_expert_parallel = True)
    with set_current_vllm_config(VllmConfig(parallel_config = parallel_config)):
        model_config = AutoConfig.from_pretrained(moe_model_path, trust_remote_code=True)
        model = LLaDAMoeModelLM(config=model_config).eval()
        model.load_weights(moe_model_path, torch_dtype=torch.bfloat16)
        model.forward = torch.compile(model.forward, mode='reduce-overhead', fullgraph=False, dynamic=True)
        tokenizer = AutoTokenizer.from_pretrained(moe_model_path, trust_remote_code=True)
        model = model.to(device)

        dllm = BlockWiseDiffusionLLM(model, decoder, BlockIteratorFactory(), early_stop=True)
        res1 = dllm.generate(input_ids, gen_length=256, block_length=32).cpu()

    # Test DP == 1 and TPEP == 1
    print('Test serving: DP == 1 and TPEP == 1')
    llm = DiffusionLLMServing(model=moe_model_path, is_moe=True, sample_params=params, num_gpus=1, server_port=random.randint(50000, 60000))
    res = llm.generate(input_ids, gen_length=256, block_length=32)
    assert res.shape == res1.shape
    res1 = res1.to(res.device)
    assert torch.all(res == res1)
    llm.stop_serving()

    input_ids2 = torch.cat([input_ids, input_ids])
    # Test DP == 2 and TPEP == 1
    print('Test serving: DP == 2 and TPEP == 1')
    llm = DiffusionLLMServing(model=moe_model_path, is_moe=True, sample_params=params, num_gpus=2, dp_size=2, tpep_size=1, server_port=random.randint(50000, 60000))
    res2 = llm.generate(input_ids2, gen_length=256, block_length=32)

    # Remove EOS and padding tokens before comparison
    assert torch.all(res2[0][res2[0] != 156892] == res[0][res[0] != 156892])
    assert torch.all(res2[1][res2[1] != 156892] == res[0][res[0] != 156892])
    llm.stop_serving()

    # Test DP == 2 and TPEP == 2
    print('Test serving: DP == 2 and TPEP == 2')
    llm = DiffusionLLMServing(model=moe_model_path, is_moe=True, sample_params=params, num_gpus=2, dp_size=1, tpep_size=2, server_port=random.randint(50000, 60000))
    res = llm.generate(input_ids, gen_length=256, block_length=32)
    llm.stop_serving()

    input_ids2 = torch.cat([input_ids, input_ids])
    llm = DiffusionLLMServing(model=moe_model_path, is_moe=True, sample_params=params, num_gpus=4, dp_size=2, tpep_size=2, server_port=random.randint(40000, 50000))
    res2 = llm.generate(input_ids2, gen_length=256, block_length=32)

    assert torch.all(res2[0][res2[0] != 156892] == res[0][res[0] != 156892])
    llm.stop_serving()

    distributed.destroy_model_parallel()
    distributed.destroy_distributed_environment()

@pytest.mark.skip(reason="Produces errors during certain runs")
def test_server():
    print('test serving of diffusion')
    params = SamplingParams(temperature=0, threshold=0.9, mask_id=126336, eos_id=126081, early_stop=True, cache='', cont_weight=0, enable_torch_compile=True)

    torch.cuda.set_device(0)
    device = torch.device(0)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    input_ids = get_prompts(tokenizer, mask_id=126336, device=device)
    input_ids = torch.tensor(input_ids)
    config = AutoConfig.from_pretrained(model_path)
    config.flash_attention = True
    model = LLaDAModelLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, config=config).eval()
    model.forward = torch.compile(model.forward, mode='reduce-overhead', fullgraph=False, dynamic=True)
    model = model.to(device)
    # Test generation without cache.
    decoder = ThresholdParallelDecoder(0, threshold=0.9)
    dllm = BlockWiseDiffusionLLM(model, decoder, BlockIteratorFactory(), early_stop=True)
    res1 = dllm.generate(input_ids, gen_length=256, block_length=32).cpu()

    # Test DP == 1 and TPEP == 1
    print('Test serving: DP == 1 and TPEP == 1')
    llm = DiffusionLLMServing(model=model_path, is_moe=False, sample_params=params, num_gpus=1, server_port=random.randint(40000, 50000))
    res = llm.generate(input_ids, gen_length=256, block_length=32)
    llm.stop_serving()
    assert res.shape == res1.shape
    res1 = res1.to(res.device)
    assert torch.all(res == res1)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    logging.basicConfig(level=logging.INFO)

    test_token_array()
    test_block_iterator()

    test_moe_diffusion()
    test_diffusion()
    test_server()
    test_moe_server(False)

    test_dist()
    test_diffusion_sp()
