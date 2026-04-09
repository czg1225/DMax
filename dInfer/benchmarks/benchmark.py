import torch
import numpy as np
import torch.nn.functional as F
import os
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch.distributed as dist
import time
import tqdm
from vllm.config import CompilationConfig, ParallelConfig
from vllm.config import VllmConfig, set_current_vllm_config, get_current_vllm_config
from vllm.forward_context import set_forward_context
import json

from dinfer.model import LLaDAMoeModelLM, LLaDAModelLM, LLaDA2MoeModelLM
from dinfer import BlockIteratorFactory, KVCacheFactory
from dinfer import ThresholdParallelDecoder,CreditThresholdParallelDecoder, HierarchyDecoder, BlockWiseDiffusionLLM, IterSmoothDiffusionLLM, VicinityCacheDiffusionLLM, IterSmoothWithVicinityCacheDiffusionLLM, BlockDiffusionLLM

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def setup_distributed(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12345'
    print(f'rank={rank}, world size={world_size}')
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def load_inputs(dataset, tokenizer):
    with open(dataset, 'r') as f:
        data = json.load(f)
    prompts = []
    questions = []
    ids = []
    all_input_ids = []
    if "judge_details" in data.keys():
        details_data = data['judge_details']
    else:
        details_data = data['details']
    for id, judge_detail in enumerate(details_data):
        ids.append(id)
        prompt = judge_detail['prompt']
        prompts.append(prompt)
        questions.append(prompt)
        prompt = '<role>SYSTEM</role>detailed thinking off<|role_end|><role>HUMAN</role>'+prompt+'<|role_end|><role>ASSISTANT</role>'   

        input_ids = tokenizer(prompt)['input_ids']
        input_ids = torch.tensor(input_ids).unsqueeze(0)
        all_input_ids.append(input_ids)
    return all_input_ids, prompts, questions, ids



@ torch.no_grad()
def main(world_size, rank, gpu_id, args):
    print('started', world_size, rank, gpu_id, args)
    torch.cuda.set_device(gpu_id)
    device = torch.device(gpu_id)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    block_length=args.block_length
    gen_length = args.gen_len

    from vllm import distributed
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(45601+args.port_offset)
    distributed.init_distributed_environment(world_size, rank, 'env://', rank, 'nccl')
    distributed.initialize_model_parallel(args.tp_size, backend='nccl')
    print("[Loading model]")
    # setup EP
    parallel_config = ParallelConfig(enable_expert_parallel = True)
    with set_current_vllm_config(VllmConfig(parallel_config = parallel_config)):
        vllm_config = get_current_vllm_config()
        print("EP Enabled:", vllm_config.parallel_config.enable_expert_parallel)

        model_config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
        if args.model_type=='llada_moe':
            model = LLaDAMoeModelLM(config=model_config).eval()
            model.load_weights(args.model_name, torch_dtype=torch.bfloat16)
            mask_id = 156895
            eos_id = 156892
        elif args.model_type=='llada2':
            model = LLaDA2MoeModelLM(config=model_config).eval()
            model.load_weights(args.model_name, torch_dtype=torch.bfloat16, device=device)
            mask_id = 156895
            eos_id = 156892
        elif args.model_type=='llada':
            model = LLaDAModelLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, init_device=str(device)).eval()
            model.init_h2e_module()
            mask_id = 126336
            eos_id = 126081
        else:
            raise ValueError('model type not supported')
        
        if args.tp_size>1 and args.use_tp:
            print('enabling tp')
            model.tensor_parallel(args.tp_size)
        model = model.to(device)
        model.forward = torch.compile(model.forward, mode='reduce-overhead', fullgraph=False, dynamic=True)

        if args.parallel_decoding == 'threshold':
            if args.use_credit:
                decoder = CreditThresholdParallelDecoder(temperature=0, threshold=args.threshold, mask_id=mask_id, eos_id=eos_id)
            else:
                decoder = ThresholdParallelDecoder(temperature=0, threshold=args.threshold, mask_id=mask_id, eos_id=eos_id)

        else:
            decoder = HierarchyDecoder(temperature=0, threshold=args.threshold, low_threshold=args.low_threshold, mask_id=mask_id, eos_id=eos_id)
        use_sw = args.prefix_look > 0 or args.after_look > 0 or args.warmup_times > 0
            
        if args.cache == 'prefix' or args.cache == 'dual':
            cache_factory=KVCacheFactory(args.cache, is_bd_model=args.use_bd)
        else:
            cache_factory=None

        if not args.use_bd and args.cont_weight>0 and use_sw:
            dllm = IterSmoothWithVicinityCacheDiffusionLLM(model, decoder, BlockIteratorFactory(start_block_align=True), cache_factory=cache_factory, early_stop=True,
                cont_weight=args.cont_weight, prefix_look=args.prefix_look, after_look=args.after_look, warmup_steps=args.warmup_times)
        elif not args.use_bd and args.cont_weight>0 and not use_sw:
            dllm = IterSmoothDiffusionLLM(model, decoder, BlockIteratorFactory(start_block_align=True), cache_factory=cache_factory, early_stop=True, cont_weight=args.cont_weight)
        elif not args.use_bd and args.cont_weight == 0 and use_sw:
            dllm = VicinityCacheDiffusionLLM(model, decoder, BlockIteratorFactory(start_block_align=True), cache_factory=cache_factory, early_stop=True,
                prefix_look=args.prefix_look, after_look=args.after_look, warmup_steps=args.warmup_times)
        elif not args.use_bd and args.cont_weight == 0 and not use_sw:
            dllm = BlockWiseDiffusionLLM(model, decoder, BlockIteratorFactory(start_block_align=True), cache_factory=cache_factory, early_stop=True, use_shift=args.use_shift)
        else:
            dllm = BlockDiffusionLLM(model, decoder, BlockIteratorFactory(start_block_align=True, use_block_diffusion=True), cache_factory=cache_factory, early_stop=True)


        prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she can run 6 kilometers per hour. How many kilometers can she run in 8 hours?"
        prompt = '<role>SYSTEM</role>detailed thinking off<|role_end|><role>HUMAN</role>'+prompt+'<|role_end|><role>ASSISTANT</role>'
        input_ids = tokenizer(prompt)['input_ids']
        input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
        # cudagraph warmup
        for i in range(2):
            dllm.generate(input_ids, gen_length=gen_length, block_length=block_length)


        prev_forwards = dllm.num_forwards
        inner_start = time.time()
        out = dllm.generate(input_ids, gen_length=gen_length, block_length=block_length)
        inner_stop = time.time()
        sample_time = inner_stop - inner_start
        nfe = dllm.num_forwards - prev_forwards

        token_number = int((out!=156892).sum() - input_ids.shape[1])
        tpf = token_number/nfe
        tps = token_number/sample_time
        fps = nfe/sample_time
        if rank == 0:
            print(f'nfe={nfe:4d}, token number={token_number:4d}, fps={fps:4.2f},tpf={tpf:2.2f}, tps={tps:4.2f}')
            print(f'generated text: {tokenizer.decode(out[0], skip_special_tokens=True)}')
         

def process_args(args):
    import warnings
    gpus = [int(gpu) for gpu in args.gpu.split(',')]
    if len(gpus) > 1 and not args.use_tp:
        warnings.warn('Using multiple GPUs without tensor parallelism is not recommended. TP will be enabled.')
    elif len(gpus) == 1 and args.use_tp:
        warnings.warn('Using tensor parallelism with only one GPU is not accepted. TP will be disabled.')
    
    if args.model_type == 'llada2' and not args.use_bd:
        warnings.warn('Using llada2 without block diffusion is not recommended.')
    
    if args.model_type == 'llada2' and args.cache == '':
        warnings.warn('Using llada2 without kvcache is not recommended. cache will be set to prefix.')
        args.cache = 'prefix'

    args.tp_size = len(gpus)
    args.use_tp = args.tp_size > 1
    args.port_offset = gpus[0]

    return args

from multiprocessing import Process
import argparse

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--gpu', type=str, default='0,1,2,3')
    parser.add_argument('--gen_len', type=int, default=1024)
    parser.add_argument('--prefix_look', type=int, default=0)
    parser.add_argument('--after_look', type=int, default=0)
    parser.add_argument('--block_length', type=int, default=64)
    parser.add_argument('--threshold', type=float, default=0.9)
    parser.add_argument('--warmup_times', type=int, default=0)
    parser.add_argument('--low_threshold', type=float, default=0.3)
    parser.add_argument('--cont_weight', type=float, default=0)
    parser.add_argument('--parallel_decoding', type=str, default='threshold')
    parser.add_argument('--use_credit', action='store_true')
    parser.add_argument('--exp_name', type=str, default='exp')
    parser.add_argument('--cache', type=str, default='')
    parser.add_argument('--use_tp', action='store_true')
    parser.add_argument('--use_shift', action='store_true')
    parser.add_argument('--use_bd', action='store_true')
    parser.add_argument('--model_type', type=str, default='llada2',
        help="llada2 (for llada2-mini or llada2-flash) | llada_moe (for llada-moe) | llada (for llada or llada-1.5)")
    args = parser.parse_args()
        

    print(f"The input args are listed as follows: {args}")

    args = process_args(args)
    gpus = [int(gpu) for gpu in args.gpu.split(',')]
    procs = []
    if len(gpus) == 1:
        main(1, 0, gpus[0], args)
    else:
        for i, gpu in enumerate(gpus):
            p = Process(target=main, args=(len(gpus), i, gpu, args))
            p.daemon = True
            procs.append(p)
            p.start()
        for p in procs:
            p.join()
