'''
This file is inspired by the code from https://github.com/NVlabs/Fast-dLLM
'''
import accelerate
import torch
import random
import torch.nn.functional as F
from datasets import Dataset
from tqdm import tqdm, trange
import accelerate
import random
import numpy as np
import json
import time
import datasets
import json
import time
import datasets
import os
from transformers import AutoTokenizer, AutoConfig
import torch.multiprocessing as mp
from multiprocessing import Process
from lm_eval.api.model import LM
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from dinfer.model import LLaDAMoeModelLM, LLaDAModelLM, LLaDA2MoeModelLM
from dinfer import BlockIteratorFactory, KVCacheFactory
from dinfer import ThresholdParallelDecoder,CreditThresholdParallelDecoder, HierarchyDecoder, BlockWiseDiffusionLLM, IterSmoothDiffusionLLM, VicinityCacheDiffusionLLM, IterSmoothWithVicinityCacheDiffusionLLM, BlockDiffusionLLM
from vllm import distributed
from vllm.config import VllmConfig, set_current_vllm_config, get_current_vllm_config
from vllm.config import ParallelConfig
from dataclasses import dataclass

datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True
datasets.config.DOWNLOAD_TIMEOUT = 180 
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

bucket_size = 32
used_buckets = []
def warmup_cudagraph(rank, device, dllm, gen_len, block_length, batch_size, vocab_size):
    if rank==0:
        print('warmup')
        print(used_buckets)
        iterator = tqdm(used_buckets)
    else:
        iterator = used_buckets
    offset = 0
    for i in iterator:   
        input_ids = torch.randint(0, vocab_size, (batch_size, i - gen_len+offset), dtype=torch.long, device=device)
        dllm.generate(input_ids, gen_length=gen_len, block_length=block_length)

def cut_eos(data, eos_id=156892):
    eos_indices = (data[0] == eos_id).nonzero(as_tuple=True)[0]
    if eos_indices.numel() > 0:
        first_eos_idx = eos_indices[0].item()
        return data[:, :first_eos_idx]
    else:
        return data


@ torch.no_grad()
def run_benchmark(world_size, rank, gpu_id, tokenizer, args):
    print('started', world_size, rank, gpu_id, args)
    torch.cuda.set_device(gpu_id)
    device = torch.device(gpu_id)

    all_input_ids, padded_gen_lens = args.all_input_ids, args.padded_gen_lens

    block_length=args.block_length
    # print()
    from vllm import distributed
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(args.master_port+args.port_offset)
    distributed.init_distributed_environment(world_size, rank, 'env://', rank, 'nccl')
    distributed.initialize_model_parallel(args.tp_size, backend='nccl')
    print("[Loading model]")
    # setup EP
    parallel_config = ParallelConfig(enable_expert_parallel = True)
    with set_current_vllm_config(VllmConfig(parallel_config = parallel_config)):
        vllm_config = get_current_vllm_config()
        print("EP Enabled:", vllm_config.parallel_config.enable_expert_parallel)

        model_config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
        if 'llada_moe' == args.model_type:
            model = LLaDAMoeModelLM(config=model_config).eval()
            model.load_weights(args.model_name, torch_dtype=torch.bfloat16)
            print('llada_moe')
        elif 'llada2' == args.model_type:
            model = LLaDA2MoeModelLM(config=model_config).eval()
            model.load_weights(args.model_name, torch_dtype=torch.bfloat16)
        elif 'llada' == args.model_type:
            model = LLaDAModelLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, init_device=device).eval()
        else:
            raise ValueError('model type not supported')

        if args.tp_size>1 and args.use_tp:
            print('enabling tp')
            model.tensor_parallel(args.tp_size)

        x = torch.arange(50+args.gen_len, dtype=torch.long, device=device).unsqueeze(0)
        model = model.to(device)
        out = model(x, use_cache=False)
        out = model(x, use_cache=True)

        if args.use_compile:
            if args.use_cudagraph:
                model.forward = torch.compile(model.forward, mode='reduce-overhead', fullgraph=False, dynamic=True)
            else:
                model.forward = torch.compile(model.forward, fullgraph=False, dynamic=True)


        if args.parallel_decoding == 'threshold':
            if args.use_credit:
                decoder = CreditThresholdParallelDecoder(temperature=0, threshold=args.threshold, mask_id=args.mask_id, eos_id=args.eos_id)
            else:
                decoder = ThresholdParallelDecoder(temperature=0, threshold=args.threshold, mask_id=args.mask_id, eos_id=args.eos_id)

        else:
            decoder = HierarchyDecoder(temperature=0, threshold=args.threshold, low_threshold=args.low_threshold, mask_id=args.mask_id, eos_id=args.eos_id)

        use_sw = args.prefix_look > 0 or args.after_look > 0 or args.warmup_times > 0

        if args.cache == 'prefix' or args.cache == 'dual':
            cache_factory=KVCacheFactory(args.cache, is_bd_model=args.use_bd)
        else:
            cache_factory=None

        if not args.use_bd:
            if args.cont_weight>0:
                if use_sw:
                    print("IterSmoothWithVicinityCacheDiffusionLLM")
                    dllm = IterSmoothWithVicinityCacheDiffusionLLM(model, decoder, BlockIteratorFactory(start_block_align=True), cache_factory=cache_factory, early_stop=True,
                        cont_weight=args.cont_weight, prefix_look=args.prefix_look, after_look=args.after_look, warmup_steps=args.warmup_times)
                else:
                    print("IterSmoothDiffusionLLM")
                    dllm = IterSmoothDiffusionLLM(model, decoder, BlockIteratorFactory(start_block_align=True), cache_factory=cache_factory, early_stop=True, cont_weight=args.cont_weight)
            else:
                if use_sw:
                    print("VicinityCacheDiffusionLLM")
                    dllm = VicinityCacheDiffusionLLM(model, decoder, BlockIteratorFactory(start_block_align=True), cache_factory=cache_factory, early_stop=True,prefix_look=args.prefix_look, after_look=args.after_look, warmup_steps=args.warmup_times)
                else:
                    print("BlockWiseDiffusionLLM")
                    dllm = BlockWiseDiffusionLLM(model, decoder, BlockIteratorFactory(start_block_align=True), cache_factory=cache_factory, early_stop=True, use_shift=args.use_shift)
        else:
            print("BlockDiffusionLLM")
            dllm = BlockDiffusionLLM(model, decoder, BlockIteratorFactory(start_block_align=True), cache_factory=cache_factory, early_stop=True)

        warmup_cudagraph(rank, device, dllm, args.gen_len, block_length, args.batch_size, args.vocab_size)
                
        for wi in range(1):
            outputs = []
            total_forward = 0
            if rank==0:
                iterator = trange(0, len(all_input_ids), args.batch_size)
            else:
                iterator = range(0, len(all_input_ids), args.batch_size)
            start = time.time()
            tpfs = []
            tpss = []
            fpss = []
            total_token = 0
            token_numbers = []
            for i in iterator:   
                input_ids = all_input_ids[i:i+args.batch_size]
                max_length = 0
                min_padded_length = 10000
                for j, seq in enumerate(input_ids):
                    if seq.shape[1] > max_length:
                        max_length = seq.shape[1]
                        min_padded_length = padded_gen_lens[i+j]
                batch_input_ids= torch.zeros((len(input_ids), max_length), dtype=torch.long, device=device).fill_(args.mask_id)
                for j in range(len(input_ids)):
                    batch_input_ids[j, :input_ids[j].shape[1]] = input_ids[j].to(device)
                input_ids = batch_input_ids
                padded_gen_len = padded_gen_lens[i]
                inner_start = time.time()
                prev_forwards = dllm.num_forwards
                out = dllm.generate(input_ids, gen_length=min_padded_length, block_length=block_length)
                nfe = dllm.num_forwards - prev_forwards
                inner_stop = time.time()
                sample_time = inner_stop - inner_start
                for j in range(input_ids.shape[0]):
                    outputs.append(out[j].unsqueeze(0))
                total_forward += nfe
                batch_token_number = 0
                for j in range(input_ids.shape[0]):
                    token_number = int((out[j]!=156892).sum() - all_input_ids[i+j].shape[1])
                    batch_token_number += token_number
                    token_numbers.append(token_number)
                tpf = batch_token_number/nfe/args.batch_size
                tps = batch_token_number/sample_time
                fps = nfe/sample_time
                if rank == 0:
                    print(f'[iter {i:4d}]nfe={nfe:4d}, token number={batch_token_number:4d}, fps={fps:4.2f},tpf={tpf:2.2f}, tps={tps:4.2f}')
                    if wi==0 and i<5:
                        for j in range(input_ids.shape[0]):
                            answer = cut_eos(out[j, all_input_ids[i+j].shape[1]:].unsqueeze(0))[0]
                            # print(answer)
                            print(f'generated text {j}: {tokenizer.decode(answer, skip_special_tokens=False)}')
                tpfs.append(tpf)
                tpss.append(tps)
                fpss.append(fps)
                total_token += token_number

        total_token = total_token

        stop = time.time()
        answers = []
        if rank==0:
            for i in trange(len(outputs)):
                out = outputs[i]
                answer = (tokenizer.decode(out[0, all_input_ids[i].shape[1]:], skip_special_tokens=True))
                answers.append(answer)
            print(f'Forward: {total_forward}, Time: {stop-start}, FPS: {total_forward/(stop-start)}({np.mean(fpss)}), TPS: {total_token/(stop-start)}({np.mean(tpss)}), TPF: {total_token/total_forward}({np.mean(tpfs)})')
            filename = args.save_path
            with open (filename, 'w') as f:
                for i in range(len(answers)):
                    answer = answers[i]
                    json.dump({'answer': answer, 'generated_length': token_numbers[i], 'tpf':tpfs[i//args.batch_size], 'tps':tpss[i//args.batch_size], 'fps':fpss[i//args.batch_size], }, f)
                    f.write('\n')
            print(f'Forward: {total_forward}, Time: {stop-start}, FPS: {total_forward/(stop-start)}({np.mean(fpss)}), TPS: {total_token/(stop-start)}({np.mean(tpss)}), TPF: {total_token/total_forward}({np.mean(tpfs)})')

            with open(args.speed_path, 'a+') as f:
                print( args.config, args.parallel_decoding, args.threshold, args.prefix_look, args.batch_size, args.block_length, total_forward, stop-start, total_token / len(all_input_ids), total_forward/(stop-start), total_token/(stop-start), total_token/total_forward, sum(padded_gen_lens)/total_forward, np.mean(fpss), np.mean(tpss), np.mean(tpfs), file=f)
        return 


@dataclass
class EvalConfig:
    model_name: str = ''
    gpu: str = '0,1,2,3'
    batch_size: int = 1
    gen_len: int = 1024
    prefix_look: int = 0
    after_look: int = 0
    block_length: int = 64
    threshold: float = 0.9
    warmup_times: int = 0
    low_threshold: float = 0.3
    cont_weight: float = 0
    parallel_decoding: str = 'threshold'
    use_credit: bool = False
    cache: str = ''
    use_tp: bool = False
    save_path: str = ''
    config: int = 0
    tp_size: int = 1
    port_offset: int = 0
    all_input_ids = None
    padded_gen_lens = None
    use_cudagraph: bool = False
    use_compile: bool = True
    use_bd: bool = False
    use_shift: bool = False
    model_type: str = 'llada'
    vocab_size: int = 156896
    master_port: int = 23456
    mask_id: int = 156895
    eos_id: int = 156892
    save_dir: str = './res'
    save_samples: bool = False
    speed_path: str = ''


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@register_model("dInfer_eval")
class DInferEvalHarness(LM):
    def __init__(
        self,
        model_path='',
        device="cuda",
        mask_id=126336,
        eos_id=126081,
        max_length=4096,
        batch_size=2,
        mc_num=128,
        is_check_greedy=True,
        gen_length=1024,
        block_length=1024,
        save_dir=None,
        show_speed=False,
        parallel_decoding="threshold",
        threshold: float=0.9,
        cache: str="",
        warmup_times: int=0,
        low_threshold: float=0.3,
        cont_weight: float=0,
        use_credit: bool=False,
        tp_size: int=1,
        parallel = 'dp',
        use_compile = True,
        master_port = 23456,
        use_cudagraph = True,
        gpus = '0,1,2,3',
        use_bd = False,
        prefix_look = 0,
        after_look = 0,
        use_shift = False,
        model_type = 'llada',
        save_samples = False,
        **kwargs
    ):

        super().__init__()
        
        self.model_path = model_path
        self.mask_id = mask_id
        self.eos_id = eos_id
        self.mc_num = mc_num
        self.batch_size = int(batch_size)
        assert mc_num % self.batch_size == 0
        self.sampling_eps = 0.
        self.max_length = max_length
        self.is_check_greedy = is_check_greedy
        self.gen_length = gen_length
        self.block_length = block_length
        self.save_dir = save_dir
        self.show_speed = show_speed
        self.parallel_decoding = parallel_decoding
        self.threshold = threshold
        self.cache = cache
        self.warmup_times = warmup_times
        self.low_threshold = low_threshold
        self.cont_weight = cont_weight
        self.use_credit = use_credit
        self.master_port = master_port
        self.tp_size = tp_size
        self.use_compile = use_compile
        self.parallel = parallel
        self.use_cudagraph = use_cudagraph
        self.gpus = gpus
        self.prefix_look = prefix_look
        self.after_look = after_look
        self.use_bd = use_bd
        self.kwargs = kwargs
        self.use_shift = use_shift
        self.model_type = model_type
        self.save_samples = save_samples

        if self.model_type == 'llada_moe':
            self.mask_id = 156895
            self.eos_id = 156892
            self.vocab_size = 156896
            self.is_moe = True
        elif self.model_type == 'llada2': 
            self.mask_id = 156895
            self.eos_id = 156892
            self.vocab_size = 156896
            self.is_moe = True
        elif self.model_type == 'llada':
            self.vocab_size = 126464
            self.is_moe = False
        else:
            raise ValueError('model type not supported')

        accelerator = accelerate.Accelerator()
        if accelerator.num_processes > 1:
            self.accelerator = accelerate.Accelerator()
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.accelerator = None
        
        model_kwargs = {}
        if self.accelerator is not None:
            model_kwargs.update({'device_map': {'': f'{self.accelerator.device}'}})  
        
        # set decoder
        if parallel_decoding == "threshold":
          if use_credit:
              decoder = CreditThresholdParallelDecoder(temperature=0, threshold=threshold, mask_id=self.mask_id, eos_id=self.eos_id)
          else:
              decoder = ThresholdParallelDecoder(temperature=0, threshold=threshold, mask_id=self.mask_id, eos_id=self.eos_id)
        else:
            decoder = HierarchyDecoder(temperature=0, threshold=threshold, low_threshold=low_threshold,
                                      mask_id=self.mask_id, eos_id=self.eos_id)
        if parallel == 'dp':
            self.device= torch.device(device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = str(self.master_port + self.rank)
            distributed.init_distributed_environment(1, 0, 'env://', 0, 'nccl')
            distributed.initialize_model_parallel(1, backend='nccl')
            parallel_config = ParallelConfig(enable_expert_parallel = True)
            with set_current_vllm_config(VllmConfig(parallel_config = parallel_config)):
                vllm_config = get_current_vllm_config()
                print("EP Enabled:", vllm_config.parallel_config.enable_expert_parallel)
                config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                # load model
                if self.model_type == 'llada_moe':
                    self.model = LLaDAMoeModelLM(config=config).eval()
                    self.model.load_weights(self.model_path, torch_dtype=torch.bfloat16)
                elif self.model_type == 'llada2':
                    self.model = LLaDA2MoeModelLM(config=config).eval()
                    self.model.load_weights(self.model_path, torch_dtype=torch.bfloat16)
                elif self.model_type == 'llada':
                    self.model = LLaDAModelLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, init_device=str(device)).eval()
                else:
                    raise ValueError('model type not supported')
                self.vllm_config = vllm_config

            
            if self.accelerator is not None:
                self.model = self.accelerator.prepare(self.model)
                self.device = torch.device(f'{self.accelerator.device}')
            else:
                self.model = self.model.to(self.device)

            if self.use_compile:
                # compile model
                if self.use_cudagraph:
                    self.model.forward = torch.compile(self.model.forward, fullgraph=False, dynamic=True, mode='reduce-overhead')
                else:
                    self.model.forward = torch.compile(self.model.forward, fullgraph=False, dynamic=True)
            
            if self.cache == 'prefix' or self.cache == 'dual':
                cache_factory=KVCacheFactory(self.cache, is_bd_model=self.use_bd)
            else:
                cache_factory=None
            use_sw = self.cache != '' and (self.prefix_look > 0 or self.after_look > 0 or self.warmup_times > 0 )
            if not self.use_bd:
                if self.cont_weight>0:
                    if use_sw:
                        self.dllm = IterSmoothWithVicinityCacheDiffusionLLM(self.model, decoder, BlockIteratorFactory(start_block_align=True), cache_factory=cache_factory, early_stop=True,
                            cont_weight=self.cont_weight, prefix_look=self.prefix_look, after_look=self.after_look, warmup_steps=self.warmup_times)
                    else:
                        self.dllm = IterSmoothDiffusionLLM(self.model, decoder, BlockIteratorFactory(start_block_align=True), cache_factory=cache_factory, early_stop=True, cont_weight=self.cont_weight)
                else:
                    if use_sw:
                        self.dllm = VicinityCacheDiffusionLLM(self.model, decoder, BlockIteratorFactory(start_block_align=True), cache_factory=cache_factory, early_stop=True,
                            prefix_look=self.prefix_look, after_look=self.after_look, warmup_steps=self.warmup_times)
                    else:
                        self.dllm = BlockWiseDiffusionLLM(self.model, decoder, BlockIteratorFactory(start_block_align=True), cache_factory=cache_factory, early_stop=True, use_shift=self.use_shift)
            else:
                self.dllm = BlockDiffusionLLM(self.model, decoder, BlockIteratorFactory(start_block_align=True), cache_factory=cache_factory, early_stop=True)
            
        elif parallel == 'tp':
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        else:
            raise NotImplementedError(parallel)
        
        

    @property
    def rank(self):
        return self._rank
    
    @property
    def world_size(self):
        return self._world_size
    
    @property
    def tokenizer_name(self) -> str:
        return self.model_path
    
    def apply_chat_template(self, chat_history, **kwargs) -> str:
        if "tokenize" not in kwargs:
            kwargs["tokenize"] = False
        return self.tokenizer.apply_chat_template(chat_history, **kwargs)

    def _forward_process(self, batch, prompt_index):
        b, l = batch.shape

        target_len = (l - prompt_index.sum()).item()
        k = torch.randint(1, target_len + 1, (), device=batch.device)

        x = torch.round(torch.linspace(float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device)).long()
        x = ((x - 1) % target_len) + 1
        assert x.min() >= 1 and x.max() <= target_len

        indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
        is_mask = indices < x.unsqueeze(1)

        for i in range(b):
            is_mask[i] = is_mask[i][torch.randperm(target_len)]

        is_mask = torch.cat((torch.zeros(b, prompt_index.sum(), dtype=torch.bool, device=batch.device), is_mask), dim=1)

        noisy_batch = torch.where(is_mask, self.mask_id, batch)

        return noisy_batch, (x / target_len).unsqueeze(1).repeat(1, l)

    @torch.no_grad()
    def get_logits(self, batch, prompt_index):
        if self.cfg > 0.:
            assert len(prompt_index) == batch.shape[1]
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index] = self.mask_id
            batch = torch.cat([batch, un_batch])

        logits = self.model(batch).logits

        if self.cfg > 0.:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (self.cfg + 1) * (logits - un_logits)
        return logits[:, :batch.shape[1]]

    @torch.no_grad()
    def get_loglikelihood(self, prefix, target):
        seq = torch.concatenate([prefix, target])[None, :]
        seq = seq.repeat((self.batch_size, 1)).to(self.device)

        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)

        loss_acc = []
        for _ in range(self.mc_num // self.batch_size):
            perturbed_seq, p_mask = self._forward_process(seq, prompt_index)

            mask_indices = perturbed_seq == self.mask_id

            logits = self.get_logits(perturbed_seq, prompt_index)

            loss = F.cross_entropy(logits[mask_indices], seq[mask_indices], reduction='none') / p_mask[mask_indices]
            loss = loss.sum() / self.batch_size
            loss_acc.append(loss.item())

        return - sum(loss_acc) / len(loss_acc)

    @torch.no_grad()
    def suffix_greedy_prediction(self, prefix, target):
        if not self.is_check_greedy:
            return False

        seq = torch.full((1, len(prefix) + len(target)), self.mask_id, device=self.device)
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        prefix, target = prefix.to(self.device), target.to(self.device)
        seq[0, :len(prefix)] = prefix

        for i in range(len(target)):
            mask_index = (seq == self.mask_id)
            logits = self.get_logits(seq, prompt_index)[mask_index]
            x0 = torch.argmax(logits, dim=-1)

            p = torch.softmax(logits.to(torch.float32), dim=-1)
            confidence = torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(dim=-1)
            _, index = torch.sort(confidence, descending=True)
            x0[index[1:]] = self.mask_id
            seq[mask_index] = x0.clone()
        correct = target == seq[0, len(prefix):]
        correct = torch.all(correct)
        return correct

    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tokenizer(context + continuation)["input_ids"]
        context_enc = self.tokenizer(context)["input_ids"]

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        return context_enc, continuation_enc

    def loglikelihood(self, requests):
        def _tokenize(e):
            prefix, target = self._encode_pair(e["prefix"], e["target"])
            return {
                "prefix_text": e["prefix"],
                "target_text": e["target"],
                "prefix": prefix,
                "target": target,
            }

        ds = []
        ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")
        prompt_len = [len(x["prefix"]) + len(x["target"]) for x in ds]

        assert max(prompt_len) <= 4096

        out = []
        with torch.no_grad():
            for elem in tqdm(ds, desc="Computing likelihood..."):
                prefix = elem["prefix"]
                target = elem["target"]

                ll = self.get_loglikelihood(prefix, target)

                is_target_greedy_dec = self.suffix_greedy_prediction(prefix, target)

                out.append((ll, 1.0 if is_target_greedy_dec else 0.0))
        torch.cuda.empty_cache()
        return out

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError
    
    
    def generate_until(self, requests):
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
            self.save_path = os.path.join(self.save_dir, f'rank_{self.rank}.jsonl')
            print(f"save_path: {self.save_path}")
            self.speed_path = os.path.join(self.save_dir, f'results.txt')
        

        def get_bucket_length(length):
            bucket_length = bucket_size*(length//bucket_size)
            if bucket_length not in used_buckets:
                used_buckets.append(bucket_length)
            return bucket_length

        def load_inputs(prompts, tokenizer):
            all_input_ids = []
            for id, prompt in enumerate(prompts):
                input_ids = tokenizer(prompt.args[0])['input_ids']
                input_ids = torch.tensor(input_ids).unsqueeze(0)
                all_input_ids.append(input_ids)
            return all_input_ids

        def cal_bucket_len(gen_len, all_input_ids):
            max_prompt_length = 0
            padded_gen_lens = []

            for i in range(len(all_input_ids)):
                input_ids = all_input_ids[i]
                if input_ids.shape[1] > max_prompt_length:
                    max_prompt_length = input_ids.shape[1]
                padded_length = get_bucket_length(input_ids.shape[1]+gen_len)
                padded_gen_lens.append(padded_length - input_ids.shape[1])
            return padded_gen_lens


        all_input_ids = load_inputs(requests, self.tokenizer)
        padded_gen_lens = cal_bucket_len(self.gen_length, all_input_ids)
        
        answers = []
        outputs = []
        total_forward = 0
        start = time.time()
        tpfs = []
        tpss = []
        fpss = []
        total_token = 0
        token_numbers = []
    
        if self.parallel == 'dp':
            with set_current_vllm_config(self.vllm_config):
                    if self.use_cudagraph and self.use_cudagraph:
                        warmup_cudagraph(self.rank, self.device, self.dllm, self.gen_length, self.block_length, self.batch_size, self.vocab_size)
                    for i, req in enumerate(tqdm(requests, desc="Generating...")):
                        input_ids = all_input_ids[i]
                        padded_gen_len = padded_gen_lens[i]
                        inner_start = time.time()
                        input_ids = input_ids.to(self.device)
                        prev_forwards = self.dllm.num_forwards
                        out = self.dllm.generate(input_ids, gen_length=padded_gen_len,block_length=self.block_length)
                        nfe = self.dllm.num_forwards - prev_forwards
                        inner_stop = time.time()
                        sample_time = inner_stop - inner_start
                        outputs.append(out)
                        answer = (self.tokenizer.decode(out[0, all_input_ids[i].shape[1]:], skip_special_tokens=True))
                        answers.append(answer)
                        total_forward += nfe
                        token_number = out.shape[1] - input_ids.shape[1]
                        token_numbers.append(token_number)
                        tpf = token_number/nfe
                        tps = token_number/sample_time
                        fps = nfe/sample_time
                        if self.rank == 0:
                            print(f'iter={i}, fps={fps}, nfe={nfe}')
                        tpfs.append(tpf)
                        tpss.append(tps)
                        fpss.append(fps)
                        total_token += token_number
            total_token = total_token

            stop = time.time()
            print(f'Forward: {total_forward}, Time: {stop-start}, FPS: {total_forward/(stop-start)}({np.mean(fpss)}), TPS: {total_token/(stop-start)}({np.mean(tpss)}), TPF: {total_token/total_forward}({np.mean(tpfs)})')

            if self.show_speed and self.save_dir is not None:
                with open (self.save_dir+f'/rank{self.rank}_results.jsonl', 'w', encoding='utf-8') as file:
                    data={'rank':f'rank{self.rank}',
                        'forward per second': np.mean(fpss),
                        'tokens per second': np.mean(tpss),
                        'tokens per forward': np.mean(tpfs),
                        'average generated length': total_token / len(all_input_ids)
                        }
                    file.write(json.dumps(data, ensure_ascii=False) + '\n')
        elif self.parallel == 'tp':
            procs = []
            answers = []
            gpus = [int(gpu) for gpu in self.gpus.split(';')]
            args = {"gpu": gpus, "batch_size": self.batch_size, "model_name": self.model_path, "gen_len": self.gen_length, "block_length": self.block_length, "prefix_look": self.prefix_look, "after_look": self.after_look, "warmup_times": self.warmup_times, "low_threshold": self.low_threshold, "threshold": self.threshold, "cont_weight": self.cont_weight, "use_credit": self.use_credit, "cache": self.cache, "parallel_decoding": self.parallel_decoding, "tp_size": self.tp_size, "save_path": self.save_path, "use_cudagraph": self.use_cudagraph, "use_compile": self.use_compile,"use_bd": self.use_bd, "use_shift": self.use_shift, "model_type": self.model_type, "vocab_size": self.vocab_size, "mask_id": self.mask_id, "eos_id": self.eos_id, "save_dir": self.save_dir, "save_samples": self.save_samples, "speed_path": self.speed_path}
            args = EvalConfig(**args)
            args.tp_size = len(gpus)
            args.master_port = self.master_port
            args.use_tp = args.tp_size > 1
            args.port_offset = gpus[0]
            args.all_input_ids = all_input_ids
            args.padded_gen_lens = padded_gen_lens
            
            if len(gpus) == 1:
                run_benchmark(1, 0, gpus[0], self.tokenizer, args)
            else:
                for i, gpu in enumerate(gpus):
                    ctx = mp.get_context('spawn')
                    p = ctx.Process(target=run_benchmark, args=(len(gpus), i, gpu, self.tokenizer, args))
                    p.daemon = True
                    procs.append(p)
                    p.start()
                for p in procs:
                    p.join()
            answers = []
            with open(self.save_path, 'r') as f:
                for line in f :
                    answers.append(json.loads(line)["answer"])
            if not self.save_samples:
                os.remove(self.save_path)
        return answers


if __name__ == "__main__":
    set_seed(1234)
    cli_evaluate()