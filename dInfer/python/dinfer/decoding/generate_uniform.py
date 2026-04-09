import torch
import numpy as np
import logging

from transformers.models.layoutlmv2.modeling_layoutlmv2 import relative_position_bucket

from .utils import TokenArray, DistAlignedTokenArray, gather_sequence_block
from .utils import calculate_op_num, BlockLoc

logger = logging.getLogger(__name__)


def _contiguous(x: torch.Tensor):
    return x if x.is_contiguous() else x.contiguous()


def _device_key(device):
    dev = torch.device(device)
    return dev.type, dev.index


def align_exp2(x: torch.Tensor):
    assert x.ndim == 0 and x.item() >= 0
    shift = 0 if x == 0 else int(torch.floor(torch.log2(x.to(torch.float64)))) + 1
    return 1 << shift

class DiffusionLLM:
    """ Diffusion LLM inference
    """

    @ torch.no_grad()
    def generate(self, prompt, gen_length=128, block_length=128):
        ''' Generate tokens with diffusion iterations.

        Parameters:
        ----------
        prompt: Torch.Tensor
            A tensor of shape (1, L) that contains the input prompt.
        gen_length: int
            Generated answer length.
        block_length: int
            Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.

        Returns
        -------
        Torch.Tensor: A tensor of shape (1, L') that contains the prompt tokens and the generated tokens.
            EOS and any tokens after EOS have been removed.
        '''

def select_undecoded(seq_idx, orig_x, x, block, block_loc, mask_id, writeback=False):
    if x.batch_size == 1:
        return seq_idx, x
    bool_idx = torch.all(block != mask_id, dim=1)

    if writeback:
        # Write the decoded tokens back
        finished_idx = seq_idx[bool_idx]
        orig_x[finished_idx, block_loc.start:block_loc.end] = block[bool_idx]

    # Select the undecoded sequences
    return seq_idx, x

class BlockRunner:
    """ The class decodes all tokens in a block

    Parameters
    ----------
    diff_iteration : DiffusionIteration
        Run forward computation on a block to decode tokens
    early_stop : bool
        Whether or not to have early stop
    maximum_unroll : int
        The max number of iterations to unroll
    expected_tpf : int
        The expected TPF for loop unrolling.
    """
    def __init__(self, diff_iteration, early_stop, maximum_unroll, expected_tpf):
        self.diff_iteration = diff_iteration
        self.early_stop = early_stop
        self.maximum_unroll = maximum_unroll
        self.expected_tpf = expected_tpf

    def decode(self, model, decoder, x, kv_cache, block, block_loc, block_id):
        """ Decode all tokens in a block.

        Parameters
        ----------
        model : pytorch model
            The diffusion LLM
        decoder : ParallelDecoder
            The decoder
        x : TokenArray
            The input tokens. The decoded tokens are also stored in this array.
        kv_cache: KVCache
            The KV-cache
        block : torch.Tensor
            The input tokens in the block.
        block_loc : BlockLoc
            The start and the end of the location of the decoding block.
        block_id : int
            The block ID

        Returns
        -------
        torch.Tensor : a bool tensor that indicates whether the sequences have finished decoding.
        """
        orig_x = x
        seq_idx = torch.arange(x.batch_size, device=block.device)
        seq_idx, x = select_undecoded(seq_idx, orig_x, x, block, block_loc, decoder.mask_id, writeback=False)
        block = x[:, block_loc.start:block_loc.end]
        batch_size = x.batch_size
        while (block == decoder.mask_id).sum() > 0:
            unroll_k = int(max(min((block == decoder.mask_id).sum()//self.expected_tpf, self.maximum_unroll), 1))
            for unroll_i in range(unroll_k):
                self.diff_iteration.forward(model, decoder, x, kv_cache, block, block_loc, block_id)

            # If there are more than one sequence, we should filter the sequences and only decode
            # on the sequences that still have masked tokens.
            if batch_size > 1:
                seq_idx, x = select_undecoded(seq_idx, orig_x, x, block, block_loc, decoder.mask_id, writeback=True)
                block = x[:, block_loc.start:block_loc.end]
                # If all blocks have been decoded, we can jumpt out.
                if len(seq_idx) == 0:
                    break
            batch_size = x.batch_size

        eos_idx = torch.any(orig_x[:, block_loc.start:block_loc.end] == decoder.eos_id, dim=1)
        if self.early_stop:
            # Find the first location of EOS and set all tokens after the location to EOS.
            # Here we assume that don't perform remasking.
            orig_x[eos_idx, block_loc.end:] = decoder.eos_id
        return eos_idx

class BlockDiffusionRunner(BlockRunner):
    """ The class decodes all tokens in a block

    Parameters
    ----------
    diff_iteration : BlockDiffusionIteration
        Run forward computation on a block to decode tokens
    early_stop : bool
        Whether or not to have early stop
    maximum_unroll : int
        The max number of iterations to unroll
    expected_tpf : int
        The expected TPF for loop unrolling.
    """
    def __init__(self, diff_iteration, early_stop, maximum_unroll, expected_tpf, backend):
        super().__init__(diff_iteration, early_stop, maximum_unroll, expected_tpf)
        self.backend = backend
        self.cache_update_count = 0
        self.hidden_cache_update_count = 0
        self.need_cross_block_update = False
        self._embedding_layer = None

    def prefill(self, model, prefilling_x, kv_cache, pos_ids, attn_mask, prefilling_limit, block_length):
        """ Prefill for KV Cache
        Parameters
        ----------
        model : pytorch model
            The diffusion LLM
        prefilling_x : torch.Tensor
            The input IDs of the tokens in the prefilling range.
        kv_cache: KVCache
            The KV-cache
        pos_ids: torch.Tensor
            The position IDs of the tokens in the prefilling range.
        attn_mask: torch.Tensor
            The attention mask of the tokens in the prefilling range.
        prefilling_limit: int
            The limit of the first prefilling step.
        block_length: int
            The block length, used for the following prefilling steps if needed.
        """
        if kv_cache is None:
            return
        else:
            if prefilling_limit > prefilling_x.shape[1]:
                output = model(
                    _contiguous(prefilling_x),
                    use_cache=True,
                    attention_mask=attn_mask,
                    position_ids=_contiguous(pos_ids),
                )
                if self.backend == 'vllm':
                    kv_cache.update(output.past_key_values)
                else:
                    kv_cache.range_update(output.past_key_values, 0, prefilling_x.size(1), 0)
            else:
                # limit prefilling length to avoid OOM
                # first prefill partial prompt
                output = model(
                    _contiguous(prefilling_x[:, :prefilling_limit]),
                    use_cache=True,
                    attention_mask=attn_mask[:, :prefilling_limit, :prefilling_limit],
                    position_ids=_contiguous(pos_ids[:, :prefilling_limit]),
                )
                if self.backend == 'vllm':
                    kv_cache.update(output.past_key_values)
                else:
                    kv_cache.range_update(output.past_key_values, 0, prefilling_limit, 0)
                # continue prefilling other parts, using block length to be able to replay already captured cuda graph
                for block_start in range(prefilling_limit, prefilling_x.shape[1], block_length):
                    block_end = block_start+block_length
                    kv_cache.extend_cache(block_end)
                    past_key_values, replace_position = kv_cache.get_key_values(block_start, block_end)
                    output = model(
                        _contiguous(prefilling_x[:, block_start:block_end]),
                        use_cache=True,
                        past_key_values=past_key_values,
                        position_ids=_contiguous(pos_ids[:, block_start:block_end]),
                    )
                    if self.backend == 'vllm':
                        kv_cache.update(output.past_key_values)
                    else:
                        kv_cache.range_update(output.past_key_values, 0, block_end, block_length)     

            self.diff_iteration.num_forwards +=1
            self.diff_iteration.iter_no +=1
        self.need_cross_block_update = False

    def decode(self, model, decoder, x, kv_cache, block, block_loc, block_id, pos_ids, attn_mask, block_length=32, cross_block_attn_mask=None):
        """ Decode all tokens in a block.

        Parameters
        ----------
        model : pytorch model
            The diffusion LLM
        decoder : ParallelDecoder
            The decoder
        x : TokenArray
            The input tokens. The decoded tokens are also stored in this array.
        kv_cache: KVCache
            The KV-cache
        block : torch.Tensor
            The input IDs of the tokens in the current decoding block.
        block_loc : BlockLoc
            The start and the end of the location of the decoding block.
        block_id : int
            The block ID
        pos_ids: torch.Tensor
            The position IDs of all the tokens.
        attn_mask: torch.Tensor
            The attention mask of all the tokens. 
        Returns
        -------
        torch.Tensor : a bool tensor that indicates whether the sequences have finished decoding.
        """
        orig_x = x
        seq_idx = torch.arange(x.batch_size, device=block.device)
        seq_idx, x = select_undecoded(seq_idx, orig_x, x, block, block_loc, decoder.mask_id, writeback=False)
        block = x[:, block_loc.start:block_loc.end]
        batch_size = x.batch_size

        if kv_cache is not None:
            kv_cache.extend_cache(block_loc.end)
            past_key_values, replace_position = kv_cache.get_key_values(block_loc.start, block_loc.end)
        else:
            past_key_values, replace_position = None, None

        input_block_mask_number = 0
        output = None
        while (block == decoder.mask_id).sum() > 0:
            unroll_k = int(max(min((block == decoder.mask_id).sum()//self.expected_tpf, self.maximum_unroll), 1))
            for unroll_i in range(unroll_k):
                input_block_mask_number = (block == decoder.mask_id).sum()
                if self.need_cross_block_update:
                    cross_block_loc = BlockLoc(block_loc.start-block_length, block_loc.end)
                    cross_block_x = x[:, block_loc.start-block_length:block_loc.end]
                    cross_block_replace_positions = (block_loc.start-block_length, block_loc.end)
                    output = self.diff_iteration.forward(model, decoder, x, kv_cache, cross_block_x, cross_block_loc, 
                                block_id, pos_ids, cross_block_attn_mask, past_key_values, cross_block_replace_positions, 
                                self.backend, is_cross_block=True, block_length=block_length)
                    if self.backend=='vllm':
                        if isinstance(output.past_key_values, list):
                            kv_cache.update([past_key_value[:, :, :block_loc.start] for past_key_value in output.past_key_values])
                        else:
                            kv_cache.update(output.past_key_values._data[:, :, :, :, :block_loc.start])
                        kv_cache.extend_cache(block_loc.end)
                    else:
                        last_block_past_key_values = [past_key_value[:, :, :block_loc.start-block_loc.end] for past_key_value in output.past_key_values]
                        kv_cache.range_update(last_block_past_key_values, 0, block_loc.start, block_length)
                    past_key_values, replace_position = kv_cache.get_key_values(block_loc.start, block_loc.end)
                    self.need_cross_block_update = False
                else:
                    output = self.diff_iteration.forward(model, decoder, x, kv_cache, block, block_loc, block_id, pos_ids, attn_mask, past_key_values, replace_position, self.backend)
            if batch_size > 1:
                seq_idx, x = select_undecoded(seq_idx, orig_x, x, block, block_loc, decoder.mask_id, writeback=True)
                block = x[:, block_loc.start:block_loc.end]
                # If all blocks have been decoded, we can jumpt out.
                if len(seq_idx) == 0:
                    break

        if output==None:
            # force update if output is None
            output = self.diff_iteration.forward(model, decoder, x, kv_cache, block, block_loc, block_id, pos_ids, attn_mask, past_key_values, replace_position, self.backend)

        if kv_cache is not None:
            self.cache_update_count += 1
            if input_block_mask_number > 0:
                self.need_cross_block_update = True
            else:
                self.need_cross_block_update = False
                self.hidden_cache_update_count += 1
                if self.backend=='vllm':
                    kv_cache.update(output.past_key_values)
                else:
                    kv_cache.range_update(output.past_key_values, 0, block_loc.end, block_loc.end - block_loc.start)


        eos_idx = torch.any(orig_x[:, block_loc.start:block_loc.end] == decoder.eos_id, dim=1)
        if self.early_stop:
            orig_x[eos_idx, block_loc.end:] = decoder.eos_id

        return eos_idx
    
    

    def decode_uniform(self, model, decoder, x, kv_cache, block, block_loc, block_id, pos_ids, attn_mask, block_length=32, cross_block_attn_mask=None):
        """ Decode all tokens in a block.

        Parameters
        ----------
        model : pytorch model
            The diffusion LLM
        decoder : ParallelDecoder
            The decoder
        x : TokenArray
            The input tokens. The decoded tokens are also stored in this array.
        kv_cache: KVCache
            The KV-cache
        block : torch.Tensor
            The input IDs of the tokens in the current decoding block.
        block_loc : BlockLoc
            The start and the end of the location of the decoding block.
        block_id : int
            The block ID
        pos_ids: torch.Tensor
            The position IDs of all the tokens.
        attn_mask: torch.Tensor
            The attention mask of all the tokens. 
        Returns
        -------
        torch.Tensor : a bool tensor that indicates whether the sequences have finished decoding.
        """
        orig_x = x
        seq_idx = torch.arange(x.batch_size, device=block.device)
        seq_idx, x = select_undecoded(seq_idx, orig_x, x, block, block_loc, decoder.mask_id, writeback=False)
        block = x[:, block_loc.start:block_loc.end]
        batch_size = x.batch_size

        if kv_cache is not None:
            kv_cache.extend_cache(block_loc.end)
            past_key_values, replace_position = kv_cache.get_key_values(block_loc.start, block_loc.end)
        else:
            past_key_values, replace_position = None, None


        step=0
        Breakflag = False
        embeddings = None
        if self._embedding_layer is None:
            self._embedding_layer = model.model.model.word_embeddings
        embedding_layer = self._embedding_layer
        active_index = (x[:, block_loc.start:block_loc.end] ==  decoder.mask_id)

        while step<block_length:
            step+=1

            if self.need_cross_block_update:
                cross_block_loc = BlockLoc(block_loc.start-block_length, block_loc.end)
                cross_block_x = x[:, block_loc.start-block_length:block_loc.end]
                cross_block_replace_positions = (block_loc.start-block_length, block_loc.end) 
                output, Breakflag, embeddings = self.diff_iteration.forward_uniform(
                model, decoder, x, kv_cache, cross_block_x, cross_block_loc, block_id, 
                pos_ids, cross_block_attn_mask, past_key_values, cross_block_replace_positions, self.backend, is_cross_block=True, active_index=active_index, embeddings=embeddings, embedding_layer=embedding_layer
            )   

                if self.backend=='vllm':
                    if isinstance(output.past_key_values, list):
                        kv_cache.update([past_key_value[:, :, :block_loc.start] for past_key_value in output.past_key_values])
                    else:
                        kv_cache.update(output.past_key_values._data[:, :, :, :, :block_loc.start])
                    kv_cache.extend_cache(block_loc.end)
                else:
                    last_block_past_key_values = [past_key_value[:, :, :block_loc.start-block_loc.end] for past_key_value in output.past_key_values]
                    kv_cache.range_update(last_block_past_key_values, 0, block_loc.start, block_length)
                past_key_values, replace_position = kv_cache.get_key_values(block_loc.start, block_loc.end)
                self.need_cross_block_update = False
            else:
                # 执行 Forward (这一步会更新 x，从而更新 block 的内容)
                output, Breakflag, embeddings = self.diff_iteration.forward_uniform(
                    model, decoder, x, kv_cache, block, block_loc, block_id, 
                    pos_ids, attn_mask, past_key_values, replace_position, self.backend, active_index=active_index, embeddings=embeddings, embedding_layer=embedding_layer
                )

            if Breakflag:
                break

        self.need_cross_block_update = True

        eos_idx = torch.any(orig_x[:, block_loc.start:block_loc.end] == decoder.eos_id, dim=1)
        if self.early_stop:
            orig_x[eos_idx, block_loc.end:] = decoder.eos_id

        return eos_idx

class DiffusionIteration:
    """ A diffusion iteration to decode tokens
    """
    def __init__(self):
        self.num_forwards = 0
        self.cache_updates = 0

    def forward(self, model, x, kv_cache, block, block_loc, block_id):
        """ The forward computation to decode tokens.
        """
        pass

class BaseDiffusionIteration(DiffusionIteration):
    """ A base implementation of diffusion iteration to decode.
    """
    def __init__(self):
        super().__init__()
        self.iter_no = 0

    def forward(self, model, decoder, x, kv_cache, block, block_loc, block_id):
        """ Decode tokens in a forward run on a block.

        The forward run decodes tokens in the input array.

        Parameters
        ----------
        model : pytorch model
            The diffusion LLM
        decoder : ParallelDecoder
            The decoder
        x : TokenArray
            The input tokens. The decoded tokens are also stored in this array.
        kv_cache: KVCache
            The KV-cache
        block : torch.Tensor
            The input IDs of the tokens in the current decoding block.
        block_loc : BlockLoc
            The start and the end of the location of the decoding block.
        block_id : int
            The block ID
        """
        cache_update_kv = None
        # Update KV-cache
        if kv_cache is not None and kv_cache.require_update(self.iter_no, block_loc.start, block_loc.end):
            output = model(x.data, use_cache=True)
            cache_update_kv = output.past_key_values
            self.num_forwards += 1
            # use the generated output to decode.
            decoder.decode(output.logits[:, block_loc.start:block_loc.end], block_loc.start, block_loc.end, x)
            # update KV-cache
            kv_cache.update(output.past_key_values)
            self.cache_updates += 1

        if kv_cache is None:
            logits = model(x.data).logits[:, block_loc.start:block_loc.end]
        elif kv_cache.cache_type == 'prefix':
            past_key_values, replace_position = kv_cache.get_key_values(block_loc.start, block_loc.end)
            logits = model(x[:, block_loc.start:], past_key_values=past_key_values, use_cache=True,
                    replace_position=replace_position).logits
            block_length = block_loc.end - block_loc.start
            logits = logits[:, :block_length]
        else:
            past_key_values, replace_position = kv_cache.get_key_values(block_loc.start, block_loc.end)
            # cache position is the position between current_block_start and current_block_end
            logits = model(block, past_key_values=past_key_values, use_cache=True,
                    replace_position=replace_position).logits

        decoder.decode(logits, block_loc.start, block_loc.end, x)
        self.num_forwards += 1
        self.iter_no += 1
        return cache_update_kv, logits

class BlockDiffusionIteration:
    """ An implementation of block diffusion iteration to decode.
    """
    def __init__(self):
        self.num_forwards = 0
        self.cache_updates = 0
        self.iter_no = 0

    def forward(self, model, decoder, x, kv_cache, block, block_loc, block_id, pos_ids, attn_mask, past_key_values, replace_position, backend, is_cross_block=False, block_length=32):
        """ Decode tokens in a forward run on a block.

        The forward run decodes tokens in the input array.

        Parameters
        ----------
        model : pytorch model
            The diffusion LLM
        decoder : ParallelDecoder
            The decoder
        x : TokenArray
            The input tokens. The decoded tokens are also stored in this array.
        kv_cache: KVCache
            The KV-cache
        block : torch.Tensor
            The input IDs of the tokens in the current decoding block.
        block_loc : BlockLoc
            The start and the end of the location of the decoding block.
        block_id : int
            The block ID
        pos_ids: torch.Tensor
            The position IDs of all the tokens.
        attn_mask: torch.Tensor
            The attention mask of all the tokens. 
        past_key_values: List[List[torch.Tensor]]
            The key-values required to decode the specified block.
        replace_position: torch.Tensor 
            The tensor indicates the valid locations in the returned key-values.
        """
        if kv_cache is None:
            output = model(x.data[:, :block_loc.end], 
                attention_mask=attn_mask[:,:block_loc.end,:block_loc.end],
                position_ids=pos_ids[:, :block_loc.end])
            logits = output.logits[:, block_loc.start:block_loc.end]
        else:
            block_input = _contiguous(block)
            pos_block = _contiguous(pos_ids[:, block_loc.start:block_loc.end])
            if is_cross_block:
                output = model(block_input,
                    position_ids=pos_block,
                    use_cache=True,
                    past_key_values=past_key_values,
                    attention_mask=attn_mask,
                    replace_position=(0,0) if backend=='sglang' else replace_position)
            else:
                output = model(block_input,
                    position_ids=pos_block,
                    use_cache=True,
                    past_key_values=past_key_values,
                    replace_position=(0,0) if backend=='sglang' else replace_position)
            logits = output.logits
            # TODO(dulun): we don't need update kv cache for every step.
            if backend == 'vllm':
                kv_cache.update(output.past_key_values)
        if is_cross_block:
            decoder.decode(logits[:, block_length:], block_loc.start+block_length, block_loc.end, x)
        else: 
            decoder.decode(logits, block_loc.start, block_loc.end, x)
        self.num_forwards += 1
        self.iter_no += 1
        return output
    

    def forward_uniform(self, model, decoder, x, kv_cache, block, block_loc, block_id, pos_ids, attn_mask, past_key_values, replace_position, backend, active_index, embeddings, embedding_layer, is_cross_block=False, block_length=32):
        """ Decode tokens in a forward run on a block.

        The forward run decodes tokens in the input array.

        Parameters
        ----------
        model : pytorch model
            The diffusion LLM
        decoder : ParallelDecoder
            The decoder
        x : TokenArray
            The input tokens. The decoded tokens are also stored in this array.
        kv_cache: KVCache
            The KV-cache
        block : torch.Tensor
            The input IDs of the tokens in the current decoding block.
        block_loc : BlockLoc
            The start and the end of the location of the decoding block.
        block_id : int
            The block ID
        pos_ids: torch.Tensor
            The position IDs of all the tokens.
        attn_mask: torch.Tensor
            The attention mask of all the tokens. 
        past_key_values: List[List[torch.Tensor]]
            The key-values required to decode the specified block.
        replace_position: torch.Tensor 
            The tensor indicates the valid locations in the returned key-values.
        """
        if kv_cache is None:
            full_input_ids = x.data[:, :block_loc.end] if embeddings is None else None
            output = model(
                full_input_ids,
                inputs_embeds=embeddings,
                attention_mask=attn_mask[:,:block_loc.end,:block_loc.end],
                position_ids=pos_ids[:, :block_loc.end],
            )

            logits = output.logits[:, block_loc.start:block_loc.end]
        else:
            block_input = _contiguous(block)
            pos_block = _contiguous(pos_ids[:, block_loc.start:block_loc.end])
            if is_cross_block:
                output = model(
                    block_input if embeddings is None else None,
                    inputs_embeds=embeddings,
                    position_ids=pos_block,
                    use_cache=True,
                    past_key_values=past_key_values,
                    attention_mask=attn_mask,
                    replace_position=(0,0) if backend=='sglang' else replace_position,
                )
            else:
                output = model(
                    block_input if embeddings is None else None,
                    inputs_embeds=embeddings,
                    position_ids=pos_block,
                    use_cache=True,
                    past_key_values=past_key_values,
                    replace_position=(0,0) if backend=='sglang' else replace_position,
                )


                    
            logits = output.logits
            # TODO(dulun): we don't need update kv cache for every step.
            if backend == 'vllm':
                kv_cache.update(output.past_key_values)
        if is_cross_block:
            Breakflag, embeddings = decoder.decode_uniform(
                logits[:, block_length:],
                block_loc.start+block_length,
                block_loc.end,
                x,
                active_index,
                embedding_layer,
                prev_embeddings=embeddings,
            )
        else: 
            Breakflag, embeddings = decoder.decode_uniform(
                logits,
                block_loc.start,
                block_loc.end,
                x,
                active_index,
                embedding_layer,
                prev_embeddings=embeddings,
            )

        self.num_forwards += 1
        self.iter_no += 1
        return output, Breakflag, embeddings


class ShiftDiffusionIteration(DiffusionIteration):
    """ A shift implementation of diffusion iteration to decode.
    """
    def __init__(self, use_shift = False):
        super().__init__()
        self.iter_no = 0

    def forward(self, model, decoder, x, kv_cache, block, block_loc, block_id):
        """ Decode tokens in a forward run on a block.

        The forward run decodes tokens in the input array.

        Parameters
        ----------
        model : pytorch model
            The diffusion LLM
        decoder : ParallelDecoder
            The decoder
        x : TokenArray
            The input tokens. The decoded tokens are also stored in this array.
        kv_cache: KVCache
            The KV-cache
        block : torch.Tensor
            The input IDs of the tokens in the current decoding block.
        block_loc : BlockLoc
            The start and the end of the location of the decoding block.
        block_id : int
            The block ID
        """
        block_start, block_end = block_loc.start-1, block_loc.end-1
        # Update KV-cache
        if kv_cache is not None and kv_cache.require_update(self.iter_no, block_start, block_end):
            output = model(x.data, use_cache=True)
            self.num_forwards += 1
            # use the generated output to decode.
            # TODO(dulun): need to improve efficiency
            x_shifted = TokenArray(x.data[:, 1:], 0, decoder.mask_id, decoder.eos_id, model.device)
            decoder.decode(output.logits[:, block_start:block_end], block_start, block_end, x_shifted)
            x.data[:, 1:] = x_shifted.data
            # update KV-cache
            kv_cache.update(output.past_key_values)
            self.cache_updates += 1

        if kv_cache is None:
            logits = model(x.data).logits[:, block_start:block_end]
        elif kv_cache.cache_type == 'prefix':
            past_key_values, replace_position = kv_cache.get_key_values(block_start, block_end)
            logits = model(x[:, block_start:], past_key_values=past_key_values, use_cache=True,
                    replace_position=replace_position).logits
            block_length = block_end - block_start
            logits = logits[:, :block_length]
        else:
            # cache position is the position between current_block_start and current_block_end
            past_key_values, replace_position = kv_cache.get_key_values(block_start, block_end)
            logits = model(x[:, block_start:block_end], past_key_values=past_key_values, use_cache=True,
                    replace_position=replace_position).logits
        # TODO(dulun): need to improve efficiency
        x_shifted = TokenArray(x.data[:, 1:], 0, decoder.mask_id, decoder.eos_id, model.device)
        decoder.decode(logits, block_start, block_end, x_shifted)
        x.data[:, 1:] = x_shifted.data
        self.num_forwards += 1
        self.iter_no += 1

class BlockWiseDiffusionLLM(DiffusionLLM):
    """ Diffusion LLM inference

    This diffusion LLM inference generates tokens block by block.

    The decoding algorithm break the generation sequence into blocks.
    It runs diffusion iterations on the first block and decodes all tokens
    in the block before moving to the next block.
    This is a classifical dLLM decoding algorithm.

    Parameters
    ----------
    model : Torch.Module
        The LLM model
    decoder : ParallelDecoder
        The decoder that decodes the tokens from the logits computed by the Transformer model
    iterator_facotry : IteratorFactory
        The factory class that generates the iterator on the input token array.
    cache_factory : KVCacheFactory (optional)
        The KV-cache factory that generates a kv-cache for LLM.
    """
    def __init__(self, model, decoder, iterator_factory, early_stop=True, cache_factory=None, maximum_unroll=4, expected_tpf=8, use_shift=False):
        self.model = model
        self.cache_factory = cache_factory
        self.decoder = decoder
        self.iterator_factory = iterator_factory
        if use_shift:
            self.diff_iteration = ShiftDiffusionIteration()
        else:
            self.diff_iteration = BaseDiffusionIteration()
        self.block_decoder = BlockRunner(self.diff_iteration, early_stop, maximum_unroll, expected_tpf)
        

    @property
    def num_forwards(self):
        return self.diff_iteration.num_forwards

    @property
    def cache_updates(self):
        return self.diff_iteration.cache_updates

    @ torch.no_grad()
    def generate(self, prompt, gen_length=128, block_length=128):
        ''' Generate tokens with diffusion iterations block by block.
        '''
        x = TokenArray(prompt, gen_length, self.decoder.mask_id, self.decoder.eos_id, self.model.device)
        it = self.iterator_factory.create(x, block_length)

        # We need to reset iter_no at the beginning of generating a sequence.
        self.diff_iteration.iter_no = 0
        kv_cache = self.cache_factory.create() if self.cache_factory is not None else None
        for block_id, (block_loc, block) in enumerate(it):
            self.decoder.block_init(block, block_id)
            decode_compl = self.block_decoder.decode(self.model, self.decoder, x, kv_cache, block, block_loc, block_id)
            # If all sequences have EOS, we have finished decoding.
            if torch.all(decode_compl):
                break
        logger.info(f'The number of diffusion iterations: {self.num_forwards}')
        return x.get_generated_tokens()

class IterationSmooth(DiffusionIteration):
    """ A diffusion iteration to decode tokens
    """
    def __init__(self, model, cont_weight=0.3, cont_weight_init=0.15, cont_weight_growth=0.02, threshold_decay=0.02):
        super().__init__()
        self.cont_weight = cont_weight
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            self.h2e = model.module.h2e
        else:
            self.h2e = model.h2e
        self.cont_weight_init = cont_weight_init
        self.cont_weight_growth = cont_weight_growth
        self.threshold_decay = threshold_decay
        self.inputs_embeds = None
        self.iter_no = 0

    def reset_input_embeds(self, x):
        """ Reset input embedding with new input sequence
        """
        self.inputs_embeds = self.h2e(x.data)

    def forward(self, model, decoder, x, kv_cache, block, block_loc, block_id):
        """ The forward computation to decode tokens.
        """
        iter_cont_weight = min(self.cont_weight_init+self.cont_weight_growth*self.iter_no, self.cont_weight)
        iter_threshold = max(1-self.iter_no*self.threshold_decay, decoder.threshold)
        # Update KV-cache
        if kv_cache is not None and kv_cache.require_update(self.iter_no, block_loc.start, block_loc.end):
            output = model(inputs_embeds=self.inputs_embeds, use_cache=True)
            self.num_forwards += 1
            # use the generated output to decode.
            decoder.decode(output.logits[:, block_loc.start:block_loc.end], block_loc.start, block_loc.end, x, iter_threshold)
            # update KV-cache
            mask_index = (x.data == decoder.mask_id)
            self.inputs_embeds = self.h2e(x.data, mask_index, output.logits, iter_cont_weight)
            kv_cache.update(output.past_key_values)
            self.cache_updates += 1
            self.iter_no += 1

        iter_cont_weight = min(self.cont_weight_init+self.cont_weight_growth*self.iter_no, self.cont_weight)
        iter_threshold = max(1-self.iter_no*self.threshold_decay, decoder.threshold)
        if kv_cache is None:
            logits = model(inputs_embeds=self.inputs_embeds).logits
            decoder.decode(logits[:, block_loc.start:block_loc.end], block_loc.start, block_loc.end, x, iter_threshold)
            mask_index = (x.data == decoder.mask_id)
            self.inputs_embeds = self.h2e(x.data, mask_index, logits, iter_cont_weight)
        elif kv_cache.cache_type == 'prefix':
            past_key_values, replace_position = kv_cache.get_key_values(block_loc.start, block_loc.end)
            logits = model(inputs_embeds=self.inputs_embeds[:, block_loc.start:], past_key_values=past_key_values, use_cache=True,
                    replace_position=replace_position).logits
            block_length = block_loc.end - block_loc.start
            decoder.decode(logits[:, :block_length], block_loc.start, block_loc.end, x, iter_threshold)
            mask_index = (x.data[:, block_loc.start:] == decoder.mask_id)
            self.inputs_embeds[:, block_loc.start:] = self.h2e(x.data[:, block_loc.start:], mask_index, logits, iter_cont_weight)
        else:
            past_key_values, replace_position = kv_cache.get_key_values(block_loc.start, block_loc.end)
            # cache position is the position between current_block_start and current_block_end
            logits = model(inputs_embeds=self.inputs_embeds[:, block_loc.start:block_loc.end], past_key_values=past_key_values, use_cache=True,
                    replace_position=replace_position).logits
            decoder.decode(logits, block_loc.start, block_loc.end, x, iter_threshold)
            mask_index = (x.data[:, block_loc.start:block_loc.end] == decoder.mask_id)
            self.inputs_embeds[:, block_loc.start:block_loc.end] = self.h2e(x.data[:, block_loc.start:block_loc.end], mask_index, logits, iter_cont_weight)
        self.num_forwards += 1
        self.iter_no += 1

class IterSmoothDiffusionLLM(BlockWiseDiffusionLLM):
    """ This diffusion LLM inference generates tokens block by block.

    The decoding algorithm break the generation sequence into blocks.
    It runs diffusion iterations on the first block and decodes all tokens
    in the block before moving to the next block.
    This is a classifical dLLM decoding algorithm.
    """
    def __init__(self, model, decoder, iterator_factory, early_stop=True, cache_factory=None, maximum_unroll=4, expected_tpf=8,
                cont_weight=0.3, cont_weight_init=0.15, cont_weight_growth=0.02, threshold_decay=0.02):
        self.model = model
        self.cache_factory = cache_factory
        self.decoder = decoder
        self.iterator_factory = iterator_factory
        self.early_stop = early_stop
        self.maximum_unroll = maximum_unroll
        self.expected_tpf = expected_tpf
        self.diff_iteration = IterationSmooth(self.model, cont_weight, cont_weight_init, cont_weight_growth, threshold_decay)
        self.block_decoder = BlockRunner(self.diff_iteration, early_stop, maximum_unroll, expected_tpf)

    @property
    def num_forwards(self):
        return self.diff_iteration.num_forwards

    @property
    def cache_updates(self):
        return self.diff_iteration.cache_updates
    
    @ torch.no_grad()
    def generate(self, prompt, gen_length=128, block_length=128):
        ''' Generate tokens with diffusion iterations block by block.
        '''
        x = TokenArray(prompt, gen_length, self.decoder.mask_id, self.decoder.eos_id, self.model.device)
        it = self.iterator_factory.create(x, block_length)

        # We need to reset iter_no at the beginning of generating a sequence.
        self.diff_iteration.iter_no = 0
        self.diff_iteration.reset_input_embeds(x)
        kv_cache = self.cache_factory.create() if self.cache_factory is not None else None
        for block_id, (block_loc, block) in enumerate(it):
            self.decoder.block_init(block, block_id)
            decode_compl = self.block_decoder.decode(self.model, self.decoder, x, kv_cache, block, block_loc, block_id)
            # If all sequences have EOS, we have finished decoding.
            if torch.all(decode_compl):
                break
        logger.info(f'The number of diffusion iterations: {self.num_forwards}')
        return x.get_generated_tokens()

class VicinityCacheIteration(DiffusionIteration):
    """ A diffusion iteration to decode tokens
    """
    def __init__(self, prefix_look, after_look, warmup_steps):
        super().__init__()
        self.prefix_look = int(prefix_look)
        self.after_look = int(after_look)
        self.warmup_steps = int(warmup_steps)
        self.iter_no = 0

    def forward(self, model, decoder, x, kv_cache, block, block_loc, block_id):
        """ The forward computation to decode tokens.
        """
        total_len = x.total_length
        block_start, block_end = block_loc.start, block_loc.end
        left_start = max(0, block_start - self.prefix_look)
        right_end = min(total_len, block_end + self.after_look)

        if self.iter_no < self.warmup_steps:
            out_full = model(x.data)
            self.num_forwards += 1
            decoder.decode(out_full.logits[:, block_start:block_end], block_start, block_end, x)
            self.iter_no += 1
            return

        if kv_cache.past_key_values is None or (kv_cache.require_update(self.iter_no, block_start, block_end) and block_id > 0):
            out_full = model(x.data, use_cache=True)
            self.num_forwards += 1
            decoder.decode(out_full.logits[:, block_start:block_end], block_start, block_end, x)
            kv_cache.update(out_full.past_key_values)
            self.cache_updates += 1
            self.iter_no += 1

        window_input = x.data[:, left_start:right_end]
        past_key_values, replace_position = kv_cache.get_key_values(left_start, right_end)
        out_step = model(window_input, past_key_values=past_key_values, use_cache=True, replace_position=replace_position)
        self.num_forwards += 1
        offset = block_start - left_start
        logits_block = out_step.logits[:, offset:offset + (block_end - block_start)]
        decoder.decode(logits_block, block_start, block_end, x)
        self.iter_no += 1

class VicinityCacheDiffusionLLM(BlockWiseDiffusionLLM):
    """ This diffusion LLM inference generates tokens with Vicinity Cache Update.

    The decoding algorithm defines a window to update KV-cache in each diffusion iteration.
    The window can be larger than the decoding block.
    """
    def __init__(self, model, decoder, iterator_factory, cache_factory, maximum_unroll=4, expected_tpf=8,
                 prefix_look=0, after_look=0, warmup_steps=0, early_stop=True):
        self.model = model
        self.cache_factory = cache_factory
        self.decoder = decoder
        self.iterator_factory = iterator_factory
        assert cache_factory is not None, "This class requires a KV-cache."
        self.diff_iteration = VicinityCacheIteration(prefix_look, after_look, warmup_steps)
        self.block_decoder = BlockRunner(self.diff_iteration, early_stop, maximum_unroll, expected_tpf)

    @property
    def num_forwards(self):
        return self.diff_iteration.num_forwards

    @property
    def cache_updates(self):
        return self.diff_iteration.cache_updates

class IterSmoothWithVicinityCache(DiffusionIteration):
    """ A diffusion iteration to decode tokens
    """
    def __init__(self, model, prefix_look, after_look, warmup_steps,
            cont_weight=0.3, cont_weight_init=0.15, cont_weight_growth=0.02, threshold_decay=0.02):
        super().__init__()
        self.prefix_look = int(prefix_look)
        self.after_look = int(after_look)
        self.warmup_steps = int(warmup_steps)

        self.cont_weight = cont_weight
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            self.h2e = model.module.h2e
        else:
            self.h2e = model.h2e
        self.cont_weight_init = cont_weight_init
        self.cont_weight_growth = cont_weight_growth
        self.threshold_decay = threshold_decay
        self.inputs_embeds = None
        self.iter_no = 0
    
    def reset_input_embeds(self, x):
        """ Reset input embedding with new input sequence
        """
        self.inputs_embeds = self.h2e(x.data)

    def forward(self, model, decoder, x, kv_cache, block, block_loc, block_id):
        """ The forward computation to decode tokens.
        """
        total_len = x.total_length
        block_start, block_end = block_loc.start, block_loc.end
        left_start = max(0, block_start - self.prefix_look)
        right_end = min(total_len, block_end + self.after_look)

        iter_cont_weight = min(self.cont_weight_init+self.cont_weight_growth*self.iter_no, self.cont_weight)
        iter_threshold = max(1-self.iter_no*self.threshold_decay, decoder.threshold)
        if self.iter_no < self.warmup_steps:
            out_full = model(inputs_embeds=self.inputs_embeds)
            self.num_forwards += 1
            decoder.decode(out_full.logits[:, block_start:block_end], block_start, block_end, x, iter_threshold)
            mask_index = (x.data == decoder.mask_id)
            self.inputs_embeds = self.h2e(x.data, mask_index, out_full.logits, iter_cont_weight)
            self.iter_no += 1
            return

        if kv_cache.past_key_values is None or (kv_cache.require_update(self.iter_no, block_start, block_end) and block_id > 0):
            out_full = model(inputs_embeds=self.inputs_embeds, use_cache=True)
            self.num_forwards += 1
            decoder.decode(out_full.logits[:, block_start:block_end], block_start, block_end, x, iter_threshold)
            mask_index = (x.data == decoder.mask_id)
            self.inputs_embeds = self.h2e(x.data, mask_index, out_full.logits, iter_cont_weight)
            kv_cache.update(out_full.past_key_values)
            self.cache_updates += 1
            self.iter_no += 1

        iter_cont_weight = min(self.cont_weight_init+self.cont_weight_growth*self.iter_no, self.cont_weight)
        iter_threshold = max(1-self.iter_no*self.threshold_decay, decoder.threshold)
        past_key_values, replace_position = kv_cache.get_key_values(left_start, right_end)
        out_step = model(
                inputs_embeds=self.inputs_embeds[:, left_start:right_end],
                past_key_values=past_key_values,
                use_cache=True,
                replace_position=replace_position
        )

        self.num_forwards += 1
        self.iter_no += 1
        offset = block_start - left_start
        logits_block = out_step.logits[:, offset:offset + (block_end - block_start)]
        decoder.decode(logits_block, block_start, block_end, x, iter_threshold)
        mask_index = (x.data[:, left_start:right_end] == decoder.mask_id)
        self.inputs_embeds[:, left_start:right_end] = self.h2e(x.data[:, left_start:right_end], mask_index, out_step.logits, iter_cont_weight)

class IterSmoothWithVicinityCacheDiffusionLLM(IterSmoothDiffusionLLM):
    """ This diffusion LLM inference generates tokens with vicinity cache and iteration smoothing.
    """
    def __init__(self, model, decoder, iterator_factory, cache_factory, maximum_unroll=4, expected_tpf=8,
                 prefix_look=0, after_look=0, warmup_steps=0, early_stop=True, cont_weight=0.3,
                 cont_weight_init=0.15, cont_weight_growth=0.02, threshold_decay=0.02):
        self.model = model
        self.cache_factory = cache_factory
        self.decoder = decoder
        self.iterator_factory = iterator_factory
        assert cache_factory is not None, "This class requires a KV-cache."
        self.diff_iteration = IterSmoothWithVicinityCache(model, prefix_look, after_look, warmup_steps,
                cont_weight=cont_weight, cont_weight_init=cont_weight_init, cont_weight_growth=cont_weight_growth,
                threshold_decay=threshold_decay)
        self.block_decoder = BlockRunner(self.diff_iteration, early_stop, maximum_unroll, expected_tpf)

    @property
    def num_forwards(self):
        return self.diff_iteration.num_forwards

    @property
    def cache_updates(self):
        return self.diff_iteration.cache_updates


class BlockWiseDiffusionLLMWithSP(DiffusionLLM):
    """ Diffusion LLM inference with sequence parallel.

    This class performs diffusion LLM inference with sequence parallel.

    Parameters
    ----------
    rank : int
        The rank of the process
    world_size : int
        The number of processes to perform diffusion LLM inference with sequence parallel.
    model : Torch.Module
        The diffusion LLM model
    decoder : ParallelDecoder
        The decoder that decodes the tokens from the logits computed by the Transformer model
    iterator_facotry : IteratorFactory
        The factory class that generates the iterator on the input token array.
    """
    def __init__(self, rank, world_size, model, decoder, iterator_factory):
        self.model = model
        self.decoder = decoder
        self.iterator_factory = iterator_factory
        self.rank = rank
        self.world_size = world_size
        self.num_forwards = 0

    @ torch.no_grad()
    def generate(self, prompt, gen_length=128, block_length=128):
        '''
        Args:
            prompt: A tensor of shape (1, L).
            gen_length: Generated answer length.
            block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        '''
        op_num = 0
        x = DistAlignedTokenArray(prompt, gen_length, self.decoder.mask_id, self.decoder.eos_id, self.model.device, self.rank, self.world_size)
        it = self.iterator_factory.create(x, block_length)

        for block_id, (block_loc, block) in enumerate(it):
            self.decoder.block_init(block, block_id)
            while (block == self.decoder.mask_id).sum()>0:
                part = x.total_length // self.world_size
                # TODO(zhengda) How does the model collect KV from other processes.
                partial_logits = self.model(x[:, (self.rank * part):((self.rank + 1) * part)].clone()).logits
                op_num += calculate_op_num(x[:, self.rank*part:(self.rank+1)*part])

                logits = gather_sequence_block(partial_logits, self.rank * part, (self.rank + 1) * part, block_loc.start, block_loc.end,
                        self.rank, self.world_size)
                self.decoder.decode(logits, block_loc.start, block_loc.end, x)
                self.num_forwards += 1
        return x.get_generated_tokens()

class BlockDiffusionLLMAttnmask(DiffusionLLM):
    """ Diffusion LLM inference

    This diffusion LLM inference generates tokens block by block with the implementation of Attention Mask.

    Comparing to the BlockWiseDiffusionLLM, this one does not feed the subsequent blocks 
    (which consist only of mask tokens) into the transformer when generating the earlier blocks, 
    thereby reducing overhead.

    Parameters
    ----------
    model : Torch.Module
        The LLM model
    decoder : ParallelDecoder
        The decoder that decodes the tokens from the logits computed by the Transformer model
    iterator_facotry : IteratorFactory
        The factory class that generates the iterator on the input token array.

    """
    def __init__(self, model, decoder, iterator_factory, early_stop=True, maximum_unroll=4, expected_tpf=8, backend='vllm'):
        self.model = model
        self.decoder = decoder
        self.iterator_factory = iterator_factory
        self.diff_iteration = BlockDiffusionIteration()
        self.block_runner = BlockDiffusionRunner(self.diff_iteration, early_stop, maximum_unroll, expected_tpf, backend)
        

    @property
    def num_forwards(self):
        return self.diff_iteration.num_forwards

    @property
    def cache_updates(self):
        return 0

    @ torch.no_grad()
    def generate(self, prompt, gen_length=128, block_length=128):
        ''' Generate tokens with diffusion iterations block by block.
        '''
        assert prompt.shape[0] == 1, "We currently only support batch size = 1."
        # recalculate gen length and init iteratory
        # TODO(dulun): the implementation align with original bd decoder implementation.
        # We may need to refine to let users control the gen_length.
        prompt_length=prompt.shape[1]
        num_blocks = (prompt_length + gen_length + block_length - 1) // block_length
        total_length = num_blocks * block_length
        new_gen_length=total_length-prompt_length
        
        
        # prepare block_mask and position IDs
        block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=self.model.device))
        bd_attn_mask = block_mask.repeat_interleave(block_length, dim=0)\
                                        .repeat_interleave(block_length, dim=1).unsqueeze(0)
        pos_ids = torch.arange(total_length, device=self.model.device).unsqueeze(0)


        x = TokenArray(prompt, new_gen_length, self.decoder.mask_id, self.decoder.eos_id, self.model.device)
        it = self.iterator_factory.create(x, block_length)

        # We need to reset iter_no at the beginning of generating a sequence.
        self.diff_iteration.iter_no = 0
        # We don't need kv_cache for the implementation of attention mask
        kv_cache = None
        for block_id, (block_loc, block) in enumerate(it):
            self.decoder.block_init(block, block_id)
            decode_compl = self.block_runner.decode(self.model, self.decoder, x, kv_cache, block, block_loc, block_id, 
                pos_ids, bd_attn_mask)
            if decode_compl:
                break
        logger.info(f'The number of diffusion iterations: {self.num_forwards}')
        return x.get_generated_tokens()


def gather_blocks(x: torch.Tensor, idx: torch.Tensor, block_length: int) -> torch.Tensor:
    # Gather blocks from a batch to form a mini-batch input ids, select on block from each sequence,
    # idx is a 1-d tensor indicating the block start position of each sequence, block length indecates the length of a block
    n, L = x.shape
    offsets = torch.arange(block_length, device=x.device).unsqueeze(0)          # (1, block_length)
    indices = idx.unsqueeze(1) + offsets                                       # (n, block_length)
    blocks = torch.gather(x, dim=1, index=indices)
    return blocks

def select_batch_sequences_by_mask_number(x, valid_flag, mask_id, batch_size):
    # Select sequences to build a mini-batch, using sequences with most mask tokens
    cand_idx = torch.nonzero(valid_flag, as_tuple=False).squeeze(1)  # shape (N,)
    _, sorted_order = torch.sort(-(x.data[cand_idx]==mask_id).sum(dim=1), stable=True)  
    top_order = sorted_order[:batch_size] 
    return cand_idx[top_order]

def select_batch_sequences_by_order(x, valid_flag, mask_id, batch_size):
    # Select sequences to build a mini-batch, selecting simply by sequence order
    return torch.nonzero(valid_flag, as_tuple=True)[0][:batch_size]

select_prefilling_batch_sequences = select_batch_sequences_by_mask_number
select_decoding_batch_sequences = select_batch_sequences_by_mask_number

class BlockDiffusionLLM(DiffusionLLM):
    """ Diffusion LLM inference

    This diffusion LLM inference generates tokens block by block with the implementation of KV-Cache

    Comparing to the BlockWiseDiffusionLLM, this one does not feed the subsequent blocks 
    (which consist only of mask tokens) into the transformer when generating the earlier blocks, 
    thereby reducing overhead.

    Parameters
    ----------
    model : Torch.Module
        The LLM model
    decoder : ParallelDecoder
        The decoder that decodes the tokens from the logits computed by the Transformer model
    iterator_facotry : IteratorFactory
        The factory class that generates the iterator on the input token array.
    cache_factory : CacheFactory
        The factory class that creates the KV-cache.
    early_stop : bool, default=True
        If True, generation of each sequence stops as soon as it has generated the EOS token.
    maximum_unroll : int, default=1
        Maximum number of forwards to unroll at a time, to reduce cuda graph overhead.
    expected_tpf : int, default=15
        Expected tokens per forward pass, used for controling unrolling).
    backend : str, default='vllm'
        Inference backend. Options include 'vllm', 'sglang'
    mini_batch_size : int, default=4
        Size of mini-batches used in dynamic batching.
    prefilling_limit : int, default=128
        Maximum length for prefilling the KV-cache, rest of the input prompt will be prefilled block by block.
    use_naive_batching : bool, default=True
        If True, uses naive batching; otherwise, uses dynamic batching.
    """
    def __init__(self, model, decoder, iterator_factory, cache_factory, early_stop=True, maximum_unroll=1, expected_tpf=15, backend='vllm', 
                 mini_batch_size=4, prefilling_limit=128, use_naive_batching=True):
        self.model = model
        self.decoder = decoder
        self.iterator_factory = iterator_factory
        self.cache_factory = cache_factory
        self.diff_iteration = BlockDiffusionIteration()
        self.block_runner = BlockDiffusionRunner(self.diff_iteration, early_stop, maximum_unroll, expected_tpf, backend)
        self.early_stop = early_stop
        self.backend = backend
        self.mini_batch_size = mini_batch_size
        self.prefilling_limit = prefilling_limit
        self.use_naive_batching = use_naive_batching
        self._bd_mask_cache = {}
        self._pos_ids_cache = {}
        self._cross_block_mask_cache = {}
        if self.use_naive_batching or self.backend != 'sglang':
            self.generate = self.naive_batching_generate_uniform
        else:
            raise RuntimeError("dynamic_batching_generate is not supported in this configuration.")

    def _get_bd_attn_mask(self, batch_size, attn_mask_num_blocks, block_length):
        key = (batch_size, attn_mask_num_blocks, block_length, *_device_key(self.model.device))
        bd_attn_mask = self._bd_mask_cache.get(key)
        if bd_attn_mask is None:
            block_mask = torch.tril(
                torch.ones(
                    attn_mask_num_blocks,
                    attn_mask_num_blocks,
                    device=self.model.device,
                    dtype=torch.bool,
                )
            )
            bd_attn_mask = block_mask.repeat_interleave(block_length, dim=0).repeat_interleave(
                block_length, dim=1
            ).unsqueeze(0).repeat(batch_size, 1, 1)
            self._bd_mask_cache[key] = bd_attn_mask
        return bd_attn_mask

    def _get_pos_ids(self, batch_size, total_length, device):
        key = (batch_size, total_length, *_device_key(device))
        pos_ids = self._pos_ids_cache.get(key)
        if pos_ids is None:
            pos_ids = torch.arange(total_length, device=device).unsqueeze(0).repeat(batch_size, 1)
            self._pos_ids_cache[key] = pos_ids
        return pos_ids

    def _get_cross_block_attn_mask(self, batch_size, block_length, cache_length, device):
        key = (batch_size, block_length, cache_length, *_device_key(device))
        cross_block_attn_mask = self._cross_block_mask_cache.get(key)
        if cross_block_attn_mask is None:
            cross_block_attn_mask = torch.ones(
                batch_size,
                2 * block_length,
                cache_length,
                device=device,
                dtype=torch.bool,
            )
            cross_block_attn_mask[:, :block_length, -block_length:].fill_(False)
            self._cross_block_mask_cache[key] = cross_block_attn_mask
        return cross_block_attn_mask

    @property
    def num_forwards(self):
        return self.diff_iteration.num_forwards

    @property
    def cache_updates(self):
        return self.diff_iteration.cache_updates

    @ torch.no_grad()
    def naive_batching_generate(self, prompt, gen_length=128, block_length=128):
        ''' Generate tokens with diffusion iterations block by block.
        '''
        # recalculate gen length and init iteratory
        # TODO(dulun): the implementation align with original bd decoder implementation.
        # We may need to refine to let users control the gen_length.
        batch_size = prompt.shape[0]
        prompt_length = prompt.shape[1]
        num_blocks = (prompt_length + gen_length + block_length - 1) // block_length
        total_length = num_blocks * block_length
        new_gen_length = total_length-prompt_length
        
        mask_length = (max(self.cache_factory.max_length, prompt_length + gen_length)+block_length-1)//block_length*block_length
        attn_mask_num_blocks = mask_length // block_length

        # prepare block_mask and position IDs
        bd_attn_mask = self._get_bd_attn_mask(batch_size, attn_mask_num_blocks, block_length)
        pos_ids = self._get_pos_ids(batch_size, total_length, self.model.device)

        x = TokenArray(prompt, new_gen_length, self.decoder.mask_id, self.decoder.eos_id, self.model.device)
        it = self.iterator_factory.create(x, block_length)
        prompt_length = it._get_first_block_start()
        kv_cache = self.cache_factory.create()

        # prefill for kv_cache
        prefill_blocks = prompt_length // block_length
        prefill_length = prefill_blocks * block_length
        prefill_length = max(prefill_length, block_length)
        self.block_runner.prefill(self.model, x[:, :prefill_length], kv_cache, pos_ids[:, :prefill_length], bd_attn_mask[:,:prefill_length,:prefill_length], self.prefilling_limit, block_length)
        
        # We need to reset iter_no at the beginning of generating a sequence.
        self.diff_iteration.iter_no = 0
        for block_id, (block_loc, block) in enumerate(it):
            self.decoder.block_init(block, block_id)
            if self.backend == 'vllm':
                cross_block_attn_mask = bd_attn_mask[:,block_loc.start-block_length:block_loc.end, :block_loc.end]
            else:
                cross_block_attn_mask = self._get_cross_block_attn_mask(
                    batch_size,
                    block_length,
                    kv_cache.past_key_values._data.shape[4],
                    prompt.device,
                )
            decode_compl = self.block_runner.decode(self.model, self.decoder, x, kv_cache, block, block_loc, block_id, pos_ids, bd_attn_mask, block_length, cross_block_attn_mask)
            if torch.all(decode_compl) and self.early_stop:
                break
        logger.info(f'The number of diffusion iterations: {self.num_forwards}')
        return x.get_generated_tokens()
    

    @ torch.no_grad()
    def naive_batching_generate_uniform(self, prompt, gen_length=128, block_length=128):
        ''' Generate tokens with diffusion iterations block by block.
        '''
        # recalculate gen length and init iteratory
        # TODO(dulun): the implementation align with original bd decoder implementation.
        # We may need to refine to let users control the gen_length.
        batch_size = prompt.shape[0]
        prompt_length = prompt.shape[1]
        num_blocks = (prompt_length + gen_length + block_length - 1) // block_length
        total_length = num_blocks * block_length
        new_gen_length = total_length-prompt_length
        
        mask_length = (max(self.cache_factory.max_length, prompt_length + gen_length)+block_length-1)//block_length*block_length
        attn_mask_num_blocks = mask_length // block_length

        # prepare block_mask and position IDs
        bd_attn_mask = self._get_bd_attn_mask(batch_size, attn_mask_num_blocks, block_length)

        pos_ids = self._get_pos_ids(batch_size, total_length, self.model.device)

        x = TokenArray(prompt, new_gen_length, self.decoder.mask_id, self.decoder.eos_id, self.model.device)
        it = self.iterator_factory.create(x, block_length)
        prompt_length = it._get_first_block_start()
        kv_cache = self.cache_factory.create()

        # prefill for kv_cache
        prefill_blocks = prompt_length // block_length
        prefill_length = prefill_blocks * block_length
        prefill_length = max(prefill_length, block_length)
        self.block_runner.prefill(self.model, x[:, :prefill_length], kv_cache, pos_ids[:, :prefill_length], bd_attn_mask[:,:prefill_length,:prefill_length], self.prefilling_limit, block_length)
        
        # We need to reset iter_no at the beginning of generating a sequence.
        self.diff_iteration.iter_no = 0
        for block_id, (block_loc, block) in enumerate(it):
            self.decoder.block_init(block, block_id)
            if self.backend == 'vllm':
                cross_block_attn_mask = bd_attn_mask[:,block_loc.start-block_length:block_loc.end, :block_loc.end]
            else:
                cross_block_attn_mask = self._get_cross_block_attn_mask(
                    batch_size,
                    block_length,
                    kv_cache.past_key_values._data.shape[4],
                    prompt.device,
                )
            decode_compl = self.block_runner.decode_uniform(self.model, self.decoder, x, kv_cache, block, block_loc, block_id, pos_ids, bd_attn_mask, block_length, cross_block_attn_mask)
            if torch.all(decode_compl) and self.early_stop:
                break
        logger.info(f'The number of diffusion iterations: {self.num_forwards}')
        return x.get_generated_tokens()





