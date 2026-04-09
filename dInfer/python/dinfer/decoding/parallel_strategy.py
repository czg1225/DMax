import math
import torch
import numpy as np
import torch.nn.functional as F

from .utils import add_gumbel_noise, get_num_transfer_tokens
import torch.distributed as dist


def broadcast_if_needed(x, src=0, group=None):
    if dist.is_available() and dist.is_initialized() and dist.get_world_size(group) > 1:
        dist.broadcast(x, src=src)


def _get_prob_stats(logits, indices, use_float64=False, x_is_argmax=False):
    logits_fp = logits.to(torch.float64 if use_float64 else torch.float32)
    log_norm = torch.logsumexp(logits_fp, dim=-1)
    chosen_logits = torch.gather(
        logits_fp, dim=-1, index=indices.unsqueeze(-1)
    ).squeeze(-1)
    chosen_probs = torch.exp(chosen_logits - log_norm)
    if x_is_argmax:
        max_indices = indices
        max_probs = chosen_probs
    else:
        max_logits, max_indices = torch.max(logits_fp, dim=-1)
        max_probs = torch.exp(max_logits - log_norm)
    return chosen_probs, max_probs, max_indices


@torch.no_grad()
@torch.compile(dynamic=True)
def get_transfer_index_hierarchy_fast_v2(
    logits,
    temperature,
    remasking,
    mask_index,
    x,
    num_transfer_tokens,
    mask_id,
    threshold=None,
    low_threshold=None,
):
    if not math.isclose(temperature, 0.0):
        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    else:
        logits_with_noise = logits

    x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l

    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float32), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)  # b, l
    else:
        raise NotImplementedError(remasking)

    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    if num_transfer_tokens is not None:
        assert threshold is None
        for j in range(confidence.shape[0]):
            _, select_index = torch.topk(
                confidence[j], k=num_transfer_tokens[j]
            )
            transfer_index[j, select_index] = True

    else:
        for i in range(mask_index.shape[0]):

            mask_i = mask_index[i].int()
            conf_i = confidence[i]

            if low_threshold is not None:
                max_value, max_index = torch.max(conf_i, dim=0)
                if max_value < low_threshold:
                    transfer_index[i, max_index] = True
                    continue

            diff = torch.diff(
                torch.cat([mask_i[:1] * 0, mask_i, mask_i[-1:] * 0])
            )
            starts = (diff == 1).nonzero(as_tuple=True)[0]
            ends = (diff == -1).nonzero(as_tuple=True)[0]

            if len(starts) > 0:
                max_indices = [
                    s + torch.argmax(conf_i[s:e])
                    for s, e in zip(starts.tolist(), ends.tolist())
                ]
                transfer_index[i, max_indices] = True

            if low_threshold is not None:
                transfer_index[i] = torch.logical_and(
                    transfer_index[i], conf_i > low_threshold
                )

        if threshold is not None:
            transfer_index = torch.logical_or(
                transfer_index, confidence > threshold
            )

    return x0, transfer_index


@torch.no_grad()
def get_transfer_index_hierarchy_remask(
    logits,
    temperature,
    mask_index,
    x,
    num_transfer_tokens,
    mask_id,
    threshold=None,
    low_threshold=None,
    remask_threshold=0.4,
):
    if not math.isclose(temperature, 0.0):
        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    else:
        logits_with_noise = logits

    x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l

    p = F.softmax(logits, dim=-1)
    x0_p = torch.squeeze(
        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
    )  # b, l

    lower_index = x0_p < remask_threshold
    remask_index = torch.logical_and(
        lower_index, torch.logical_not(mask_index)
    )
    mask_new = torch.logical_or(lower_index, mask_index)

    confidence = torch.where(mask_new, x0_p, float('-inf'))

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)

    remask_cnt = remask_index.sum(dim=1)

    if num_transfer_tokens is not None:
        assert threshold is None
        for j in range(confidence.shape[0]):
            _, select_index = torch.topk(
                confidence[j], k=num_transfer_tokens[j]
            )
            transfer_index[j, select_index] = True

    else:
        for i in range(mask_new.shape[0]):
            mask_i = mask_new[i].int()
            conf_i = confidence[i]

            diff = torch.diff(
                torch.cat([mask_i[:1] * 0, mask_i, mask_i[-1:] * 0])
            )
            starts = (diff == 1).nonzero(as_tuple=True)[0]
            ends = (diff == -1).nonzero(as_tuple=True)[0]

            if len(starts) > 0:
                max_indices = [
                    s + torch.argmax(conf_i[s:e])
                    for s, e in zip(starts.tolist(), ends.tolist())
                ]
                transfer_index[i, max_indices] = True

            if low_threshold is not None:
                transfer_index[i] = torch.logical_and(
                    transfer_index[i], conf_i > low_threshold
                )

            if threshold is not None:
                transfer_index[i] = torch.logical_or(
                    transfer_index[i], conf_i > threshold
                )

            gap = int((remask_cnt[i] + 1 - transfer_index[i].sum()).item())
            if gap > 0:
                conf_i[transfer_index[i]] = float('-inf')
                values, indices = torch.topk(
                    conf_i, gap, largest=True, sorted=False
                )
                transfer_index[i][indices] = True

    remask_index = torch.logical_and(
        remask_index, torch.logical_not(transfer_index)
    )
    x0[remask_index] = mask_id
    transfer_index[remask_index] = True

    return x0, transfer_index


def get_transfer_index_cache(
    logits,
    mask_index,
    x,
    block_end,
    num_transfer_tokens,
    temperature,
    remasking,
    threshold=None,
    minimal_topk=1,
):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l
    if remasking == 'low_confidence':
        p = F.softmax(logits[mask_index].to(torch.float32), dim=-1).to(
            logits.dtype
        )
        x0_p = torch.squeeze(
            torch.gather(
                p,
                dim=-1,
                index=torch.unsqueeze(x0[mask_index], -1),
            ),
            -1,
        )  # b, l
        confidence = torch.full(
            x0.shape, -np.inf, device=x0.device, dtype=logits.dtype
        )
        confidence[mask_index] = x0_p
        confidence[:, block_end:] = -np.inf

    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
        x0_p[:, block_end:] = -np.inf
        x0 = torch.where(mask_index, x0, x)
        confidence = torch.where(mask_index, x0_p, -np.inf)
    else:
        raise NotImplementedError(remasking)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    if threshold is not None:
        num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    for j in range(confidence.shape[0]):
        _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j])
        transfer_index[j, select_index] = True
        if threshold is not None:
            for k in range(minimal_topk, num_transfer_tokens[j]):
                if confidence[j, select_index[k]] < threshold:
                    transfer_index[j, select_index[k]] = False
    return x0, transfer_index


def get_transfer_index(
    logits,
    temperature,
    remasking,
    mask_index,
    x,
    num_transfer_tokens,
    threshold=None,
):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l

    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)  # b, l
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)

    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    if threshold is not None:
        num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    for j in range(confidence.shape[0]):
        _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j])
        transfer_index[j, select_index] = True
        if threshold is not None:
            for k in range(1, num_transfer_tokens[j]):
                if confidence[j, select_index[k]] < threshold:
                    transfer_index[j, select_index[k]] = False
    return x0, transfer_index


def get_transfer_index_dynamic(
    logits,
    temperature,
    remasking,
    mask_index,
    x,
    num_transfer_tokens,
    factor=1,
):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l
    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)  # b, l
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)

    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)

    for j in range(confidence.shape[0]):
        ns = list(range(1, num_transfer_tokens[j] + 1))
        es = [factor / (n + 1) for n in ns]
        threshs = [1 - e for e in es]

        # at least one token is transferred
        threshs[0] = -1
        sorted_confidence = torch.sort(
            confidence[j][mask_index[j]], dim=-1, descending=True
        )[0]
        assert len(sorted_confidence) == len(threshs)
        for top_i in range(len(threshs)):
            if sorted_confidence[top_i] < threshs[top_i]:
                break

        if top_i == 0 or top_i == len(threshs) - 1:
            top_i += 1

        _, select_index = torch.topk(confidence[j], k=top_i)
        transfer_index[j, select_index] = True

    return x0, transfer_index


class ParallelDecoder:
    """ This is a parallel decoder that decodes tokens in a block.
    """
    def __init__(
            self,
            temperature,
            remasking='low_confidence',
            mask_id=126336,
    ):
        self.temperature = temperature
        self.remasking = remasking
        self.mask_id = mask_id

    def block_init(self, block_x, block_id):
        pass

    def decode(self, logits, block_start, block_end, x):
        """ Decode the logits in a block.

        Parameters
        ----------
        logits : Tensor
            The logits in a block
        block_start : int
            The location of the starting token in the block
        block_end : int
            The location of the ending token in the block.
        x : Tensor
            The tensor where the decoded tokens are written to.
        """

# Parallel decoding only
@torch.compile(dynamic=True)
def get_transfer_index_threshold(
    logits,
    temperature,
    mask_index,
    x,
    mask_id,
    threshold,
    rm_mask=True,
    use_float64=False,
    **kwargs,
):


    if math.isclose(temperature, 0.0):
        x0 = torch.argmax(logits, dim=-1)  # b, l
        x0_p, _, _ = _get_prob_stats(
            logits, x0, use_float64=use_float64, x_is_argmax=True
        )
    else:
        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l
        x0_p, _, _ = _get_prob_stats(logits, x0, use_float64=use_float64)

    # gurantee the denoised token will not be the mask_id
    if rm_mask:
        mask_index = mask_index & (x0 != mask_id)
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)
    # print(x0_p)

    actual_threshold = (
        torch.max(confidence, dim=1)[0] - 1e-5
    ).clamp(-1000, threshold).unsqueeze(-1)
    transfer_index = confidence >= actual_threshold
    return x0, transfer_index

# Parallel decoding only
@torch.compile(dynamic=True)
def get_transfer_index_uniform(
    logits,
    temperature,
    mask_index,
    mask_id,
    threshold,
    rm_mask=True,
    use_float64=False,
    **kwargs,
):  
   
    if temperature == 0:
        # Greedy 
        x0 = torch.argmax(logits, dim=-1) # b, l
    else:
        # Sampling 
        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l

    # logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    # x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l

    x0_p, max_probs, max_indices = _get_prob_stats(
        logits,
        x0,
        use_float64=use_float64,
        x_is_argmax=math.isclose(temperature, 0.0),
    )



    # 1. Compute confidence
    # Set non-mask positions to -inf so they do not affect selection
    confidence = torch.where(mask_index, x0_p, float('-inf'))

    # 2. Mark low-confidence masked positions
    # A position is low-confidence if it is masked and below the threshold
    is_low_conf = mask_index & (confidence < threshold)

    # 3. Find the cutoff boundary from left to right
    # After the first low-confidence position, all later positions are marked
    has_encountered_failure = torch.cumsum(is_low_conf.long(), dim=1) > 0

    # 4. Build candidate mask
    # Select masked positions before the first low-confidence masked token
    candidates = mask_index & (~has_encountered_failure)

    # 5. Fallback rule
    # If no candidate is selected, decode the leftmost masked token

    # 5.1 Check whether each batch has any selected token
    batch_has_selection = candidates.any(dim=-1, keepdim=True)

    # 5.2 Find the first masked token in each batch
    mask_cumsum = torch.cumsum(mask_index.long(), dim=1)
    first_mask_token = (mask_cumsum == 1) & mask_index

    # 5.3 Final selection
    # Use candidates if available; otherwise use the first masked token
    high_conf_index = torch.where(batch_has_selection, candidates, first_mask_token)

    return x0, high_conf_index, max_probs, max_indices





class ThresholdParallelDecoder(ParallelDecoder):
    """Parallel decoding driven by a confidence threshold."""
    def __init__(
            self,
            temperature,
            threshold,
            remasking='low_confidence',
            mask_id=126336,
            eos_id=126081,
            use_float64=False,
    ):
        super().__init__(temperature, remasking, mask_id)
        self.threshold = threshold
        self.eos_id = eos_id
        self.use_float64 = use_float64
        self._mask_token_tensor = {}
        self._mask_embed = {}
        self._mask_norm = {}
        self._block_offsets = {}
        self._batch_offsets = {}

    def _should_sync_across_ranks(self):
        return not math.isclose(self.temperature, 0.0)

    def _get_mask_token_tensor(self, device):
        key = (device.type, device.index)
        mask_token_tensor = self._mask_token_tensor.get(key)
        if mask_token_tensor is None:
            mask_token_tensor = torch.tensor(
                [self.mask_id], dtype=torch.long, device=device
            )
            self._mask_token_tensor[key] = mask_token_tensor
        return mask_token_tensor

    def _get_block_offsets(self, block_length, device):
        key = (block_length, device.type, device.index)
        offsets = self._block_offsets.get(key)
        if offsets is None:
            offsets = torch.arange(block_length, device=device).unsqueeze(0)
            self._block_offsets[key] = offsets
        return offsets

    def _get_mask_embed_and_norm(self, embedding_layer, device):
        key = (
            device.type,
            device.index,
            embedding_layer.weight.data_ptr(),
        )
        mask_embed = self._mask_embed.get(key)
        mask_norm = self._mask_norm.get(key)
        if mask_embed is None or mask_norm is None:
            mask_embed = embedding_layer(self._get_mask_token_tensor(device)).unsqueeze(0)
            mask_norm = torch.norm(mask_embed, p=2, dim=-1, keepdim=True)
            self._mask_embed[key] = mask_embed
            self._mask_norm[key] = mask_norm
        return mask_embed, mask_norm

    def _get_batch_offsets(self, batch_size, seq_length, device):
        key = (batch_size, seq_length, device.type, device.index)
        batch_offsets = self._batch_offsets.get(key)
        if batch_offsets is None:
            batch_offsets = (
                torch.arange(batch_size, device=device).unsqueeze(1) * seq_length
            )
            self._batch_offsets[key] = batch_offsets
        return batch_offsets
        

    def decode_uniform(self, logits, block_start, block_end, x, active_index, embedding_layer, prev_embeddings=None, iter_threshold=None, top_k=1):
        """ Decode the logits in the same block of multiple samples.
            Added functionality: Compute and return soft embeddings for specified positions.
        """
        if iter_threshold is None:
            iter_threshold = self.threshold
        mask_index = (x[:, block_start:block_end] == self.mask_id)
        assert active_index.shape[1] == logits.shape[1]


        curr_x = x[:, block_start:block_end]
   
        x0, high_conf_index, max_probs, max_indices = get_transfer_index_uniform(
            logits,
            self.temperature,
            mask_index,
            self.mask_id,
            threshold=iter_threshold,
            use_float64=self.use_float64,
        )
        
        # update block token
        if mask_index.any():
            cond = active_index & (~mask_index)
            update_mask = high_conf_index | cond
        else:
            update_mask = active_index

        changed_mask = update_mask & (x0 != curr_x)
        if update_mask.any():
            curr_x[update_mask] = x0[update_mask]

        if self._should_sync_across_ranks():
            broadcast_if_needed(x.data)
        
        # Get the maximum confidence for each token in the current block
        # Condition 1: all active tokens in the block have confidence > 0.9
        if (max_probs[active_index] >= 0.9).all():
            Breakflag = True
        # Condition 2: if not, check whether nothing changed
        elif not changed_mask.any():
            Breakflag = True
        # Otherwise, keep iterating
        else:
            Breakflag = False
        
        if Breakflag:
            return Breakflag, prev_embeddings

        
        # 1. Identify positions for soft embedding
        new_mask_index = (curr_x == self.mask_id)
        
        # Update soft_cond using intersection (&)
        # Must be active and not masked
        soft_cond = active_index & (~new_mask_index) 
        if prev_embeddings is None:
            # The first iteration still uses full hard embeddings
            base_embeds = embedding_layer(curr_x)
        else:
            base_embeds = prev_embeddings
            hard_refresh_mask = (~soft_cond) & changed_mask
            if hard_refresh_mask.any():
                base_embeds[hard_refresh_mask] = embedding_layer(curr_x[hard_refresh_mask])

        if not soft_cond.any():
            if self._should_sync_across_ranks():
                broadcast_if_needed(base_embeds.data)
            return Breakflag, base_embeds
        
        # 3. Compute soft embeddings by mixing Top-K tokens with the mask token
        if top_k == 1:
            topk_probs = max_probs.unsqueeze(-1)
            topk_indices = max_indices.unsqueeze(-1)
        else:
            probs = F.softmax(
                logits.to(torch.float64 if self.use_float64 else torch.float32),
                dim=-1,
            )
            topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)

        residual_probs = torch.clamp(
            1.0 - topk_probs.sum(dim=-1, keepdim=True), min=0.0
        )

        topk_embeds = embedding_layer(topk_indices)

        mask_embed, mask_norm = self._get_mask_embed_and_norm(embedding_layer, x.device)

        topk_weighted = (topk_embeds * topk_probs.unsqueeze(-1)).sum(dim=2)
        mask_weighted = mask_embed * residual_probs
        
        soft_embeds = topk_weighted + mask_weighted
        
        # Renormalization
        # 1. Compute the L2 norm of the mixed vector
        current_norm = torch.norm(soft_embeds, p=2, dim=-1, keepdim=True)
        
        # 2. Compute the L2 norm of top-k and mask embeddings separately
        # topk_embeds shape is usually [batch, seq_len, top_k, hidden_dim]
        # topk_norms shape is [batch, seq_len, top_k]
        topk_norms = torch.norm(topk_embeds, p=2, dim=-1)
        
        # 3. Compute the target norm by weighting norms with probabilities
        # Expected norm of the top-k part, final shape [batch, seq_len, 1]
        expected_topk_norm = (topk_norms * topk_probs).sum(dim=-1, keepdim=True)
        # Expected norm of the mask part, residual_probs shape [batch, seq_len, 1]
        expected_mask_norm = mask_norm * residual_probs
        
        # Target norm is the sum of both parts
        target_norm = expected_topk_norm + expected_mask_norm

        # 4. Rescale to restore the normal hypersphere manifold
        # Add 1e-6 to avoid division by zero
        soft_embeds = soft_embeds * (target_norm / (current_norm + 1e-6))
        if soft_embeds.dtype != base_embeds.dtype:
            soft_embeds = soft_embeds.to(base_embeds.dtype)
        # ------------------------------------------------

        # 4. Merge embeddings based on the condition
        base_embeds[soft_cond] = soft_embeds[soft_cond]

        if self._should_sync_across_ranks():
            broadcast_if_needed(base_embeds.data)


        return Breakflag, base_embeds



    def decode(self, logits, block_start, block_end, x, iter_threshold=None):
        """ Decode the logits in the same block of multiple samples.
        """
        if iter_threshold is None:
            iter_threshold = self.threshold
        mask_index = (x[:, block_start:block_end] == self.mask_id)
        assert mask_index.shape[1] == logits.shape[1]

        curr_x = x[:, block_start:block_end]
        x0, transfer_index = get_transfer_index_threshold(
            logits,
            self.temperature,
            mask_index,
            curr_x,
            self.mask_id,
            threshold=iter_threshold,
            use_float64=self.use_float64,
        )


        transfer_index = torch.logical_and(transfer_index, mask_index)
        assert transfer_index.dtype == torch.bool
        x[:, block_start:block_end] = torch.where(transfer_index, x0, curr_x)
        if self._should_sync_across_ranks():
            broadcast_if_needed(x.data)


    def batch_decode(self, logits, block_start, x, block_length, iter_threshold=None):
        """ Decode the logits in the different blocks of multiple samples, indicated by 1-d block_start tensor.
        """
        if iter_threshold is None:
            iter_threshold = self.threshold
        B, T = x.data.shape
        device = x.data.device

        offset = self._get_block_offsets(block_length, device) + block_start.unsqueeze(1)

        x_block = torch.gather(x.data, 1, offset.clamp(max=T - 1)) 

        mask_index = (x_block == self.mask_id)

        x0, transfer_index = get_transfer_index_threshold(
            logits,
            self.temperature,
            mask_index,
            x_block,
            self.mask_id,
            threshold=iter_threshold,
            use_float64=self.use_float64,
        )

        transfer_index = transfer_index & mask_index

        x_updated = torch.where(transfer_index, x0, x_block)

        x_flat = x.data.view(-1)
        flat_idx = offset + self._get_batch_offsets(B, T, device)
        x_flat[flat_idx] = x_updated
        if self._should_sync_across_ranks():
            broadcast_if_needed(x.data)


class CreditThresholdParallelDecoder(ThresholdParallelDecoder):
    """ This decoder deocdes tokens in parallel based on a threshold + credit.
    The decoder decodes a token when its confidence is larger than a threshold.
    """
    def __init__(self,
                 credit_alpha=0.7,
                 boost_gamma=0.2,
                 decay_beta=0.8,
                 **kwargs):
        super().__init__(**kwargs)

        self.credit_alpha = credit_alpha
        self.boost_gamma = boost_gamma
        self.decay_beta = decay_beta

        self._credit_mats = {}
        self._credit_iters = {}

    def _apply_credit_fusion(self, logits, mask_index, key):
        """
        EMA-based credit fusion (no CM, no pre-credit):
        - Maintains a per-block CreditMatrix (EMA with decay).
        - Accumulates enhanced top-1 probability only on masked positions.
        - Returns fused_logits.
        """
        B, L, V = logits.shape
        device = logits.device

        mat = self._credit_mats.get(key, None)
        if mat is None or mat.shape != (B, L, V) or mat.device != device:
            mat = torch.zeros((B, L, V), dtype=torch.float32, device=device)
            self._credit_mats[key] = mat
            self._credit_iters[key] = 0

        iter_idx = self._credit_iters[key]

        if iter_idx > 0:
            mat.mul_(self.decay_beta)

        probs = F.softmax(logits.to(torch.float32), dim=-1)
        top1_probs, top1_idx = torch.max(probs, dim=-1)
        enhanced = top1_probs.pow(self.boost_gamma).to(mat.dtype)
        update_vals = enhanced * mask_index.to(enhanced.dtype)
        mat.scatter_add_(2, top1_idx.unsqueeze(-1), update_vals.unsqueeze(-1))

        fused_logits = logits + self.credit_alpha * torch.log(mat + 1)
        self._credit_iters[key] = iter_idx + 1
        return fused_logits

    def decode(self, logits, block_start, block_end, x, iter_threshold=None):
        """ Decode the logits in a block."""
        if iter_threshold is None:
            iter_threshold = self.threshold
        mask_index = (x[:, block_start:block_end] == self.mask_id)
        assert mask_index.shape[1] == logits.shape[1]

        curr_x = x[:, block_start:block_end]
        key = (block_start, block_end)
        used_logits = self._apply_credit_fusion(logits, mask_index, key)

        x0, transfer_index = get_transfer_index_threshold(
            used_logits,
            self.temperature,
            mask_index,
            curr_x,
            self.mask_id,
            threshold=iter_threshold,
            use_float64=self.use_float64,
        )

        transfer_index = torch.logical_and(transfer_index, mask_index)
        assert transfer_index.dtype == torch.bool
        x[:, block_start:block_end] = torch.where(transfer_index, x0, curr_x)

        if hasattr(x, 'data'):
            has_mask = (x.data == self.mask_id).any()
        else:
            if x.dim() > 0:
                has_mask = (x == self.mask_id).any()
            else:
                has_mask = (x == self.mask_id)

        if not has_mask:
            self._credit_mats.clear()
            self._credit_iters.clear()
        broadcast_if_needed(x.data)


class FixedParallelDecoder(ParallelDecoder):
    """ This decoder decodes tokens in a fixed number of steps."""
    def __init__(
            self,
            temperature,
            steps,
            remasking='low_confidence',
            mask_id=126336,
    ):
        super().__init__(temperature, remasking, mask_id)
        self.steps = steps
        self.iter = 0
        self.mask_id = mask_id

    def block_init(self, block_x, block_id):
        # TODO(zhengda) fix steps when distributed version changes gen length.
        block_mask_index = block_x == self.mask_id
        self.num_transfer_tokens = get_num_transfer_tokens(
            block_mask_index, self.steps
        )
        self.iter = 0

    def decode(self, logits, block_start, block_end, x, iter_threshold=None):
        """ Decode the logits in a block."""
        mask_index = (x[:, block_start:block_end] == self.mask_id)
        assert mask_index.shape[1] == logits.shape[1]

        curr_x = x[:, block_start:block_end]
        x0, transfer_index = get_transfer_index(
            logits,
            self.temperature,
            self.remasking,
            mask_index,
            curr_x,
            self.num_transfer_tokens[:, self.iter],
            None,
        )
        self.iter += 1
        x[:, block_start:block_end][transfer_index] = x0[transfer_index]
        broadcast_if_needed(x.data)


class HierarchyDecoder(ParallelDecoder):
    """Decode tokens hierarchically to force separate decisions."""
    def __init__(
            self,
            temperature,
            remasking='low_confidence',
            mask_id=126336,
            eos_id=126081,
            threshold=None,
            low_threshold=0.4,
    ):
        super().__init__(temperature, remasking, mask_id)
        self.iter = 0
        self.mask_id = mask_id
        self.eos_id = eos_id
        self.threshold = threshold
        self.low_threshold = low_threshold

    def get_transfer_index(self, logits, mask_index, iter_threshold, **kwargs):

        B, L = mask_index.shape

        # TODO(DuLun): support batch size > 1
        assert B == 1

        device = logits.device

        if not math.isclose(self.temperature, 0.0):
            logits_with_noise = add_gumbel_noise(
                logits, temperature=self.temperature
            )
        else:
            logits_with_noise = logits

        x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l

        x0_logp = F.log_softmax(logits, dim=-1).gather(
            -1, x0.unsqueeze(-1)
        ).squeeze(-1)
        x0_p = x0_logp.exp()  # b, l

        neg_inf_val = torch.finfo(x0_p.dtype).min
        confidence = torch.where(
            mask_index,
            x0_p,
            torch.tensor(neg_inf_val, device=device, dtype=x0_p.dtype),
        )

        prev = torch.cat(
            [
                mask_index.new_zeros((B, 1), dtype=torch.bool),
                mask_index[:, :-1],
            ],
            dim=1,
        )
        starts = torch.logical_and(mask_index, torch.logical_not(prev))

        seg_id = torch.cumsum(starts.to(torch.int64), dim=-1) - 1
        seg_id = torch.where(mask_index, seg_id, 0)

        seg_max = torch.full(
            (B, L), neg_inf_val, device=device, dtype=confidence.dtype
        )
        seg_max = torch.scatter_reduce(
            seg_max,
            dim=1,
            index=seg_id,
            src=confidence,
            reduce='amax',
            include_self=True,
        )

        seg_max_at_pos = seg_max.gather(dim=1, index=seg_id)
        transfer_index = (confidence == seg_max_at_pos)

        if self.low_threshold is not None:
            transfer_index = torch.logical_and(
                transfer_index, torch.gt(confidence, self.low_threshold)
            )
        if iter_threshold is not None:
            transfer_index = torch.logical_or(
                transfer_index, torch.gt(confidence, iter_threshold)
            )

        top1_idx = torch.argmax(confidence, dim=-1)
        top1 = torch.nn.functional.one_hot(
            top1_idx, num_classes=L
        ).to(torch.bool)
        transfer_index = torch.logical_or(transfer_index, top1)

        return x0, transfer_index

    def block_init(self, block_x, block_id):
        # TODO(zhengda) fix steps when distributed version changes gen length.
        self.iter = 0

    def decode(self, logits, block_start, block_end, x, iter_threshold=None):
        """ Decode the logits in a block.
        """
        if iter_threshold is None:
            iter_threshold = self.threshold
        mask_index = (x[:, block_start:block_end] == self.mask_id)
        assert mask_index.shape[1] == logits.shape[1]

        x0, transfer_index = self.get_transfer_index(
            logits, mask_index, iter_threshold
        )
        self.iter += 1
        transfer_index = torch.logical_and(transfer_index, mask_index)
        x[:, block_start:block_end][transfer_index] = x0[transfer_index]
        broadcast_if_needed(x.data)
