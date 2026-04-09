"""
dInfer Diffusion Algorithm implementation for SGLang.

This module implements the dInferDiffusionAlgorithm class that integrates
dInfer's parallel decoding strategies into SGLang's diffusion framework.
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import numpy as np
import logging

from sglang.srt.diffusion.algorithm.base import DiffusionAlgorithm
from sglang.srt.diffusion.config import DiffusionConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner

from dinfer.decoding.parallel_strategy import ThresholdParallelDecoder, HierarchyDecoder

logger = logging.getLogger(__name__)


class dInferDiffusionAlgorithm(DiffusionAlgorithm):
    """
    An integration of dInfer parallel decoding algorithm into SGLang.
    
    This class wraps dInfer's parallel decoding strategies to work within
    SGLang's diffusion framework, enabling efficient parallel decoding for dllm.
    """
    
    def __init__(self, config: DiffusionConfig):
        super().__init__(config)
        self.decoder = ThresholdParallelDecoder(
            temperature=0, 
            threshold=0.9, 
            mask_id=self.mask_id, 
            eos_id=156894
        )
    
    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch, 
    ) -> Tuple[Union[LogitsProcessorOutput, torch.Tensor], List[torch.Tensor], bool]:
        """
        Run the dInfer parallel decoding algorithm.
        
        Args:
            model_runner: The SGLang model runner for forward passes.
            forward_batch: The batch information for forward pass.
            
        Returns:
            A tuple of (logits_output, new_next_token_ids, can_run_cuda_graph).
        """
        batch_size = forward_batch.batch_size

        # Reshape for batch processing
        # Note: block must be a view (not a copy) so modifications sync to forward_batch.input_ids
        block = forward_batch.input_ids.contiguous().view(batch_size, -1)
        
        # Calculate start positions for each block (vectorized)
        block_mask_index = (block == self.mask_id)  # [batch_size, block_size]

        # Fast path: if there is no mask token, forward and save kv cache
        if block_mask_index.sum() == 0:
            logits_output, can_run_cuda_graph = model_runner.forward(
                forward_batch, pp_proxy_tensors=None
            )
            next_token_ids = []
            return logits_output, next_token_ids, can_run_cuda_graph
        
        if forward_batch.return_step_maps:
            step_maps = torch.full_like(block, -1)
        
        start_list = (self.block_size - block_mask_index.sum(dim=1)).tolist()  # List of ints

        while (block_mask_index).sum() > 0:
            
            logits_output, can_run_cuda_graph = model_runner.forward(
                forward_batch, pp_proxy_tensors=None
            )
            # Here, the forward_batch full logits contains all the blocks 
            # such as [diffusion_block_size * batch_size, hidden_size]
            # Reshape logits to (batch_size, block_size, vocab_size)
            logits = logits_output.full_logits.view(batch_size, self.block_size, -1)
            self.decoder.decode(logits, 0, self.block_size, block)
            
            # Update step_maps: track which tokens were decoded in this iteration
            block_mask_index_next = (block == self.mask_id)
            if forward_batch.return_step_maps:
                # Find tokens that changed from mask to non-mask
                changed_mask = block_mask_index & (~block_mask_index_next)  # [batch_size, block_size]
                
                # Update diffusion_steps for batches that have changes
                changed_batch_index = changed_mask.sum(dim=1) > 0  # [batch_size]
                forward_batch.diffusion_steps[changed_batch_index] += 1
                
                # Update step_maps: assign step number to changed positions
                # Broadcast diffusion_steps to match changed positions
                step_values = forward_batch.diffusion_steps.unsqueeze(1)  # [batch_size, 1]
                step_maps[changed_mask] = step_values.expand_as(block)[changed_mask]
            
            block_mask_index = block_mask_index_next
        
        logits_output, can_run_cuda_graph = model_runner.forward(
            forward_batch, pp_proxy_tensors=None
        )

        # Extract only the newly generated tokens for each batch
        next_token_ids = torch.reshape(forward_batch.input_ids, (batch_size, -1))
        new_next_token_ids = [
            next_token_ids[i, start_list[i]:] for i in range(batch_size)
        ]

        if forward_batch.return_step_maps:
            logits_output.diffusion_steps = forward_batch.diffusion_steps
            new_step_maps = [step_maps[i, start_list[i]:] for i in range(batch_size)]
            logits_output.step_maps = new_step_maps

        return logits_output, new_next_token_ids, can_run_cuda_graph
