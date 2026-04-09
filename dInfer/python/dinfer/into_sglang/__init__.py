"""
dInfer integration module for SGLang.

This module provides the integration layer between dInfer's parallel decoding
algorithms and SGLang's diffusion framework.
"""

from dinfer.into_sglang.algorithm import dInferDiffusionAlgorithm

__all__ = ["dInferDiffusionAlgorithm"]
