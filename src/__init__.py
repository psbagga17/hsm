# -*- coding: utf-8 -*-
"""
Hierarchical Softmax (HSM) for Language Models.

Modules:
- hsm: HierarchicalSoftmaxHead with PRF toggle
- prf: FAVOR+ PRF implementation
- inference: O(log V) generation methods
- train: Training loop
- eval: Evaluation and metrics
- model: GPT-2 model wrappers
- utils: Data loading and tree building
"""

from .hsm import HierarchicalSoftmaxHead
from .prf import (
    PRFSigmoid,
    prf_sigmoid_batched,
    prf_sigmoid_single,
    standard_sigmoid,
    orthogonal_random_features,
    positive_random_features,
)
from .inference import generate, greedy_decode_token, top_k_sample, top_p_sample
from .model import GPT2WithHSM, GPT2WithSoftmax, create_model
from .train import train, train_epoch, validate, TrainingMetrics
from .eval import ExperimentConfig, EvaluationResults, evaluate_loss
from .utils import Node, build_tree, generate_paths, get_tokenizer, load_wikitext_data

__all__ = [
    # HSM
    "HierarchicalSoftmaxHead",
    # PRF
    "PRFSigmoid",
    "prf_sigmoid_batched",
    "prf_sigmoid_single",
    "standard_sigmoid",
    "orthogonal_random_features",
    "positive_random_features",
    # Inference
    "generate",
    "greedy_decode_token",
    "top_k_sample",
    "top_p_sample",
    # Models
    "GPT2WithHSM",
    "GPT2WithSoftmax",
    "create_model",
    # Training
    "train",
    "train_epoch",
    "validate",
    "TrainingMetrics",
    # Evaluation
    "ExperimentConfig",
    "EvaluationResults",
    "evaluate_loss",
    # Utils
    "Node",
    "build_tree",
    "generate_paths",
    "get_tokenizer",
    "load_wikitext_data",
]
