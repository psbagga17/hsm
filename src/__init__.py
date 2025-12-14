# -*- coding: utf-8 -*-
"""HSM - GPT-2 with Hierarchical Softmax for efficient language generation."""

from .generation import (
    hsm_beam_predict,
    hsm_generate,
    hsm_greedy_predict,
    hsm_top_k_generate,
    hsm_top_k_predict,
    hsm_top_p_predict,
)
from .modules import (
    GPT2HierarchicalSoftmaxModel,
    GPT2RegularSoftmaxModel,
    HierarchicalSoftmaxHead,
)
from .utils import (
    Node,
    build_tree,
    execution_timer,
    generate_paths,
    get_tokenizer,
    inspect_batch,
    load_wikitext_data,
    max_depth,
)

__all__ = [
    # Models
    "GPT2HierarchicalSoftmaxModel",
    "GPT2RegularSoftmaxModel",
    "HierarchicalSoftmaxHead",
    # HSM generation
    "hsm_beam_predict",
    "hsm_generate",
    "hsm_greedy_predict",
    "hsm_top_k_generate",
    "hsm_top_k_predict",
    "hsm_top_p_predict",
    # Utils
    "Node",
    "build_tree",
    "execution_timer",
    "generate_paths",
    "get_tokenizer",
    "inspect_batch",
    "load_wikitext_data",
    "max_depth",
]
