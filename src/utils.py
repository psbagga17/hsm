# -*- coding: utf-8 -*-
"""Utility functions for Hierarchical Softmax Language Generation."""

import heapq
import time
from functools import wraps

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling, GPT2Tokenizer


def execution_timer(func):
    """Decorator to measure and print execution time of a function."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Function {func.__name__} executed in {elapsed_time:.4f} seconds")
        return result

    return wrapper


class Node:
    """A node in the Huffman tree for hierarchical softmax."""

    def __init__(self, symbol=None, frequency=None):
        self.symbol = symbol
        self.frequency = frequency
        self.left = None
        self.right = None
        self.idx = None  # Index for internal nodes (set during HSM init)

    def __lt__(self, other):
        return self.frequency < other.frequency

    def is_leaf(self):
        return self.left is None and self.right is None


def build_tree(leaf_nodes, frequencies):
    """
    Build a Huffman tree from leaf nodes and their frequencies.

    Args:
        leaf_nodes: List of token IDs (vocabulary values)
        frequencies: List of frequencies for each token

    Returns:
        Root node of the Huffman tree
    """
    # Create a priority queue of nodes
    priority_queue = [Node(val, freq) for val, freq in zip(leaf_nodes, frequencies)]
    heapq.heapify(priority_queue)

    internal_node_counter = 0
    # Build the Huffman tree
    while len(priority_queue) > 1:
        left_child = heapq.heappop(priority_queue)
        right_child = heapq.heappop(priority_queue)
        merged_node = Node(
            symbol=f"Internal Node {internal_node_counter}",
            frequency=left_child.frequency + right_child.frequency,
        )
        merged_node.left = left_child
        merged_node.right = right_child
        heapq.heappush(priority_queue, merged_node)
        internal_node_counter += 1
    return priority_queue[0]


def generate_paths(node, code, path_dict):
    """
    Generate binary paths from root to each leaf node.

    Args:
        node: Current node in traversal
        code: Current binary path (list of 0s and 1s)
        path_dict: Dictionary to store token_id -> path mapping

    Returns:
        Dictionary mapping token IDs to their binary paths
    """
    if node is not None:
        if node.symbol is not None and not isinstance(node.symbol, str):
            path_dict[node.symbol] = code
        generate_paths(node.left, code + [0], path_dict)
        generate_paths(node.right, code + [1], path_dict)
    return path_dict


def max_depth(node):
    """Calculate the maximum depth of a tree."""
    if node is None:
        return 0
    left_depth = max_depth(node.left)
    right_depth = max_depth(node.right)
    return max(left_depth, right_depth) + 1


def get_tokenizer():
    """Initialize and return a GPT-2 tokenizer."""
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_wikitext_data(
    tokenizer,
    max_seq_len: int = 128,
    batch_size: int = 8,
    shuffle: bool = True,
    train_samples: int = 10000,
    val_samples: int = 500,
    test_samples: int = 100,
    num_workers: int = 8,
):
    """
    Load and preprocess WikiText-2 dataset.

    Args:
        tokenizer: HuggingFace tokenizer
        max_seq_len: Maximum sequence length for chunking
        batch_size: Batch size for data loaders
        shuffle: Whether to shuffle data
        train_samples: Number of training samples to use
        val_samples: Number of validation samples to use
        test_samples: Number of test samples to use
        num_workers: Number of workers for data loading

    Returns:
        Tuple of (train_dataloader, val_dataloader, test_dataloader, lm_datasets)
    """
    # Load dataset
    datasets = load_dataset("wikitext", "wikitext-2-raw-v1")

    # Subset the data
    datasets["train"] = datasets["train"].select(
        range(min(train_samples, len(datasets["train"])))
    )
    datasets["validation"] = datasets["validation"].select(
        range(min(val_samples, len(datasets["validation"])))
    )
    datasets["test"] = datasets["test"].select(
        range(min(test_samples, len(datasets["test"])))
    )

    def tokenize(examples):
        return tokenizer(examples["text"])

    tokenized_datasets = datasets.map(
        tokenize, batched=True, num_proc=4, remove_columns=["text"]
    )

    def group_texts(examples, block_size=max_seq_len):
        """Concatenate all texts and split into fixed-length chunks."""
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # Drop the small remainder
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=4,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False  # False for Causal Language Modeling (CLM)
    )

    # pin_memory only works with CUDA, not MPS
    use_pin_memory = torch.cuda.is_available()

    train_dataloader = DataLoader(
        lm_datasets["train"],
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        collate_fn=data_collator,
    )

    val_dataloader = DataLoader(
        lm_datasets["validation"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        collate_fn=data_collator,
    )

    test_dataloader = DataLoader(
        lm_datasets["test"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        collate_fn=data_collator,
    )

    print(f"Total number of training sequences: {len(lm_datasets['train'])}")
    print(f"Number of batches per epoch: {len(train_dataloader)}")

    return train_dataloader, val_dataloader, test_dataloader, lm_datasets


def inspect_batch(dataloader, tokenizer, name=""):
    """Print information about a batch from the dataloader."""
    print(f"\n--- {name} Batch Inspection ---")
    for batch in dataloader:
        print(f"Input IDs shape: {batch['input_ids'].shape}")
        print(f"Labels shape: {batch['labels'].shape}")
        print(f"Attention Mask shape: {batch['attention_mask'].shape}")
        print("\nFirst sequence (decoded):")
        print(tokenizer.decode(batch["input_ids"][0], skip_special_tokens=True))
        break
