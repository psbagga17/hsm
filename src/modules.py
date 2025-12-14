# -*- coding: utf-8 -*-
"""Model modules for Hierarchical Softmax Language Generation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, GenerationConfig

from .utils import execution_timer, generate_paths


# vectorized hierarchical softmax head
class HierarchicalSoftmaxHead(nn.Module):

    def __init__(self, root, tokenizer, hidden_size):
        super().__init__()
        self.root = root
        self.tokenizer = tokenizer
        self.hidden_size = hidden_size
        print(f"Using hidden size: {self.hidden_size}")
        self.vocab_size = len(tokenizer.get_vocab())

        self.paths = generate_paths(root, [], {})

        self.node_name_map = {}
        self.node_weights = nn.ModuleDict()
        self.param_counter = 0

        def initialize_node_parameters(node):
            if node is None or node.is_leaf():
                return None
            node_str = str(self.param_counter)
            self.node_name_map[node] = node_str
            node.idx = self.param_counter
            self.node_weights[node_str] = nn.Linear(self.hidden_size, 1, bias=False)
            self.param_counter += 1
            initialize_node_parameters(node.left)
            initialize_node_parameters(node.right)

        initialize_node_parameters(root)
        self.num_internal_nodes = self.param_counter

        print(
            f"HSM: hidden_size={hidden_size}, internal_nodes={self.num_internal_nodes}"
        )

        self._precompute_paths()

    def _precompute_paths(self):
        """Pre-compute path tensors for efficient batch lookup."""
        # Find max path length
        max_path_len = max(len(p) for p in self.paths.values()) if self.paths else 1

        path_nodes = torch.full((self.vocab_size, max_path_len), -1, dtype=torch.long)
        path_targets = torch.zeros((self.vocab_size, max_path_len), dtype=torch.float)
        path_masks = torch.zeros((self.vocab_size, max_path_len), dtype=torch.float)

        for token_id, choices in self.paths.items():
            if token_id >= self.vocab_size:
                continue
            curr = self.root
            for step, choice in enumerate(choices):
                if curr.is_leaf():
                    break
                if curr.idx is not None:
                    path_nodes[token_id, step] = curr.idx
                    path_targets[token_id, step] = float(choice)
                    path_masks[token_id, step] = 1.0
                curr = curr.left if choice == 0 else curr.right

        self.register_buffer("path_nodes", path_nodes)
        self.register_buffer("path_targets", path_targets)
        self.register_buffer("path_masks", path_masks)

        print(
            f"HSM: max_path_length={max_path_len}, paths_precomputed={len(self.paths)}"
        )

    def forward(self, hidden_state, target_ids):
        batch_size, seq_len, hidden_size = (
            hidden_state.shape
        )  # puneet double check this, sometimes its seq_len, bsz, hidden
        device = hidden_state.device

        h_flat = hidden_state.view(-1, hidden_size)  # [N, hidden_size]
        targets_flat = target_ids.view(-1)  # [N]

        num_tokens = h_flat.shape[0]

        targets_clamped = targets_flat.clamp(0, self.vocab_size - 1)

        # path info for tokens [num_tokens (N), max_path_len (L)]
        token_path_nodes = self.path_nodes[targets_clamped]  # [N, L]
        token_path_targets = self.path_targets[targets_clamped]  # [N, L]
        token_path_masks = self.path_masks[targets_clamped]  # [N, L]

        # compute logits for all nodes at once
        # num_nodes, hidden_size
        all_weights = torch.cat(
            [self.node_weights[str(i)].weight for i in range(self.num_internal_nodes)],
            dim=0,
        )  # [num_nodes, hidden_size]
        all_logits = h_flat @ all_weights.T  # N, num_nodes
        # print(f"all_logits: {all_logits.shape}")
        # print(f"all_logits: {all_logits}")

        # gather the logits for the specific path nodes that each token would have taken
        valid_node_indices = token_path_nodes.clamp(min=0)  # N, L
        # print(f"valid_node_indices: {valid_node_indices.shape}")
        # print(f"valid_node_indices: {valid_node_indices}")
        path_logits = torch.gather(all_logits, dim=1, index=valid_node_indices)

        # print(f"path_logits: {path_logits.shape}")
        # print(f"path_logits: {path_logits}")

        bce_losses = F.binary_cross_entropy_with_logits(
            path_logits, token_path_targets, reduction="none"
        )

        masked_losses = bce_losses * token_path_masks  # N, L

        total_loss = masked_losses.sum()
        num_valid_steps = token_path_masks.sum()

        if num_valid_steps > 0:
            return total_loss / num_valid_steps
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)


# GPT-2 with Hierarchical Softmax


class GPT2HierarchicalSoftmaxModel(nn.Module):
    """
    GPT-2 model with Hierarchical Softmax output head.

    Uses pre-trained GPT-2 weights and replaces the standard softmax head
    with a Hierarchical Softmax head. Both the transformer backbone and
    the HSM head are fine-tuned together.
    """

    def __init__(self, root, tokenizer, hidden_size, model_name="gpt2"):
        super().__init__()
        # Load pre-trained GPT-2 weights
        self.transformer = GPT2Model.from_pretrained(model_name)
        self.generation_config = GenerationConfig()
        self.hsm_head = HierarchicalSoftmaxHead(root, tokenizer, hidden_size)
        # Note: transformer weights are NOT frozen - full fine-tuning

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.last_hidden_state

        if labels is not None:
            shifted_hidden_states = hidden_states[:, :-1, :].contiguous()
            shifted_labels = labels[:, 1:].contiguous()
            loss = self.hsm_head(shifted_hidden_states, shifted_labels)
            return {"loss": loss, "hidden_states": shifted_hidden_states}

        return {"hidden_states": hidden_states}


# normal


def ce_loss(logits, labels):
    """Numerically stable cross-entropy loss."""
    C = logits.max().item()
    shifted_logits = logits - C
    exp_shifted_logits = torch.exp(shifted_logits)
    sum_exp_shifted_logits = torch.sum(exp_shifted_logits, dim=1, keepdim=True)
    log_probabilities = shifted_logits - torch.log(sum_exp_shifted_logits)
    batch_indices = torch.arange(labels.shape[0], device=labels.device)
    log_p_true_class = log_probabilities[batch_indices, labels]
    mean_loss = -torch.mean(log_p_true_class)
    return mean_loss


class GPT2RegularSoftmaxModel(nn.Module):
    """
    Standard GPT-2 model with regular softmax output head.
    Used as a baseline for comparison with HSM model.

    NOTE: The transformer is NOT frozen here, allowing full fine-tuning
    of both the backbone and the output head. This is intentional for
    comparison - it shows traditional fine-tuning vs. frozen backbone + HSM.
    """

    def __init__(self, config, vocab_size, freeze_transformer=False):
        super().__init__()
        self.transformer = GPT2Model(config)
        self.generation_config = GenerationConfig()
        self.lm_head = nn.Linear(config.n_embd, vocab_size, bias=False)
        self.vocab_size = vocab_size

        # Optionally freeze transformer (default: False for full fine-tuning)
        if freeze_transformer:
            for param in self.transformer.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.last_hidden_state

        if labels is not None:
            shifted_hidden_states = hidden_states[:, :-1, :].contiguous()
            shifted_labels = labels[:, 1:].contiguous()
            logits = self.lm_head(shifted_hidden_states)
            logits = logits.view(-1, self.vocab_size)
            labels = shifted_labels.view(-1)

            max_label = labels.max().item()
            if max_label >= self.vocab_size:
                raise ValueError(
                    f"Label ID ({max_label}) is >= vocab_size ({self.vocab_size})!"
                )

            loss = ce_loss(logits, labels)
            return {
                "loss": loss,
                "logits": logits,
                "hidden_states": shifted_hidden_states,
            }

        return {"hidden_states": hidden_states}

    def _greedy_predict(self, hidden_state):
        h_i = hidden_state.unsqueeze(0)
        logits = self.lm_head(h_i)
        next_token = torch.argmax(logits, dim=-1).item()
        return next_token

    @torch.no_grad()
    @execution_timer
    def generate(self, input_ids, generation_config=None, tokenizer=None, **kwargs):
        """Generate text using greedy decoding."""
        self.eval()
        if tokenizer is None:
            raise ValueError("Tokenizer must be provided for generation.")

        eos_token_id = tokenizer.eos_token_id
        generation_config = generation_config or self.generation_config
        cur_length = input_ids.shape[1]
        max_length = (
            generation_config.max_length
            or cur_length + generation_config.max_new_tokens
        )

        if input_ids.shape[0] != 1:
            raise ValueError(
                "This generation implementation supports only batch size 1."
            )

        while cur_length < max_length:
            outputs = self.transformer(input_ids)
            hidden_state = outputs.last_hidden_state[0, -1, :]
            next_token = self._greedy_predict(hidden_state)
            if next_token == eos_token_id:
                break
            next_token_tensor = torch.tensor([[next_token]], device=input_ids.device)
            input_ids = torch.cat([input_ids, next_token_tensor], dim=-1)
            cur_length += 1

        return input_ids


if __name__ == "__main__":
    # Test forward pass of HSM head
    from src.utils import build_tree

    print("=" * 50)
    print("Testing HierarchicalSoftmaxHead forward pass")
    print("=" * 50)

    # Create a simple mock tokenizer class
    class MockTokenizer:
        def __init__(self, vocab_size):
            self._vocab = {f"token_{i}": i for i in range(vocab_size)}

        def get_vocab(self):
            return self._vocab

    # Test parameters
    vocab_size = 16
    hidden_size = 32
    batch_size = 2
    seq_len = 5

    # Build Huffman tree with uniform frequencies
    token_ids = list(range(vocab_size))
    frequencies = [1] * vocab_size
    root = build_tree(token_ids, frequencies)
    print(f"Built Huffman tree with {vocab_size} leaves")

    # Create tokenizer and HSM head
    tokenizer = MockTokenizer(vocab_size)
    hsm_head = HierarchicalSoftmaxHead(root, tokenizer, hidden_size)
    print(f"Created HSM head with {hsm_head.num_internal_nodes} internal nodes")

    # Create dummy inputs
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    target_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    print(f"\nInput shapes:")
    print(f"  hidden_states: {hidden_states.shape}")
    print(f"  target_ids: {target_ids.shape}")

    # Forward pass
    loss = hsm_head(hidden_states, target_ids)

    print(f"\nForward pass successful!")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Loss requires_grad: {loss.requires_grad}")

    # Test backward pass
    loss.backward()
    print(f"  Backward pass successful!")

    # Check gradients exist
    grad_count = sum(1 for p in hsm_head.parameters() if p.grad is not None)
    total_params = sum(1 for _ in hsm_head.parameters())
    print(f"  Gradients computed for {grad_count}/{total_params} parameters")

    print("\n" + "=" * 50)
    print("All tests passed!")
    print("=" * 50)
