"""Hierarchical Softmax Head with optional PRF sigmoid approximation."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .prf import (
    PRFSigmoid,
    standard_sigmoid,
    prf_sigmoid_single,
    positive_random_features as prf_positive_random_features,
)
from .utils import generate_paths


class HierarchicalSoftmaxHead(nn.Module):
    """
    Hierarchical Softmax output head using a binary tree structure.

    Unlike nn.Embedding which stores one vector per vocabulary token,
    HSM stores one vector per *internal node* of the tree. Each token's
    probability is computed by traversing root-to-leaf, making binary
    decisions at each node. This gives O(log v) inference complexity.

    Weights are stored as a single (num_nodes, d) tensor rather than
    separate parameters for efficiency - avoids O(v) Python loop.
    """

    def __init__(
        self,
        root,
        tokenizer,
        hidden_size: int,
        use_prf: bool = False,
        num_random_features: int = 256,
    ):
        super().__init__()
        self.root = root
        self.tokenizer = tokenizer
        self.hidden_size = hidden_size
        self.use_prf = use_prf
        self.num_random_features = num_random_features
        self.vocab_size = len(tokenizer.get_vocab())

        self.paths = generate_paths(root, [], {})

        self.node_name_map = {}
        self.param_counter = 0
        self._count_and_index_nodes(root)
        self.num_internal_nodes = self.param_counter

        # (num_nodes, d) - single tensor, not ParameterDict
        self.node_weights = nn.Parameter(
            torch.randn(self.num_internal_nodes, hidden_size) * 0.01
        )

        self._precompute_paths()

        if use_prf:
            self.prf_module = PRFSigmoid(hidden_size, num_random_features)

        print(f"HSM Head initialized:")
        print(f"  hidden_size={hidden_size}, vocab_size={self.vocab_size}")
        print(f"  internal_nodes={self.num_internal_nodes}, use_prf={use_prf}")

    def _count_and_index_nodes(self, node):
        if node is None or node.is_leaf():
            return
        node.idx = self.param_counter
        self.node_name_map[node] = str(self.param_counter)
        self.param_counter += 1
        self._count_and_index_nodes(node.left)
        self._count_and_index_nodes(node.right)

    def _precompute_paths(self):
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

        self.register_buffer("path_nodes", path_nodes)      # (v, l)
        self.register_buffer("path_targets", path_targets)  # (v, l)
        self.register_buffer("path_masks", path_masks)      # (v, l)
        self.max_path_len = max_path_len
        print(f"  max_path_length={max_path_len}")

    def compute_node_probability(
        self, hidden_state: torch.Tensor, node_idx: int, temperature: float = 1.0
    ) -> torch.Tensor:
        weight = self.node_weights[node_idx]  # (d,)

        if self.use_prf and hasattr(self, "prf_module"):
            if hidden_state.dim() == 1:
                return prf_sigmoid_single(
                    hidden_state, weight, self.prf_module.random_matrix, temperature
                )
            else:
                weight_batched = weight.unsqueeze(0)  # (1, d)
                probs = self.prf_module.forward_batched(
                    hidden_state, weight_batched, temperature
                )  # (n, 1)
                return probs.squeeze(-1)
        else:
            logit = hidden_state @ weight
            return standard_sigmoid(logit, temperature)

    def forward(self, hidden_states: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute HSM loss: sum BCE along each token's path, average over tokens.

        Args:
            hidden_states: (b, s, d)
            target_ids: (b, s)
        Returns:
            Scalar loss
        """
        b, s, d = hidden_states.shape
        device = hidden_states.device

        h_flat = hidden_states.view(-1, d)  # (n, d) where n = b * s
        targets_flat = target_ids.view(-1).clamp(0, self.vocab_size - 1)  # (n,)

        token_path_nodes = self.path_nodes[targets_flat]      # (n, l)
        token_path_targets = self.path_targets[targets_flat]  # (n, l)
        token_path_masks = self.path_masks[targets_flat]      # (n, l)

        all_weights = self.node_weights  # (num_nodes, d)
        valid_node_indices = token_path_nodes.clamp(min=0)  # (n, l)

        if self.use_prf and hasattr(self, "prf_module"):
            path_weights = all_weights[valid_node_indices]  # (n, l, d)
            path_probs = self._compute_prf_path_probs(h_flat, path_weights)  # (n, l)

            eps = 1e-7
            path_probs = torch.clamp(path_probs, min=eps, max=1 - eps)
            bce_losses = -(
                token_path_targets * torch.log(path_probs)
                + (1 - token_path_targets) * torch.log(1 - path_probs)
            )
        else:
            all_logits = h_flat @ all_weights.T  # (n, num_nodes)
            path_logits = torch.gather(all_logits, dim=1, index=valid_node_indices)  # (n, l)
            bce_losses = F.binary_cross_entropy_with_logits(
                path_logits, token_path_targets, reduction="none"
            )

        masked_losses = bce_losses * token_path_masks  # (n, l)
        token_losses = masked_losses.sum(dim=1)  # (n,)
        num_valid = (token_path_masks.sum(dim=1) > 0).sum()

        if num_valid > 0:
            return token_losses.sum() / num_valid
        return torch.tensor(0.0, device=device, requires_grad=True)

    def _compute_prf_path_probs(
        self, h_flat: torch.Tensor, path_weights: torch.Tensor
    ) -> torch.Tensor:
        """O(n * l * m) PRF computation for path nodes only."""
        n, l, d = path_weights.shape
        omega = self.prf_module.random_matrix  # (m, d)

        phi_h, h_norm_sq, h_max = prf_positive_random_features(h_flat, omega, return_norm=True)

        path_weights_flat = path_weights.view(n * l, d)
        phi_w_flat, w_norm_sq_flat, w_max_flat = prf_positive_random_features(
            path_weights_flat, omega, return_norm=True
        )
        phi_w = phi_w_flat.view(n, l, -1)        # (n, l, m)
        w_norm_sq = w_norm_sq_flat.view(n, l, 1)  # (n, l, 1)
        w_max = w_max_flat.view(n, l, 1)          # (n, l, 1)

        rf_kernel = torch.einsum('nm,nlm->nl', phi_h, phi_w)  # (n, l)

        log_correction = (
            (h_norm_sq.squeeze(-1).unsqueeze(1) + w_norm_sq.squeeze(-1)) / 2
            + h_max.squeeze(-1).unsqueeze(1) + w_max.squeeze(-1)
        )
        log_correction = torch.clamp(log_correction, min=-80, max=80)

        log_exp = torch.log(rf_kernel + 1e-10) + log_correction
        log_exp = torch.clamp(log_exp, min=-80, max=80)
        exp_approx = torch.exp(log_exp)

        return exp_approx / (1.0 + exp_approx)

    def get_token_probability(
        self, hidden_state: torch.Tensor, token_id: int, temperature: float = 1.0
    ) -> torch.Tensor:
        """Compute log probability of a specific token via tree traversal."""
        if token_id not in self.paths:
            return torch.tensor(float("-inf"), device=hidden_state.device)

        log_prob = torch.tensor(0.0, device=hidden_state.device)
        curr = self.root

        for choice in self.paths[token_id]:
            if curr.is_leaf():
                break
            p_right = self.compute_node_probability(hidden_state, curr.idx, temperature)
            if choice == 1:
                log_prob = log_prob + torch.log(p_right + 1e-10)
            else:
                log_prob = log_prob + torch.log(1 - p_right + 1e-10)
            curr = curr.left if choice == 0 else curr.right

        return log_prob
