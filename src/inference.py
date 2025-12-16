"""O(log V) inference methods for Hierarchical Softmax."""

import heapq
from typing import List, Tuple

import torch


def greedy_decode_token(
    hsm_head,
    hidden_state: torch.Tensor,
    temperature: float = 1.0,
) -> int:
    """
    Greedy decode: traverse tree taking more probable branch at each node.
    O(log V) - visits at most log(V) nodes.
    """
    curr = hsm_head.root

    while not curr.is_leaf():
        p_right = hsm_head.compute_node_probability(hidden_state, curr.idx, temperature)
        curr = curr.right if p_right > 0.5 else curr.left

    return curr.symbol


def top_k_candidates(
    hsm_head,
    hidden_state: torch.Tensor,
    k: int = 5,
    temperature: float = 1.0,
    max_iterations: int = 10000,
) -> List[Tuple[int, float]]:
    """
    Find top-k tokens using best-first search with min-heap.
    O(k log V) - explores at most k complete paths.

    Returns: [(token_id, probability), ...] sorted by probability desc.
    """
    device = hidden_state.device
    heap = [(0.0, hsm_head.root)]  # (neg_log_prob, node)
    candidates = []
    iterations = 0

    while heap and len(candidates) < k and iterations < max_iterations:
        iterations += 1
        neg_log_prob, curr_node = heapq.heappop(heap)

        if curr_node.is_leaf():
            prob = torch.exp(torch.tensor(-neg_log_prob, device=device)).item()
            candidates.append((curr_node.symbol, prob))
            continue

        p_right = hsm_head.compute_node_probability(hidden_state, curr_node.idx, temperature)
        p_left = 1.0 - p_right

        log_p_right = torch.log(p_right + 1e-10).item()
        log_p_left = torch.log(p_left + 1e-10).item()

        if curr_node.left is not None:
            heapq.heappush(heap, (neg_log_prob - log_p_left, curr_node.left))
        if curr_node.right is not None:
            heapq.heappush(heap, (neg_log_prob - log_p_right, curr_node.right))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates


def top_k_sample(
    hsm_head,
    hidden_state: torch.Tensor,
    k: int = 5,
    temperature: float = 1.0,
) -> int:
    """Sample from top-k candidates."""
    candidates = top_k_candidates(hsm_head, hidden_state, k, temperature)

    if not candidates:
        return 0

    probs = torch.tensor([p for _, p in candidates])
    probs = probs / probs.sum()

    if probs.dim() == 0:
        probs = probs.unsqueeze(0)

    idx = torch.multinomial(probs, num_samples=1).item()
    return candidates[idx][0]


def top_p_candidates(
    hsm_head,
    hidden_state: torch.Tensor,
    p: float = 0.9,
    temperature: float = 1.0,
    max_iterations: int = 10000,
) -> List[Tuple[int, float]]:
    """
    Find smallest set of tokens with cumulative probability >= p.
    Uses best-first search.
    """
    device = hidden_state.device
    heap = [(0.0, hsm_head.root)]
    candidates = []
    cumulative_prob = 0.0
    iterations = 0

    while heap and cumulative_prob < p and iterations < max_iterations:
        iterations += 1
        neg_log_prob, curr_node = heapq.heappop(heap)

        if curr_node.is_leaf():
            prob = torch.exp(torch.tensor(-neg_log_prob, device=device)).item()
            cumulative_prob += prob
            candidates.append((curr_node.symbol, prob))
            continue

        p_right = hsm_head.compute_node_probability(hidden_state, curr_node.idx, temperature)
        p_left = 1.0 - p_right

        log_p_right = torch.log(p_right + 1e-10).item()
        log_p_left = torch.log(p_left + 1e-10).item()

        if curr_node.left is not None:
            heapq.heappush(heap, (neg_log_prob - log_p_left, curr_node.left))
        if curr_node.right is not None:
            heapq.heappush(heap, (neg_log_prob - log_p_right, curr_node.right))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates


def top_p_sample(
    hsm_head,
    hidden_state: torch.Tensor,
    p: float = 0.9,
    temperature: float = 1.0,
) -> int:
    """Sample from nucleus (top-p) candidates."""
    candidates = top_p_candidates(hsm_head, hidden_state, p, temperature)

    if not candidates:
        return 0

    probs = torch.tensor([prob for _, prob in candidates])
    probs = probs / probs.sum()

    if probs.dim() == 0:
        probs = probs.unsqueeze(0)

    idx = torch.multinomial(probs, num_samples=1).item()
    return candidates[idx][0]


def beam_search(
    model,
    input_ids: torch.Tensor,
    num_beams: int = 5,
    max_new_tokens: int = 10,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Beam search generation.

    Args:
        model: full model with transformer and hsm_head
        input_ids: (1, s)
        num_beams: number of beams
        max_new_tokens: max tokens to generate
        temperature: sigmoid temperature

    Returns: (1, s + generated)
    """
    device = input_ids.device
    hsm_head = model.hsm_head
    eos_token_id = hsm_head.tokenizer.eos_token_id

    initial_seq = input_ids[0].tolist()
    beams = [(initial_seq, 0.0)]  # (sequence, cumulative_log_prob)

    for _ in range(max_new_tokens):
        all_candidates = []

        for seq, log_prob in beams:
            if seq[-1] == eos_token_id:
                all_candidates.append((seq, log_prob))
                continue

            seq_tensor = torch.tensor([seq], device=device)
            outputs = model.transformer(seq_tensor)
            hidden_state = outputs.last_hidden_state[0, -1, :]

            candidates = top_k_candidates(hsm_head, hidden_state, k=num_beams, temperature=temperature)

            for token_id, prob in candidates:
                new_seq = seq + [token_id]
                new_log_prob = log_prob + torch.log(torch.tensor(prob + 1e-10)).item()
                all_candidates.append((new_seq, new_log_prob))

        all_candidates.sort(key=lambda x: x[1], reverse=True)
        beams = all_candidates[:num_beams]

        if all(seq[-1] == eos_token_id for seq, _ in beams):
            break

    best_seq = beams[0][0]
    return torch.tensor([best_seq], device=device)


@torch.no_grad()
def generate(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int = 20,
    method: str = "greedy",
    temperature: float = 1.0,
    top_k: int = 5,
    top_p: float = 0.9,
    num_beams: int = 5,
) -> torch.Tensor:
    """
    Generate tokens using HSM with specified decoding method.

    Args:
        model: full model with transformer and hsm_head
        input_ids: (1, s)
        max_new_tokens: max tokens to generate
        method: "greedy", "top_k", "top_p", or "beam"
        temperature: sigmoid temperature
        top_k: k for top-k sampling
        top_p: p for nucleus sampling
        num_beams: beams for beam search

    Returns: (1, s + generated)
    """
    model.eval()
    device = input_ids.device
    hsm_head = model.hsm_head
    eos_token_id = hsm_head.tokenizer.eos_token_id

    if method == "beam":
        return beam_search(model, input_ids, num_beams, max_new_tokens, temperature)

    generated = input_ids.clone()

    for _ in range(max_new_tokens):
        outputs = model.transformer(generated)
        hidden_state = outputs.last_hidden_state[0, -1, :]

        if method == "greedy":
            next_token = greedy_decode_token(hsm_head, hidden_state, temperature)
        elif method == "top_k":
            next_token = top_k_sample(hsm_head, hidden_state, top_k, temperature)
        elif method == "top_p":
            next_token = top_p_sample(hsm_head, hidden_state, top_p, temperature)
        else:
            raise ValueError(f"Unknown method: {method}")

        if next_token == eos_token_id:
            break

        next_token_tensor = torch.tensor([[next_token]], device=device)
        generated = torch.cat([generated, next_token_tensor], dim=-1)

    return generated
