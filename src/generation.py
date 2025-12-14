import heapq

import torch
import torch.nn.functional as F

from .utils import execution_timer


def hsm_greedy_predict(model, input_ids, hidden_state, temperature=1.0, **kwargs):
    """Greedy decoding: always pick the most likely branch."""
    curr = model.hsm_head.root
    while not curr.is_leaf():
        logit = model.hsm_head.node_weights[curr.idx](hidden_state)
        choice = F.sigmoid(logit / temperature) > 0.5
        curr = curr.left if not choice else curr.right
    return curr.symbol


def hsm_top_k_generate(
    model, hidden_state, k=5, max_iterations=1000, temperature=1.0, **kwargs
):
    """Generate top-k candidates using beam search on the HSM tree."""
    beam = [(0.0, model.hsm_head.root)]
    final_candidates = []
    iterations = 0

    while beam and len(final_candidates) < k and iterations < max_iterations:
        iterations += 1
        neg_log_prob, curr_node = heapq.heappop(beam)
        neg_log_prob_tensor = torch.tensor(neg_log_prob)

        if curr_node.is_leaf():
            probability = torch.exp(-neg_log_prob_tensor).item()
            final_candidates.append((curr_node.symbol, probability))
            continue

        logit = model.hsm_head.node_weights[curr_node.idx](hidden_state)
        p_right = F.sigmoid(logit / temperature)
        p_left = 1.0 - p_right
        log_p_right = torch.log(p_right + 1e-10)
        log_p_left = torch.log(p_left + 1e-10)

        if curr_node.left:
            new_neg_log_prob_left = neg_log_prob - log_p_left.item()
            heapq.heappush(beam, (new_neg_log_prob_left, curr_node.left))
        if curr_node.right:
            new_neg_log_prob_right = neg_log_prob - log_p_right.item()
            heapq.heappush(beam, (new_neg_log_prob_right, curr_node.right))

    final_candidates.sort(key=lambda x: x[1], reverse=True)
    return final_candidates


def hsm_top_k_predict(
    model, input_ids, hidden_state, k=5, max_iterations=1000, **kwargs
):
    """Top-k sampling: sample from top k candidates."""
    final_candidates = hsm_top_k_generate(
        model, hidden_state, k, max_iterations, **kwargs
    )

    if not final_candidates:
        return 0

    raw_probabilities = torch.tensor([item[1] for item in final_candidates])
    sum_probabilities = torch.sum(raw_probabilities)

    if sum_probabilities.item() == 0:
        num_candidates = len(raw_probabilities)
        distribution = (torch.ones(num_candidates) / num_candidates).squeeze()
    else:
        distribution = (raw_probabilities / sum_probabilities).squeeze()

    # Handle 0-d tensor case
    if distribution.dim() == 0:
        distribution = distribution.unsqueeze(0)

    sampled_index_tensor = torch.multinomial(
        distribution, num_samples=1, replacement=False
    )
    return final_candidates[sampled_index_tensor.item()][0]


def hsm_top_p_predict(
    model,
    input_ids,
    hidden_state,
    p=0.6,
    max_iterations=1000,
    temperature=1.0,
    **kwargs,
):
    """Top-p (nucleus) sampling: sample from smallest set with cumulative prob >= p."""
    beam = [(0.0, model.hsm_head.root)]
    curr_p = 0.0
    final_candidates = []
    iterations = 0

    while beam and curr_p < p and iterations < max_iterations:
        iterations += 1
        neg_log_prob, curr_node = heapq.heappop(beam)
        neg_log_prob_tensor = torch.tensor(neg_log_prob)

        if curr_node.is_leaf():
            probability = torch.exp(-neg_log_prob_tensor).item()
            curr_p += probability
            final_candidates.append((curr_node.symbol, probability))
            continue

        logit = model.hsm_head.node_weights[curr_node.idx](hidden_state)
        p_right = F.sigmoid(logit / temperature)
        p_left = 1.0 - p_right
        log_p_right = torch.log(p_right + 1e-10)
        log_p_left = torch.log(p_left + 1e-10)

        if curr_node.left:
            new_neg_log_prob_left = neg_log_prob - log_p_left.item()
            heapq.heappush(beam, (new_neg_log_prob_left, curr_node.left))
        if curr_node.right:
            new_neg_log_prob_right = neg_log_prob - log_p_right.item()
            heapq.heappush(beam, (new_neg_log_prob_right, curr_node.right))

    if not final_candidates:
        return 0

    final_candidates.sort(key=lambda x: x[1], reverse=True)
    raw_probabilities = torch.tensor([item[1] for item in final_candidates])
    sum_probabilities = torch.sum(raw_probabilities)

    if sum_probabilities.item() == 0:
        num_candidates = len(raw_probabilities)
        distribution = (torch.ones(num_candidates) / num_candidates).squeeze()
    else:
        distribution = (raw_probabilities / sum_probabilities).squeeze()

    if distribution.dim() == 0:
        distribution = distribution.unsqueeze(0)

    sampled_index_tensor = torch.multinomial(
        distribution, num_samples=1, replacement=False
    )
    return final_candidates[sampled_index_tensor.item()][0]


def hsm_beam_predict(
    model,
    input_ids,
    hidden_state,
    num_beams=5,
    max_iterations=10000,
    beam_depth=2,
    **kwargs,
):
    inputs = input_ids.flatten().tolist()
    beam_results = [(inputs, hidden_state, 0.0) for _ in range(num_beams)]
    i = 0
    iterations = 0

    while i < beam_depth and iterations < max_iterations:
        candidates = []
        for prev_seq, prev_hidden_state, prev_probability in beam_results:
            if prev_seq and prev_seq[-1] == model.hsm_head.tokenizer.eos_token_id:
                continue
            if len(prev_seq) > len(inputs) + beam_depth:
                continue
            next_token_candidates = hsm_top_k_generate(
                model, prev_hidden_state, num_beams, max_iterations, **kwargs
            )
            for next_token_candidate, probability in next_token_candidates:
                updated_seq = prev_seq + [next_token_candidate]
                updated_hidden_state = model.transformer(
                    torch.tensor([updated_seq], device=input_ids.device)
                ).last_hidden_state[0, -1, :]
                candidates.append(
                    (
                        updated_seq,
                        updated_hidden_state,
                        prev_probability + torch.log(torch.tensor(probability + 1e-10)),
                    )
                )
                iterations += 1
        candidates.sort(key=lambda x: x[2], reverse=True)
        beam_results = candidates[:num_beams]
        i += 1

    if not beam_results:
        return []

    beam_results.sort(key=lambda x: x[2], reverse=True)
    raw_probabilities = torch.exp(
        torch.tensor([item[2] for item in beam_results], dtype=torch.float32)
    )
    sum_probabilities = torch.sum(raw_probabilities)

    if sum_probabilities.item() == 0:
        num_candidates = len(raw_probabilities)
        distribution = (torch.ones(num_candidates) / num_candidates).squeeze()
    else:
        distribution = (raw_probabilities / sum_probabilities).squeeze()

    if distribution.dim() == 0:
        distribution = distribution.unsqueeze(0)

    sampled_index_tensor = torch.multinomial(
        distribution, num_samples=1, replacement=False
    )
    return beam_results[sampled_index_tensor.item()][0][len(inputs) :]


@torch.no_grad()
@execution_timer
def hsm_generate(
    model, input_ids, generation_config=None, generate_method=None, **kwargs
):
    model.eval()
    generation_config = generation_config or model.generation_config
    cur_length = input_ids.shape[1]
    max_length = (
        generation_config.max_length or cur_length + generation_config.max_new_tokens
    )

    while cur_length < max_length:
        outputs = model.transformer(input_ids)
        hidden_state = outputs.last_hidden_state[0, -1, :]
        next_tokens = generate_method(
            input_ids=input_ids, hidden_state=hidden_state, **kwargs
        )

        if not isinstance(next_tokens, list):
            next_tokens = [next_tokens]
        if next_tokens[-1] == model.hsm_head.tokenizer.eos_token_id:
            break

        next_token_tensor = torch.tensor([next_tokens], device=input_ids.device)
        input_ids = torch.cat([input_ids, next_token_tensor], dim=-1)
        cur_length += 1

    return input_ids
