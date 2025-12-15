#!/usr/bin/env python3
"""Quick profiling script to identify HSM bottlenecks."""

import time
import torch

from src.model import GPT2WithHSM
from src.utils import build_tree, get_tokenizer, load_wikitext_data


def profile_forward():
    print("=" * 60)
    print("HSM Forward Pass Profiling")
    print("=" * 60)

    # Setup
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = get_tokenizer()
    vocab = tokenizer.get_vocab()
    print(f"Vocab size: {len(vocab)}")

    # Build tree
    t0 = time.perf_counter()
    root = build_tree(list(vocab.values()), [1] * len(vocab))
    print(f"Tree build time: {time.perf_counter() - t0:.2f}s")

    # Create model with PRF
    print("\n--- Testing PRF=True ---")
    model_prf = GPT2WithHSM(
        root=root,
        tokenizer=tokenizer,
        hidden_size=768,
        pretrained=True,
        freeze_transformer=True,
        use_prf=True,
        num_random_features=8,
    )
    model_prf.to(device)
    model_prf.eval()

    # Create model without PRF
    print("\n--- Testing PRF=False ---")
    model_std = GPT2WithHSM(
        root=root,
        tokenizer=tokenizer,
        hidden_size=768,
        pretrained=True,
        freeze_transformer=True,
        use_prf=False,
    )
    model_std.to(device)
    model_std.eval()

    # Load a small batch
    train_loader, _, _, _ = load_wikitext_data(
        tokenizer,
        max_seq_len=128,
        batch_size=8,  # Small batch for profiling
        train_samples=100,
        val_samples=10,
    )

    # Get one batch
    batch = next(iter(train_loader))
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)

    print(f"\nBatch shape: {input_ids.shape}")
    print(f"Num tokens: {input_ids.shape[0] * (input_ids.shape[1] - 1)}")

    # Profile PRF model
    print("\n" + "=" * 60)
    print("PRF=True Forward Pass (3 runs)")
    print("=" * 60)

    # Enable debug timing on hsm_head
    with torch.no_grad():
        for i in range(3):
            print(f"\n--- Run {i+1} ---")
            outputs = model_prf.transformer(input_ids)
            hidden = outputs.last_hidden_state[:, :-1, :].contiguous()
            labels_shifted = labels[:, 1:].contiguous()

            # Call with debug_timing=True
            loss = model_prf.hsm_head(hidden, labels_shifted, debug_timing=True)
            print(f"Loss: {loss.item():.4f}")

    # Profile standard model
    print("\n" + "=" * 60)
    print("PRF=False Forward Pass (3 runs)")
    print("=" * 60)

    with torch.no_grad():
        for i in range(3):
            print(f"\n--- Run {i+1} ---")
            outputs = model_std.transformer(input_ids)
            hidden = outputs.last_hidden_state[:, :-1, :].contiguous()
            labels_shifted = labels[:, 1:].contiguous()

            loss = model_std.hsm_head(hidden, labels_shifted, debug_timing=True)
            print(f"Loss: {loss.item():.4f}")


if __name__ == "__main__":
    profile_forward()
