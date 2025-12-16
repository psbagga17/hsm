"""
Main entry point for HSM experiments.

This script demonstrates:
- Loading/training a GPT-2 model with Hierarchical Softmax head
- Testing various generation methods (greedy, top-k, top-p, beam)
- Optional PRF (Positive Random Features) support

Usage:
    python -m src.main           # Run as module
    python src/main.py           # Run directly
"""

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Handle imports for both `python -m src.main` and `python src/main.py`
try:
    from .model import GPT2WithHSM
    from .inference import generate
    from .utils import (
        build_tree,
        execution_timer,
        generate_paths,
        get_tokenizer,
        inspect_batch,
        load_wikitext_data,
    )
except ImportError:
    from model import GPT2WithHSM
    from inference import generate
    from utils import (
        build_tree,
        execution_timer,
        generate_paths,
        get_tokenizer,
        inspect_batch,
        load_wikitext_data,
    )


# Paths - models saved in project root's models/ directory
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

HSM_MODEL_PATH = MODEL_DIR / "GPT2HSMModel.pt"

# Training hyperparameters
MAX_SEQ_LEN = 128
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 5e-5

# Data subset sizes (for faster iteration)
TRAIN_SAMPLES = 10000
VAL_SAMPLES = 500
TEST_SAMPLES = 100

# PRF configuration
USE_PRF = False  # Set to True to enable FAVOR+ PRF approximation
NUM_RANDOM_FEATURES = 256


def get_device():
    """Get the best available device (MPS > CUDA > CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


@execution_timer
def train_epoch(model, data_loader, optimizer, device):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    progress_bar = tqdm(data_loader, desc="Training")

    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs["loss"]
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / num_batches if num_batches > 0 else 0.0


@execution_timer
def evaluate(model, data_loader, device):
    """Evaluate the model and compute perplexity."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc="Evaluating")
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs["loss"]
            total_loss += loss.item()
            num_batches += 1
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    try:
        perplexity = math.exp(avg_loss)
    except OverflowError:
        perplexity = float("inf")

    print(f"\n--- Evaluation Complete ---")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Perplexity: {perplexity:.2f}")

    return {"avg_loss": avg_loss, "perplexity": perplexity}


def test_generation(model, tokenizer, device):
    """Test various generation methods."""
    model.eval()

    test_prompt = "The quick brown fox"
    input_ids = tokenizer.encode(test_prompt, return_tensors="pt").to(device)
    print(f"\nTest prompt: '{test_prompt}'")
    print(f"Input shape: {input_ids.shape}")

    # Test greedy generation
    print("\n--- Greedy Generation ---")
    output = generate(model, input_ids.clone(), max_new_tokens=10, method="greedy")
    print(f"Generated: {tokenizer.decode(output[0], skip_special_tokens=True)}")

    # Test with different temperatures
    print("\n--- Greedy with Temperature ---")
    for temp in [0.5, 1.0, 1.5]:
        output = generate(
            model, input_ids.clone(), max_new_tokens=10, method="greedy", temperature=temp
        )
        print(f"  temp={temp}: {tokenizer.decode(output[0], skip_special_tokens=True)}")

    # Test top-k sampling
    print("\n--- Top-k Sampling (k=5) ---")
    output = generate(model, input_ids.clone(), max_new_tokens=10, method="top_k", top_k=5)
    print(f"Generated: {tokenizer.decode(output[0], skip_special_tokens=True)}")

    # Test top-p (nucleus) sampling
    print("\n--- Top-p Sampling (p=0.9) ---")
    output = generate(model, input_ids.clone(), max_new_tokens=10, method="top_p", top_p=0.9)
    print(f"Generated: {tokenizer.decode(output[0], skip_special_tokens=True)}")

    # Test beam search
    print("\n--- Beam Search (beams=3) ---")
    output = generate(model, input_ids.clone(), max_new_tokens=10, method="beam", num_beams=3)
    print(f"Generated: {tokenizer.decode(output[0], skip_special_tokens=True)}")


def main():
    device = get_device()
    print(f"Using device: {device}")

    # Initialize tokenizer
    print("\n=== Initializing Tokenizer ===")
    tokenizer = get_tokenizer()
    vocabulary = tokenizer.get_vocab()
    print(f"Vocabulary size: {len(vocabulary)}")

    # Build Huffman tree for HSM
    print("\n=== Building Huffman Tree ===")
    root = build_tree(list(vocabulary.values()), [1] * len(vocabulary))
    paths = generate_paths(root, [], {})
    print(f"Paths for vocabulary: {len(paths)}")
    print(f"Path length for token 0: {len(paths[0])}")
    print(f"All vocab items in tree: {all(item in paths for item in vocabulary.values())}")

    # Load data
    print("\n=== Loading WikiText-2 Dataset ===")
    train_loader, val_loader, test_loader, _ = load_wikitext_data(
        tokenizer,
        max_seq_len=MAX_SEQ_LEN,
        batch_size=BATCH_SIZE,
        train_samples=TRAIN_SAMPLES,
        val_samples=VAL_SAMPLES,
        test_samples=TEST_SAMPLES,
    )

    # Inspect a batch
    inspect_batch(train_loader, tokenizer, "Training")

    # Check if we should train or load
    if HSM_MODEL_PATH.exists():
        # Load existing model
        print("\n" + "=" * 60)
        print("LOADING: HSM Model from saved checkpoint")
        print("=" * 60)

        model = GPT2WithHSM(
            root=root,
            tokenizer=tokenizer,
            hidden_size=768,
            pretrained=True,
            use_prf=USE_PRF,
            num_random_features=NUM_RANDOM_FEATURES,
        )

        # Load state dict to CPU first (avoids MPS memory alignment issues)
        state_dict = torch.load(HSM_MODEL_PATH, map_location="cpu", weights_only=True)

        # Strip "_orig_mod." prefix if model was saved after torch.compile()
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("_orig_mod."):
                new_key = key[len("_orig_mod."):]
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value

        # Try to load, handling potential mismatches from old model format
        try:
            model.load_state_dict(new_state_dict, strict=False)
            print(f"Model loaded from {HSM_MODEL_PATH}")
        except Exception as e:
            print(f"Warning: Could not load checkpoint ({e})")
            print("Starting with fresh pretrained weights")

        model.to(device)

    else:
        # Train new model
        print("\n" + "=" * 60)
        print("TRAINING: GPT-2 with Hierarchical Softmax")
        print(f"PRF enabled: {USE_PRF}")
        print("=" * 60)

        model = GPT2WithHSM(
            root=root,
            tokenizer=tokenizer,
            hidden_size=768,
            pretrained=True,
            freeze_transformer=False,
            use_prf=USE_PRF,
            num_random_features=NUM_RANDOM_FEATURES,
        )
        model.to(device)

        # Compile for faster training (if supported)
        if device.type == "mps":
            try:
                model = torch.compile(model, backend="aot_eager")
                print("Model compiled with aot_eager backend (MPS)")
            except Exception as e:
                print(f"torch.compile not available: {e}")
        elif device.type == "cuda":
            try:
                model = torch.compile(model)
                print("Model compiled with default backend")
            except Exception as e:
                print(f"torch.compile not available: {e}")

        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

        for epoch in range(EPOCHS):
            print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")
            avg_loss = train_epoch(model, train_loader, optimizer, device)
            print(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")

            # Validation
            val_results = evaluate(model, val_loader, device)

        # Save model
        torch.save(model.state_dict(), HSM_MODEL_PATH)
        print(f"\nModel saved to {HSM_MODEL_PATH}")

    # Test generation
    print("\n" + "=" * 60)
    print("TESTING: Generation Methods")
    print("=" * 60)
    test_generation(model, tokenizer, device)

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
