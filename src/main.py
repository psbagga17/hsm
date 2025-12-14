import math
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from transformers import GenerationConfig

# Handle imports for both `python -m src.main` and `python src/main.py`
try:
    from .generation import (
        hsm_beam_predict,
        hsm_generate,
        hsm_greedy_predict,
        hsm_top_k_predict,
        hsm_top_p_predict,
    )
    from .modules import GPT2HierarchicalSoftmaxModel
    from .utils import (
        build_tree,
        execution_timer,
        generate_paths,
        get_tokenizer,
        inspect_batch,
        load_wikitext_data,
    )
except ImportError:
    from generation import (
        hsm_beam_predict,
        hsm_generate,
        hsm_greedy_predict,
        hsm_top_k_predict,
        hsm_top_p_predict,
    )
    from modules import GPT2HierarchicalSoftmaxModel
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
HSM_LEARNING_RATE = 5e-5

# Data subset sizes (for faster iteration)
TRAIN_SAMPLES = 10000
VAL_SAMPLES = 500
TEST_SAMPLES = 100


@execution_timer
def train_epoch(model, data_loader, optimizer, device):
    """Train the model for one epoch."""
    model.train()
    progress_bar = tqdm(data_loader, desc="Training")

    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs["loss"]
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        progress_bar.set_postfix({"loss": loss.item()})


@execution_timer
def evaluate(model, data_loader, device):
    """Evaluate the model and compute perplexity."""
    model.eval()
    total_eval_loss = 0.0

    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc="Evaluating")
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs["loss"]
            total_eval_loss += loss.item()
            progress_bar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

    avg_loss = total_eval_loss / len(data_loader)
    try:
        perplexity = math.exp(avg_loss)
    except OverflowError:
        perplexity = float("inf")

    print(f"\n--- Evaluation Complete ---")
    print(f"Average Loss (NLL): {avg_loss:.4f}")
    print(f"Perplexity (PPL): {perplexity:.2f}")

    return {"avg_loss": avg_loss, "perplexity": perplexity}


def main():
    # Device setup - prefer MPS (Apple Silicon) > CUDA > CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
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
    print(f"Path from root to leaf of token 0: {paths[0]}")
    print(
        f"All vocab items in tree: {all(item in paths for item in vocabulary.values())}"
    )

    # Load data
    print("\n=== Loading WikiText-2 Dataset ===")
    train_dataloader, val_dataloader, test_dataloader, lm_datasets = load_wikitext_data(
        tokenizer,
        max_seq_len=MAX_SEQ_LEN,
        batch_size=BATCH_SIZE,
        train_samples=TRAIN_SAMPLES,
        val_samples=VAL_SAMPLES,
        test_samples=TEST_SAMPLES,
    )

    # Inspect a batch
    inspect_batch(train_dataloader, tokenizer, "Training")

    # train stuff
    # print("\n" + "=" * 60)
    # print("TRAINING: GPT-2 with Hierarchical Softmax (pre-trained weights)")
    # print("=" * 60)
    #
    # hsm_model = GPT2HierarchicalSoftmaxModel(
    #     root, tokenizer, hidden_size=768, model_name="gpt2"
    # )
    # hsm_model.to(device)
    #
    # # Compile model for faster execution (use aot_eager backend for MPS compatibility)
    # if device.type == "mps":
    #     try:
    #         hsm_model = torch.compile(hsm_model, backend="aot_eager")
    #         print("Model compiled with aot_eager backend (MPS)")
    #     except Exception as e:
    #         print(f"torch.compile not available: {e}")
    # else:
    #     hsm_model = torch.compile(hsm_model)
    #     print("Model compiled with default backend")
    #
    # hsm_optimizer = optim.AdamW(hsm_model.parameters(), lr=HSM_LEARNING_RATE)
    #
    # for epoch in range(EPOCHS):
    #     print(f"\n--- HSM Epoch {epoch + 1}/{EPOCHS} ---")
    #     train_epoch(hsm_model, train_dataloader, hsm_optimizer, device)
    #
    # # Save HSM model
    # torch.save(hsm_model.state_dict(), HSM_MODEL_PATH)
    # print(f"HSM model saved to {HSM_MODEL_PATH}")

    # load and run stuff
    print("\n" + "=" * 60)
    print("LOADING: HSM Model from saved checkpoint")
    print("=" * 60)

    hsm_model = GPT2HierarchicalSoftmaxModel(
        root, tokenizer, hidden_size=768, model_name="gpt2"
    )

    # Load state dict to CPU first (avoids MPS memory alignment issues)
    state_dict = torch.load(HSM_MODEL_PATH, map_location="cpu", weights_only=True)

    # Strip "_orig_mod." prefix if model was saved after torch.compile()
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("_orig_mod."):
            new_key = key[len("_orig_mod.") :]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value

    hsm_model.load_state_dict(new_state_dict)
    hsm_model.to(device)
    print(f"Model loaded from {HSM_MODEL_PATH}")

    # test hsm stuff
    print("\n" + "=" * 60)
    print("TESTING: HSM Generation Methods")
    print("=" * 60)

    hsm_model.eval()

    test_prompt = "preference for grassland rather"
    input_ids = tokenizer.encode(test_prompt, return_tensors="pt").to(device)
    print(f"\nTest prompt: '{test_prompt}'")
    print(f"Input shape: {input_ids.shape}")

    # Test greedy generation
    print("\n--- Greedy Generation ---")
    output = hsm_generate(
        hsm_model,
        input_ids,
        generation_config=GenerationConfig(max_new_tokens=5),
        generate_method=partial(hsm_greedy_predict, model=hsm_model),
    )
    print(f"Generated: {tokenizer.decode(output[0], skip_special_tokens=True)}")

    # Test with different temperatures
    print("\n--- Greedy with Temperature ---")
    for temp in [0.1, 1.0, 2.0]:
        output = hsm_generate(
            hsm_model,
            input_ids.clone(),
            generation_config=GenerationConfig(max_new_tokens=5),
            generate_method=partial(
                hsm_greedy_predict, model=hsm_model, temperature=temp
            ),
        )
        print(f"  temp={temp}: {tokenizer.decode(output[0], skip_special_tokens=True)}")

    # Test top-k
    print("\n--- Top-k Sampling ---")
    output = hsm_generate(
        hsm_model,
        input_ids.clone(),
        generation_config=GenerationConfig(max_new_tokens=5),
        generate_method=partial(hsm_top_k_predict, model=hsm_model, k=5),
    )
    print(f"Generated: {tokenizer.decode(output[0], skip_special_tokens=True)}")

    # Test top-p
    print("\n--- Top-p (Nucleus) Sampling ---")
    output = hsm_generate(
        hsm_model,
        input_ids.clone(),
        generation_config=GenerationConfig(max_new_tokens=5),
        generate_method=partial(hsm_top_p_predict, model=hsm_model, p=0.9),
    )
    print(f"Generated: {tokenizer.decode(output[0], skip_special_tokens=True)}")

    # Test beam search
    print("\n--- Beam Search ---")
    output = hsm_generate(
        hsm_model,
        input_ids.clone(),
        generation_config=GenerationConfig(max_new_tokens=5),
        generate_method=partial(
            hsm_beam_predict, model=hsm_model, num_beams=5, beam_depth=1
        ),
    )
    print(f"Generated: {tokenizer.decode(output[0], skip_special_tokens=True)}")

    # regular softmax stuff
    # print("\n" + "=" * 60)
    # print("TRAINING: GPT-2 with Regular Softmax (Baseline)")
    # print("=" * 60)
    #
    # config = GPT2Config.from_pretrained("gpt2")
    # sm_model = GPT2RegularSoftmaxModel(config, tokenizer.vocab_size)
    # sm_model.to(device)
    #
    # sm_optimizer = optim.AdamW(sm_model.parameters(), lr=SM_LEARNING_RATE)
    #
    # for epoch in range(EPOCHS):
    #     print(f"\n--- SM Epoch {epoch + 1}/{EPOCHS} ---")
    #     train_epoch(sm_model, train_dataloader, sm_optimizer, device)
    #
    # # Save SM model
    # torch.save(sm_model.state_dict(), SM_MODEL_PATH)
    # print(f"SM model saved to {SM_MODEL_PATH}")

    # test regular softmax stuff
    # print("\n" + "=" * 60)
    # print("TESTING: Regular Softmax Generation")
    # print("=" * 60)
    #
    # sm_model_loaded = GPT2RegularSoftmaxModel(config, tokenizer.vocab_size)
    # sm_model_loaded.load_state_dict(
    #     torch.load(SM_MODEL_PATH, map_location=device, weights_only=True)
    # )
    # sm_model_loaded.to(device)
    # sm_model_loaded.eval()
    #
    # test_prompt_sm = "He also played for the Fairfield"
    # input_ids_sm = tokenizer.encode(test_prompt_sm, return_tensors="pt").to(device)
    # print(f"\nTest prompt: '{test_prompt_sm}'")
    #
    # output = sm_model_loaded.generate(
    #     input_ids_sm,
    #     generation_config=GenerationConfig(max_new_tokens=5),
    #     tokenizer=tokenizer,
    # )
    # print(f"Generated: {tokenizer.decode(output[0], skip_special_tokens=True)}")

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
