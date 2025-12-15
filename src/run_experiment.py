#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main experiment runner for HSM experiments.

Usage:
    python -m src.run_experiment --config src/configs/run_1.yaml
    python -m src.run_experiment --config src/configs/run_1.yaml --device cuda
    python -m src.run_experiment --all  # Run all 8 experiments
"""

import argparse
from pathlib import Path

import torch
import torch.optim as optim
import yaml

from .eval import (
    ExperimentConfig,
    count_parameters,
    create_results,
    measure_inference_time,
    print_results_summary,
    save_results,
)
from .model import GPT2WithHSM
from .train import train, validate, TrainingMetrics
from .utils import build_tree, get_tokenizer, load_wikitext_data


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_device(device_str: str = None) -> torch.device:
    """Get the best available device."""
    if device_str:
        return torch.device(device_str)
    
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def run_experiment(config_path: Path, device: torch.device, output_dir: Path):
    """
    Run a single experiment from config file.
    
    Args:
        config_path: Path to YAML config
        device: Device to run on
        output_dir: Directory for outputs (checkpoints, results)
    """
    print(f"\n{'='*70}")
    print(f"Running experiment: {config_path.name}")
    print(f"Device: {device}")
    print(f"{'='*70}\n")
    
    # Load config
    config = load_config(config_path)
    run_name = config["run_name"]
    model_config = config["model"]
    train_config = config["training"]
    data_config = config["data"]
    
    # Create experiment config object
    exp_config = ExperimentConfig(
        run_name=run_name,
        pretrained=model_config["pretrained"],
        freeze_transformer=model_config["freeze_transformer"],
        use_prf=model_config["use_prf"],
        num_random_features=model_config.get("num_random_features", 256),
        epochs=train_config["epochs"],
        batch_size=train_config["batch_size"],
        learning_rate=train_config["learning_rate"],
        train_samples=data_config["train_samples"],
        val_samples=data_config["val_samples"],
        max_seq_len=data_config["max_seq_len"],
    )
    
    # Initialize tokenizer
    print("Initializing tokenizer...")
    tokenizer = get_tokenizer()
    vocab = tokenizer.get_vocab()
    print(f"Vocabulary size: {len(vocab)}")
    
    # Build Huffman tree (uniform frequencies)
    print("\nBuilding Huffman tree...")
    root = build_tree(list(vocab.values()), [1] * len(vocab))
    
    # Load data
    print("\nLoading data...")
    train_loader, val_loader, test_loader, _ = load_wikitext_data(
        tokenizer,
        max_seq_len=data_config["max_seq_len"],
        batch_size=train_config["batch_size"],
        train_samples=data_config["train_samples"],
        val_samples=data_config["val_samples"],
        test_samples=data_config.get("test_samples", 100),
    )
    
    # Create model
    print("\nCreating model...")
    model = GPT2WithHSM(
        root=root,
        tokenizer=tokenizer,
        hidden_size=768,
        pretrained=model_config["pretrained"],
        freeze_transformer=model_config["freeze_transformer"],
        use_prf=model_config["use_prf"],
        num_random_features=model_config.get("num_random_features", 256),
    )
    model.to(device)
    
    # Count parameters
    param_counts = count_parameters(model)
    print(f"\nModel parameters:")
    print(f"  Total: {param_counts['total']:,}")
    print(f"  Trainable: {param_counts['trainable']:,}")
    print(f"  HSM head: {param_counts['hsm']:,}")
    
    # Create optimizer (only for trainable params)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=train_config["learning_rate"],
    )
    
    # Create output directories
    run_output_dir = output_dir / run_name
    checkpoint_dir = run_output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Train
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    
    metrics = train(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=train_config["epochs"],
        save_dir=checkpoint_dir,
        save_prefix=run_name,
    )
    
    # Final validation
    print("\nFinal validation...")
    final_val_loss = validate(model, val_loader, device)
    
    # Measure inference time
    print("\nMeasuring inference time...")
    inference_time = measure_inference_time(
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompt="The quick brown fox",
        num_tokens=20,
        num_runs=5,
    )
    print(f"Inference time per token: {inference_time * 1000:.2f}ms")
    
    # Create and save results
    results = create_results(
        config=exp_config,
        train_metrics=metrics,
        val_loss=final_val_loss,
        inference_time=inference_time,
        param_counts=param_counts,
    )
    
    results_path = run_output_dir / "results.json"
    save_results(results, results_path)
    
    # Print summary
    print_results_summary(results)
    
    return results


def run_all_experiments(device: torch.device, output_dir: Path):
    """Run all 8 experiments."""
    config_dir = Path(__file__).parent / "configs"
    config_files = sorted(config_dir.glob("run_*.yaml"))
    
    print(f"\n{'='*70}")
    print(f"RUNNING ALL {len(config_files)} EXPERIMENTS")
    print(f"{'='*70}\n")
    
    all_results = []
    
    for config_path in config_files:
        try:
            results = run_experiment(config_path, device, output_dir)
            all_results.append(results)
        except Exception as e:
            print(f"\nERROR in {config_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print comparison
    if all_results:
        from .eval import compare_results
        compare_results(all_results)
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run HSM experiments")
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all 8 experiments",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda, mps, cpu). Auto-detected if not specified.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Output directory for results and checkpoints",
    )
    
    args = parser.parse_args()
    
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    if args.all:
        run_all_experiments(device, args.output_dir)
    elif args.config:
        run_experiment(args.config, device, args.output_dir)
    else:
        parser.print_help()
        print("\nError: Must specify either --config or --all")


if __name__ == "__main__":
    main()

